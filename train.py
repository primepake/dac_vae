from comet_ml import Experiment
import argparse
import os
import time
import typing
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Union

import numpy as np
import soundfile as sf
import torch
import yaml

from model import Discriminator
from model import DACVAE as VAE
from loss import (GANLoss, L1Loss, MelSpectrogramLoss,
                         MultiScaleSTFTLoss, kl_loss)
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR
from torch.utils.data.distributed import DistributedSampler

from audiotools import AudioSignal
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset, AudioLoader, ConcatDataset
from audiotools.ml.decorators import Tracker, timer, when


def ddp_setup():
    print("Setting up DDP")
    init_process_group(backend="nccl", timeout=timedelta(seconds=7200))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def build_transform(
    augment_prob=1.0,
    preprocess=["Identity"],
    augment=["Identity"],
    postprocess=["Identity", "RescaleAudio", "ShiftPhase"],
):
    to_tfm = lambda l: [getattr(transforms, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    return transforms.Compose(preprocess, augment, postprocess)


def build_dataset(sample_rate, folders=None, **kwargs):
    if folders is None:
        folders = {}
    datasets = []
    for _, v in folders.items():
        loader = AudioLoader(sources=v)
        transform = build_transform()
        dataset = AudioDataset(
            loader, sample_rate, num_channels=2, transform=transform, **kwargs
        )
        datasets.append(dataset)
    dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset


@dataclass
class State:
    generator: DDP
    optimizer_g: Union[AdamW, Adam]
    scheduler_g: torch.optim.lr_scheduler._LRScheduler

    discriminator: DDP
    optimizer_d: Union[AdamW, Adam]
    scheduler_d: torch.optim.lr_scheduler._LRScheduler

    stft_loss: MultiScaleSTFTLoss
    mel_loss: MelSpectrogramLoss
    gan_loss: GANLoss
    waveform_loss: L1Loss

    train_dataset: AudioDataset
    val_dataset: AudioDataset

    tracker: Tracker
    lambdas: Dict[str, float]

    # ema: EMA  # Add EMA to State


class ResumableDistributedSampler(DistributedSampler):  # pragma: no cover
    """Distributed sampler that can be resumed from a given start index."""

    def __init__(self, dataset, start_idx: int = 0, **kwargs):
        super().__init__(dataset, **kwargs)
        # Start index, allows to resume an experiment at the index it was
        self.start_idx = start_idx // self.num_replicas if start_idx is not None else 0

    def __iter__(self):
        for i, idx in enumerate(super().__iter__()):
            if i >= self.start_idx:
                yield idx
        self.start_idx = 0  # set the index back to 0 so for the next epoch


def prepare_dataloader(
    dataset: AudioDataset,
    world_size: int,
    local_rank: int,
    start_idx: int = 0,
    shuffle: bool = True,
    **kwargs,
):
    # sampler = ResumableDistributedSampler(
    #     dataset,
    #     start_idx,
    #     num_replicas=world_size,
    #     rank=local_rank,
    #     shuffle=shuffle,
    # )

    sampler = None
    if start_idx > 0:
        # Create a simple resumable sampler
        indices = list(range(start_idx, len(dataset))) + list(range(start_idx))
        sampler = torch.utils.data.SubsetRandomSampler(indices)

    # if "num_workers" in kwargs:
    #     kwargs["num_workers"] = max(kwargs["num_workers"] // world_size, 1)
    # kwargs["batch_size"] = max(kwargs["batch_size"] // world_size, 1)
    # dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, **kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        shuffle=(sampler is None),  # Only shuffle if no sampler
        num_workers=24,  # Can use more workers since no distribution
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,  # Can be higher for single GPU
        drop_last=True,
        **kwargs
    )
    return dataloader


class Trainer:
    def __init__(self, args) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(self.local_rank)
        torch.cuda.empty_cache()

        configs = yaml.safe_load(open(args.config_path, "r"))
        print("configs: ", configs)
        self.configs = configs

        self.gan_start_step = configs.get("gan_start_step", 0)

        self.num_iters = configs.get("num_iters", 250000)

        self.generator = VAE(**configs["vae"])

        self.discriminator = Discriminator(**configs["discriminator"])

        total_steps = configs["num_samples"] // configs["batch_size"]

        if configs["optimizer"]["scheduler"] == "linearlr":
            self.optimizer_g, self.scheduler_g = self.get_scheduler(
                self.generator, total_steps, configs["optimizer"]
            )
        else:
            self.optimizer_g, self.scheduler_g = self.get_constant_scheduler(
                self.generator, total_steps
            )

        if configs["disc_optimizer"]["scheduler"] == "constantlr":
            self.optimizer_d, self.scheduler_d = self.get_constant_scheduler(
                self.discriminator, total_steps
            )
        else:
            self.optimizer_d, self.scheduler_d = self.get_scheduler(
                self.discriminator, total_steps, configs["disc_optimizer"]
            )

        save_path = args.save_path
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path

        if self.local_rank == 0:
            print(f"Rank {self.local_rank}: Initializing Comet.ml")
            experiment = Experiment(
                api_key=os.environ.get(
                    "COMET_API_KEY"
                ),  # Set COMET_API_KEY in your environment
                project_name="DACVAE",
                workspace=os.environ.get("COMET_WORKSPACE"),  # Optional: Set workspace
                # experiment_key=args.run_id,  # Use run_id as experiment key
            )
            experiment.log_parameters(configs)  # Log configuration
            writer = experiment
        else:
            writer = None

        print(f"Rank {self.local_rank}: Setting up tracker")
        self.tracker = Tracker(
            writer=writer, log_file=f"{save_path}/log.txt", rank=self.local_rank
        )
        self.val_idx = configs.get("val_idx", [0, 1, 2, 3, 4, 5, 6, 7])
        self.save_iters = configs.get("save_iters", 1000)
        self.sample_freq = configs.get("sample_freq", 10000)
        self.valid_freq = configs.get("valid_freq", 1000)

        self.tracker.print(self.generator)
        self.tracker.print(self.discriminator)

        self.waveform_loss = L1Loss()
        self.stft_loss = MultiScaleSTFTLoss(**configs["MultiScaleSTFTLoss"])
        self.mel_loss = MelSpectrogramLoss(**configs["MelSpectrogramLoss"])

        print(f"{self.global_rank} Loading datasets...")
        sample_rate = configs["vae"]["sample_rate"]
        train_folders = {k: v for k, v in configs.get("train_folders", {}).items()}
        val_folders = {k: v for k, v in configs.get("val_folders", {}).items()}
        self.batch_size = configs["batch_size"]
        self.val_batch_size = configs["val_batch_size"]
        self.num_workers = configs["num_workers"]

        print(f"Rank {self.local_rank}: Validating train dataset")
        self.train_dataset = build_dataset(
            sample_rate, train_folders, **configs["train"]
        )
        print(f"Rank {self.local_rank}: Validating val dataset")
        self.val_dataset = build_dataset(sample_rate, val_folders, **configs["val"])

        self.lambdas = configs["lambdas"]

        if args.resume:
            checkpoint_dir = os.path.join(args.save_path, args.tag)
            self.resume_from_checkpoint(checkpoint_dir)

        self.gan_loss = GANLoss(self.discriminator)
        print("self.tracker.step: ", self.tracker.step)

        self.generator = self.generator.to(self.local_rank)
        self.discriminator = self.discriminator.to(self.local_rank)
        #
        self.generator = nn.SyncBatchNorm.convert_sync_batchnorm(self.generator)
        self.discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)

        # Wrap models with DDP
        self.generator = DDP(self.generator, device_ids=[self.local_rank])
        self.discriminator = DDP(self.discriminator, device_ids=[self.local_rank])

        # ema_decay = self.configs.get("ema_decay", 0.999)  # Add to your config YAML or set default
        # self.ema = EMA(self.unwrap(self.generator), decay=ema_decay, device=self.local_rank)

        self.state = State(
            generator=self.generator,
            optimizer_g=self.optimizer_g,
            scheduler_g=self.scheduler_g,
            discriminator=self.discriminator,
            optimizer_d=self.optimizer_d,
            scheduler_d=self.scheduler_d,
            tracker=self.tracker,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            stft_loss=self.stft_loss.to(self.local_rank),
            mel_loss=self.mel_loss.to(self.local_rank),
            gan_loss=self.gan_loss.to(self.local_rank),
            waveform_loss=self.waveform_loss.to(self.local_rank),
            lambdas=self.lambdas,
            # ema=self.ema,  # Add EMA to state
        )
        train_dataloader = prepare_dataloader(
            self.train_dataset,
            world_size=self.world_size,
            local_rank=self.local_rank,
            start_idx=self.state.tracker.step,  # Use step directly
            batch_size=self.batch_size,
            collate_fn=self.state.train_dataset.collate,
        )

        self.len_train = len(train_dataloader)

        self.train_dataloader = self.get_infinite_loader(train_dataloader)

        if self.global_rank == 0:
            self.val_dataloader = prepare_dataloader(
                self.state.val_dataset,
                world_size=1,
                local_rank=0,
                start_idx=0,
                shuffle=False,
                batch_size=self.val_batch_size,
                collate_fn=self.state.val_dataset.collate,
            )

        self.seed = 0
        self.val_real_audio = []
        self.val_gen_audio = []
        self.initial_norm = configs.get("initial_norm", float("inf"))
        self.max_norm = configs.get("max_norm", float("inf"))
        self.initial_norm_d = configs.get("initial_norm_d", float("inf"))
        self.max_norm_d = configs.get("max_norm_d", float("inf"))

        self.init_logs_penalty = self.state.lambdas["logs_penalty"]
        self.init_lipschitz_penalty = self.state.lambdas["lipschitz_penalty"]
        self.kl_max_beta = self.state.lambdas["kl/loss"]
        self.hold_base_steps = configs.get("hold_base_steps", 200000)

    def get_scheduler(self, model, total_steps, configs):
        warmup_steps = configs.get("warmup_steps", 0)
        if configs["type"] == "Adamw":
            optimizer = AdamW(
                model.parameters(),
                lr=configs["lr"],
                weight_decay=configs["weight_decay"],
            )
        else:
            optimizer = Adam(
                model.parameters(),
                lr=configs["lr"],
                weight_decay=configs["weight_decay"],
            )

        # Warmup from near-zero to max_lr
        warmup = LinearLR(
            optimizer,
            start_factor=1e-9,
            end_factor=1.0,  # Go up to max_lr
            total_iters=warmup_steps,
        )
        remaining_iters = total_steps - warmup_steps
        constant = ConstantLR(
            optimizer,
            factor=1.0,  # Keep the learning rate constant at max_lr
            total_iters=remaining_iters,
        )

        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, constant], milestones=[warmup_steps]
        )
        return optimizer, scheduler

    def get_constant_scheduler(self, model, total_steps):
        if self.configs["optimizer"]["type"] == "adamw":
            optimizer = AdamW(
                model.parameters(),
                lr=self.configs["optimizer"]["lr"],
                weight_decay=self.configs["optimizer"]["weight_decay"],
            )
        else:
            optimizer = Adam(
                model.parameters(),
                lr=self.configs["optimizer"]["lr"],
                weight_decay=self.configs["optimizer"]["weight_decay"],
            )
        scheduler = ConstantLR(
            optimizer,
            factor=1.0,  # Keep the learning rate constant at max_lr
            total_iters=total_steps,
        )
        return optimizer, scheduler

    def get_infinite_loader(self, dataset):
        print(
            f"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}: Starting infinite loader"
        )
        # Skip iterations if resuming
        iterator = iter(dataset)
        steps_to_skip = self.state.tracker.step
        while True:
            try:
                batch = next(iterator)
                if batch is None:
                    print(f"Rank {torch.distributed.get_rank()}: Skipping None batch")
                    continue
                yield batch
            except StopIteration:
                iterator = iter(dataset)  # Reset iterator at the end of the dataset

    def log_grad_norms(self, output, norm_threshold=1.0):
        """
        Log gradient norms for key DACVAE components to aid debugging.
        Tracks pre-clipping norms for encoder, decoder, and selected blocks.

        Args:
            output (dict): Dictionary to store gradient norm logs.
            norm_threshold (float): Log norms above this threshold to reduce noise.
        """
        # Initialize dictionaries for norms
        submodule_norms = {
            "en_conv_post": 0.0,
            "de_conv_pre": 0.0,
            "encoder_initial_conv": 0.0,
            "encoder_final_conv": 0.0,
            "encoder_snake1d_alpha": 0.0,
            "decoder_initial_conv": 0.0,
            "decoder_final_conv": 0.0,
            "decoder_snake1d_alpha": 0.0,
        }
        norm_values = []  # For distributional statistics

        # Initialize norms for a few representative blocks (e.g., first and last)
        num_enc_blocks = len(self.state.generator.module.encoder_rates)
        num_dec_blocks = len(self.state.generator.module.decoder_rates)
        for i in [0, num_enc_blocks - 1]:  # First and last encoder blocks
            submodule_norms.update(
                {
                    f"encoder_block_{i}": 0.0,
                    f"encoder_block_{i}_snake1d": 0.0,
                    f"encoder_block_{i}_conv1d": 0.0,
                }
            )
        for i in [0, num_dec_blocks - 1]:  # First and last decoder blocks
            submodule_norms.update(
                {
                    f"decoder_block_{i}": 0.0,
                    f"decoder_block_{i}_snake1d": 0.0,
                    f"decoder_block_{i}_conv_transpose": 0.0,
                }
            )

        # Calculate indices for final layers
        enc_final_conv_idx = num_enc_blocks + 2
        dec_final_conv_idx = num_dec_blocks * 2 + 1

        # Iterate through parameters
        for name, param in self.state.generator.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                norm_values.append(norm)

                # DACVAE layers
                if "en_conv_post" in name:
                    submodule_norms["en_conv_post"] += norm**2
                elif "de_conv_pre" in name:
                    submodule_norms["de_conv_pre"] += norm**2

                # Encoder components
                if "encoder.block.0" in name:
                    submodule_norms["encoder_initial_conv"] += norm**2
                elif f"encoder.block.{enc_final_conv_idx}" in name:
                    submodule_norms["encoder_final_conv"] += norm**2
                elif "encoder" in name and "alpha" in name:
                    submodule_norms["encoder_snake1d_alpha"] += norm**2
                for i in [0, num_enc_blocks - 1]:
                    block_idx = i + 1
                    if f"encoder.block.{block_idx}" in name:
                        submodule_norms[f"encoder_block_{i}"] += norm**2
                        if "block.3" in name:  # Snake1d
                            submodule_norms[f"encoder_block_{i}_snake1d"] += norm**2
                        elif "block.4" in name:  # WNConv1d
                            submodule_norms[f"encoder_block_{i}_conv1d"] += norm**2

                # Decoder components
                if "decoder.model.0" in name:
                    submodule_norms["decoder_initial_conv"] += norm**2
                elif f"decoder.model.{dec_final_conv_idx}" in name:
                    submodule_norms["decoder_final_conv"] += norm**2
                elif "decoder" in name and "alpha" in name:
                    submodule_norms["decoder_snake1d_alpha"] += norm**2
                for i in [0, num_dec_blocks - 1]:
                    block_idx = i * 2 + 1
                    if f"decoder.model.{block_idx}" in name:
                        submodule_norms[f"decoder_block_{i}"] += norm**2
                        if "block.0" in name:  # Snake1d
                            submodule_norms[f"decoder_block_{i}_snake1d"] += norm**2
                        elif "block.1" in name:  # WNConvTranspose1d
                            submodule_norms[f"decoder_block_{i}_conv_transpose"] += (
                                norm**2
                            )

        # Compute square root of summed norms and log if above threshold
        for key in submodule_norms:
            norm = submodule_norms[key] ** 0.5
            if norm > norm_threshold:
                output[f"grad_norm/{key}"] = norm

        # Log pre-clipping norm statistics
        if norm_values:
            output["grad_norm/pre_clip_max"] = max(norm_values)
            output["grad_norm/pre_clip_mean"] = sum(norm_values) / len(norm_values)
            output["grad_norm/pre_clip_95th_percentile"] = (
                torch.tensor(norm_values).quantile(0.95).item()
            )

    def compute_lipschitz_penalty(self, lambda_lip=0.01):
        penalty = 0.0
        for name, param in self.state.generator.named_parameters():
            if (
                ("decoder" in name or "de_conv_pre" in name)
                and param.grad is not None
                and "weight" in name
            ):
                grad_norm = param.grad.norm(2)
                penalty += grad_norm**2
        return lambda_lip * penalty

    def compute_gradient_penalty(self, recons, z):
        # Compute gradients of decoder output w.r.t. latents
        grads = torch.autograd.grad(
            outputs=recons,
            inputs=z,
            grad_outputs=torch.ones_like(recons),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_norm = grads.norm(2, dim=[1, 2]).mean()
        return 0.1 * grad_norm  # Weight for penalty

    def cosine_decay_with_warmup(
        self,
        cur_step,
        base_value,
        total_steps,
        final_value,
        warmup_value=0.0,
        warmup_steps=0,
        hold_base_steps=0,
    ):
        """Cosine schedule with warmup, adapted from R3GAN."""
        # Ensure cur_step is a tensor
        cur_step = torch.tensor(cur_step, dtype=torch.float32)

        # Compute decay term
        denom = float(total_steps - warmup_steps - hold_base_steps)
        if denom <= 0:
            raise ValueError(
                "total_steps must be greater than warmup_steps + hold_base_steps"
            )
        phase = torch.pi * (cur_step - warmup_steps - hold_base_steps) / denom
        decay = 0.5 * (1 + torch.cos(phase))

        # Compute current value
        cur_value = base_value + (1 - decay) * (final_value - base_value)

        # Apply hold_base_steps condition
        if hold_base_steps > 0:
            cur_value = torch.where(
                cur_step > warmup_steps + hold_base_steps,
                cur_value,
                torch.tensor(base_value, dtype=torch.float32),
            )

        # Apply warmup_steps condition
        if warmup_steps > 0:
            slope = (base_value - warmup_value) / warmup_steps
            warmup_v = slope * cur_step + warmup_value
            cur_value = torch.where(cur_step < warmup_steps, warmup_v, cur_value)

        # Apply total_steps cap
        cur_value = torch.where(
            cur_step > total_steps,
            torch.tensor(final_value, dtype=torch.float32),
            cur_value,
        )

        return cur_value.item()  # Return as float

    def smooth_increase(
        self,
        step: int,
        initial_beta: float = 0.01,
        final_beta: float = 0.0,
        total_steps: int = 50000,
    ) -> float:
        """Compute a linear decrease for beta."""
        progress = min(step / total_steps, 1.0)
        beta = initial_beta + progress * (final_beta - initial_beta)
        return beta

    @timer()
    def train_loop(self, batch):
        print(f"Rank {self.local_rank}: Starting train_loop")

        self.max_gen_norm = self.cosine_decay_with_warmup(
            cur_step=self.tracker.step,
            base_value=self.initial_norm,  # e.g., 100
            total_steps=self.num_iters,  # e.g., 250000
            final_value=self.max_norm,
            warmup_value=self.initial_norm,
            warmup_steps=0,
            hold_base_steps=self.hold_base_steps,
        )

        self.max_d_norm = self.cosine_decay_with_warmup(
            cur_step=self.tracker.step,
            base_value=self.initial_norm_d,  # e.g., 100
            total_steps=self.num_iters,  # e.g., 250000
            final_value=self.max_norm_d,
            warmup_value=self.initial_norm_d,
            warmup_steps=0,
            hold_base_steps=self.hold_base_steps,
        )

        self.state.generator.train()
        if self.tracker.step >= self.gan_start_step:
            self.state.discriminator.train()
            print(
                f"Rank {self.local_rank}: Discriminator training mode: {self.state.discriminator.training}"
            )
        output = {}
        output = {}
        timing_logs = {}

        output["max_gen_norm"] = self.max_gen_norm
        output["max_d_norm"] = self.max_d_norm

        train_loop_start = time.time()

        # Batch preparation
        batch_prepare_start = time.time()
        batch = util.prepare_batch(batch, self.local_rank)
        timing_logs["batch_prepare"] = time.time() - batch_prepare_start

        # Data transformation
        transform_start = time.time()
        with torch.no_grad():
            signal = self.train_dataset.transform(
                batch["signal"].clone(), **batch["transform_args"]
            )
            signal.audio_data = torch.clamp(signal.audio_data, -1.0, 1.0)
        timing_logs["transform"] = time.time() - transform_start

        # Generator forward
        gen_forward_start = time.time()
        out = self.state.generator(signal.audio_data, signal.sample_rate)
        recons = AudioSignal(out["audio"], signal.sample_rate)
        timing_logs["gen_forward"] = time.time() - gen_forward_start
        z, mu, logs = out["z"], out["mu"], out["logs"]
        z.requires_grad_(True)
        logs_reg = torch.mean(logs.abs())  # Penalize large logs

        output["kl/loss"] = kl_loss(logs, mu)
        output["logs_penalty"] = logs_reg

        kl_beta = self.cosine_decay_with_warmup(
            cur_step=self.tracker.step,
            base_value=self.kl_max_beta,  # e.g., 100
            total_steps=self.num_iters,  # e.g., 250000
            final_value=0.1,  # 0.1,
            warmup_value=self.initial_norm,
            warmup_steps=0,
            hold_base_steps=self.hold_base_steps,
        )

        output["kl/beta"] = kl_beta

        logs_penalty_weight = self.cosine_decay_with_warmup(
            cur_step=self.tracker.step,
            base_value=self.init_logs_penalty,  # Initial weight for logs_penalty
            total_steps=self.num_iters,  # e.g., 250000
            final_value=self.init_logs_penalty
            * 0.01,  # * 0.0001,  # 10% of initial weight
            warmup_value=self.init_logs_penalty,
            warmup_steps=0,
            hold_base_steps=self.hold_base_steps,
        )
        lipschitz_penalty_weight = self.cosine_decay_with_warmup(
            cur_step=self.tracker.step,
            base_value=self.init_lipschitz_penalty,  # Initial weight for lipschitz_penalty
            total_steps=self.num_iters,  # e.g., 250000
            final_value=self.init_lipschitz_penalty
            * 0.01,  # * 0.0001,  # 10% of initial weight
            warmup_value=self.init_lipschitz_penalty,
            warmup_steps=0,
            hold_base_steps=self.hold_base_steps,
        )

        # Discriminator loss
        if self.tracker.step >= self.gan_start_step:
            print(f"Rank {self.local_rank}: Discriminator loss")
            disc_loss_start = time.time()
            output["adv/disc_loss"] = self.state.gan_loss.discriminator_loss(
                recons, signal
            )
            timing_logs["disc_loss"] = time.time() - disc_loss_start

            # Discriminator backward
            disc_backward_start = time.time()
            self.state.optimizer_d.zero_grad(set_to_none=True)
            output["adv/disc_loss"].backward()
            output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
                self.state.discriminator.parameters(), self.max_d_norm
            )
            self.state.optimizer_d.step()
            self.state.scheduler_d.step()
            timing_logs["disc_backward"] = time.time() - disc_backward_start

            # DDP synchronization for discriminator
            disc_ddp_sync_start = time.time()
            # if torch.distributed.is_initialized():
            #     torch.distributed.barrier()
            timing_logs["disc_ddp_sync"] = time.time() - disc_ddp_sync_start
            (
                output["adv/gen_loss"],
                output["adv/feat_loss"],
            ) = self.state.gan_loss.generator_loss(recons, signal)

        # Generator losses
        gen_loss_start = time.time()
        output["stft/loss"] = self.state.stft_loss(recons, signal)

        output["mel/loss"] = self.state.mel_loss(recons, signal)
        output["waveform/loss"] = self.state.waveform_loss(recons, signal)

        output["lipschitz_penalty"] = self.compute_lipschitz_penalty(lambda_lip=0.01)
        output["grad_penalty"] = self.compute_gradient_penalty(recons.audio_data, z)

        loss_keys = [
            "stft/loss",
            "mel/loss",
            "waveform/loss",
            "kl/loss",
            "logs_penalty",
            "lipschitz_penalty",
            "grad_penalty",
        ]
        # print("self.tracker.step >= self.gan_start_step: ", self.tracker.step >= self.gan_start_step)
        if self.tracker.step >= self.gan_start_step:
            loss_keys.extend(["adv/gen_loss", "adv/feat_loss"])

        loss_weights = {k: self.state.lambdas.get(k, 1.0) for k in loss_keys}
        loss_weights["kl/loss"] = kl_beta
        loss_weights["logs_penalty"] = logs_penalty_weight
        loss_weights["lipschitz_penalty"] = lipschitz_penalty_weight

        # log the loss weights
        output.update({f"loss_weight/{k}": v for k, v in loss_weights.items()})

        output["loss"] = sum(
            [loss_weights[k] * output[k] for k in loss_keys if k in output]
        )
        timing_logs["gen_loss"] = time.time() - gen_loss_start

        # Generator backward
        print(f"Rank {self.local_rank}: Updating generator")
        gen_backward_start = time.time()
        self.state.optimizer_g.zero_grad(set_to_none=True)
        output["loss"].backward()

        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.state.generator.module.encoder.parameters(), self.max_gen_norm
        )
        decoder_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.state.generator.module.decoder.parameters(), self.max_gen_norm
        )

        if self.tracker.step % 2 == 0:  # Log every 100 iterations
            self.log_grad_norms(output, norm_threshold=0.0)

        output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
            self.state.generator.parameters(), self.max_gen_norm
        )

        # Log gradient norms
        output["other/grad_norm_encoder"] = (
            encoder_grad_norm.item()
            if torch.is_tensor(encoder_grad_norm)
            else encoder_grad_norm
        )
        output["other/grad_norm_decoder"] = (
            decoder_grad_norm.item()
            if torch.is_tensor(decoder_grad_norm)
            else decoder_grad_norm
        )

        self.state.optimizer_g.step()
        self.state.scheduler_g.step()
        timing_logs["gen_backward"] = time.time() - gen_backward_start

        # self.state.ema.update()

        # DDP synchronization for generator
        gen_ddp_sync_start = time.time()
        # if torch.distributed.is_initialized():
        #     torch.distributed.barrier()
        timing_logs["gen_ddp_sync"] = time.time() - gen_ddp_sync_start

        # Other metrics
        output["other/learning_rate"] = self.state.optimizer_g.param_groups[0]["lr"]
        output["other/batch_size"] = signal.batch_size * self.world_size

        # Total train_loop time
        timing_logs["total_train_loop"] = time.time() - train_loop_start
        output.update({f"time/{k}": v for k, v in timing_logs.items()})

        print(f"Rank {self.local_rank}: train_loop complete")
        return {k: v for k, v in sorted(output.items())}

    def checkpoint(self):
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        step = self.state.tracker.step
        tags = ["latest"]
        if step % self.save_iters == 0:
            tags.append(f"{step // 1000}k")

        self.state.tracker.print(f"Saving checkpoint at step {step}")

        # Prepare everything for saving
        checkpoint = {
            "generator": self.unwrap(self.state.generator).state_dict(),
            "discriminator": self.unwrap(self.state.discriminator).state_dict(),
            "optimizer_g": self.state.optimizer_g.state_dict(),
            "optimizer_d": self.state.optimizer_d.state_dict(),
            "scheduler_g": self.state.scheduler_g.state_dict(),
            "scheduler_d": self.state.scheduler_d.state_dict(),
            "tracker": self.state.tracker.state_dict(),
            # "ema": self.state.ema.state_dict(),  # Save EMA state
            "step": step,
            "config": self.configs,
            "metadata": {
                "logs": self.state.tracker.history,
                "step": step,
                "config": self.configs,
            },
        }

        # Save for each tag (latest, 120k, etc)
        for tag in tags:
            save_folder = f"{self.save_path}/{tag}_{timestamp}"
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, "checkpoint.pt")
            torch.save(checkpoint, save_path)
            self.state.tracker.print(f"Checkpoint saved: {save_path}")

    def resume_from_checkpoint(self, load_folder):
        checkpoint_path = os.path.join(load_folder, "checkpoint.pt")
        assert os.path.exists(
            checkpoint_path
        ), f"Checkpoint {checkpoint_path} not found!"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load model state dicts
        self.unwrap(self.generator).load_state_dict(checkpoint["generator"])
        self.unwrap(self.discriminator).load_state_dict(checkpoint["discriminator"])

        # Load optimizer and scheduler state dicts **after** model is on device
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])

        for state in self.optimizer_g.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.local_rank)
        for state in self.optimizer_d.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.local_rank)

        self.scheduler_g.load_state_dict(checkpoint["scheduler_g"])
        self.scheduler_d.load_state_dict(checkpoint["scheduler_d"])

        # Load EMA state
        # if "ema" in checkpoint:
        #     self.ema.load_state_dict(checkpoint["ema"])

        # Load tracker/logs/step
        self.tracker.load_state_dict(checkpoint["tracker"])
        self.tracker.step = checkpoint.get("step", 0)

        self.tracker.print(
            f"Checkpoint loaded from {checkpoint_path} at step {self.tracker.step}"
        )

    def unwrap(self, model):
        if hasattr(model, "module"):
            return model.module
        return model

    @torch.no_grad()
    def save_samples(self, val_idx):
        print(f"Rank {self.local_rank}: Starting save_samples")
        self.state.tracker.print("Saving audio samples to WandB")
        self.state.generator.eval()

        # Apply EMA weights
        # self.state.ema.apply_shadow()

        samples = [self.val_dataset[idx] for idx in val_idx]
        batch = self.val_dataset.collate(samples)
        batch = util.prepare_batch(batch, self.local_rank)
        signal = self.val_dataset.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )

        out = self.state.generator(signal.audio_data, signal.sample_rate)
        recons = AudioSignal(out["audio"], signal.sample_rate)

        # Restore original weights
        # self.state.ema.restore()

        audio_dict = {"recons": recons}
        # if self.state.tracker.step == 0:
        audio_dict["signal"] = signal

        audio_logs = {}
        for k, v in audio_dict.items():
            for nb in range(v.batch_size):
                audio_data = v[nb].cpu().audio_data
                if audio_data.dim() == 3:
                    audio_data = audio_data.squeeze(0)
                elif audio_data.dim() == 1:
                    audio_data = audio_data.unsqueeze(0)

                audio_data = audio_data.numpy().astype(np.float32)
                if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                    audio_data /= np.abs(audio_data).max()

                sample_rate = int(v[nb].sample_rate)
                if sample_rate <= 0:
                    raise ValueError(f"Invalid sample rate: {sample_rate}")

                # Save audio to a temporary file
                temp_file = f"temp_audio_{k}_{nb}.wav"
                sf.write(temp_file, audio_data.T, sample_rate)

                self.state.tracker.writer.log_audio(
                    temp_file,
                    metadata={
                        "caption": f"{k} sample {nb}",
                        "sample_rate": sample_rate,
                        "step": self.state.tracker.step,
                    },
                    step=self.state.tracker.step,
                )
                
                # Clean up temporary file
                os.remove(temp_file)

    def train(self):
        print(f"Rank {self.local_rank}: Starting train ")
        util.seed(self.seed)

        max_iters = self.num_iters
        train_loop = self.tracker.log("train", "value", history=False)(
            self.tracker.track("train", max_iters, completed=self.state.tracker.step)(
                self.train_loop
            )
        )

        save_samples = when(lambda: self.local_rank == 0)(self.save_samples)
        checkpoint = when(lambda: self.global_rank == 0)(self.checkpoint)

        with self.tracker.live:
            for self.tracker.step, batch in enumerate(
                self.train_dataloader, start=self.state.tracker.step
            ):
                self.tracker.print(
                    f"Rank {self.global_rank}: Iteration {self.tracker.step}/{max_iters} "
                )
                output = train_loop(batch)

                if self.global_rank == 0:
                    for k, v in output.items():
                        value = v.item() if torch.is_tensor(v) else v
                        self.tracker.writer.log_metric(k, value, step=self.tracker.step)

                last_iter = self.tracker.step == max_iters - 1

                if self.tracker.step % self.sample_freq == 0 or last_iter:
                    # torch.distributed.barrier()
                    save_samples(self.val_idx)
                    checkpoint()

                if last_iter:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed DAC training")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yml",
        help="Path to config YAML",
    )
    parser.add_argument("--run_id", type=str, required=True, help="Run ID for wandb")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--load_weights", action="store_true", help="Load weights from checkpoint"
    )
    parser.add_argument(
        "--save_path", type=str, default="ckpts", help="Path to save checkpoints"
    )
    parser.add_argument("--tag", type=str, default="latest", help="Tag for checkpoint")
    args = parser.parse_args()

    ddp_setup()
    trainer = Trainer(args)
    trainer.train()
    destroy_process_group()
