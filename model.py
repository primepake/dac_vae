import math
from typing import List, Union

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from audiotools import AudioSignal, STFTParams, ml
from audiotools.ml import BaseModel
from base import CodecMixin
from layers import WNConv1d, WNConvTranspose1d, get_activation


def init_weights(m, mean=0.0, std=0.02, init_type="xavier", gain=0.02):
    """
    Initialize weights of the entire model using xavier_normal_ or kaiming_normal_.
    Args:
        m (nn.Module): The module to initialize.
        mean (float): Mean for weight initialization.
        std (float): Standard deviation for weight initialization.
        init_type (str): Type of initialization ('xavier' or 'kaiming').
        gain (float): Gain for xavier initialization.
    """
    classname = m.__class__.__name__

    if init_type == "xavier":
        # Handle convolutional layers
        if "Depthwise_Separable" in classname:
            nn.init.xavier_normal_(m.depth_conv.weight.data, gain=gain)
            nn.init.xavier_normal_(m.point_conv.weight.data, gain=gain)
            if hasattr(m.depth_conv, "bias") and m.depth_conv.bias is not None:
                nn.init.zeros_(m.depth_conv.bias.data)
            if hasattr(m.point_conv, "bias") and m.point_conv.bias is not None:
                nn.init.zeros_(m.point_conv.bias.data)
        elif classname.find("Conv") != -1:
            nn.init.xavier_normal_(m.weight.data, gain=gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias.data)

        # Handle batch normalization layers
        elif classname.find("BatchNorm") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias.data)

        # Handle custom layers like Snake1d and SnakeBeta
        elif classname == "Snake1d":
            if hasattr(m, "alpha") and m.alpha is not None:
                if m.alpha.data.dim() >= 2:
                    nn.init.xavier_normal_(m.alpha.data, gain=gain)
                else:
                    nn.init.normal_(m.alpha.data, mean=1.0, std=std)
        elif classname == "SnakeBeta":
            # Respect the alpha_logscale setting in SnakeBeta
            if hasattr(m, "alpha") and m.alpha is not None:
                if m.alpha_logscale:
                    nn.init.constant_(m.alpha.data, 0.0)  # Matches SnakeBeta's default
                else:
                    nn.init.constant_(m.alpha.data, 1.0)
            if hasattr(m, "beta") and m.beta is not None:
                if m.alpha_logscale:
                    nn.init.constant_(m.beta.data, 0.0)  # Matches SnakeBeta's default
                else:
                    nn.init.constant_(m.beta.data, 1.0)

        # Handle residual scaling parameters
        elif hasattr(m, "residual_scale") and m.residual_scale is not None:
            nn.init.xavier_normal_(m.residual_scale.data, gain=gain)

    else:
        # Kaiming initialization
        if "Depthwise_Separable" in classname:
            nn.init.kaiming_normal_(
                m.depth_conv.weight.data, mode="fan_out", nonlinearity="relu"
            )
            nn.init.kaiming_normal_(
                m.point_conv.weight.data, mode="fan_out", nonlinearity="relu"
            )
        elif classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif classname.find("BatchNorm") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, std)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif classname == "Snake1d":
            if hasattr(m, "alpha") and m.alpha is not None:
                nn.init.normal_(m.alpha.data, 1.0, std)
        elif classname == "SnakeBeta":
            if hasattr(m, "beta") and m.beta is not None:
                nn.init.normal_(m.beta.data, 1.0, std)
            elif (
                hasattr(m, "alpha") and m.alpha is not None
            ):  # Fallback if SnakeBeta uses alpha
                nn.init.normal_(m.alpha.data, 1.0, std)

        elif hasattr(m, "residual_scale") and m.residual_scale is not None:
            nn.init.normal_(m.residual_scale.data, 0.1, std)


class ResidualUnit(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        dilation: int = 1,
        activation: str = "snake",
        alpha: float = 1.0,
        scale_residual: bool = False,
    ):
        """
        Residual Unit with weight normalization and dilated convolutions.
        Args:
            dim (int): Number of input and output channels.
            dilation (int): Dilation factor for the convolution.
            activation (str): Activation function to use.
            alpha (float): Scaling factor for the activation function.
        """
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            get_activation(activation=activation, channels=dim, alpha=alpha),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            get_activation(activation=activation, channels=dim, alpha=alpha),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.scale_residual = scale_residual
        if self.scale_residual:
            self.res_scale = nn.Parameter(torch.tensor(0.0))  # start at 0

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        if self.scale_residual:
            y = self.res_scale * y
        return x + y


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        stride: int = 1,
        activation: str = "snake",
        alpha: float = 1.0,
        scale_residual: bool = False,
    ):
        """
        Encoder block that downsamples the input and applies residual units.
        """
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(
                dim // 2,
                dilation=1,
                activation=activation,
                alpha=alpha,
                scale_residual=scale_residual,
            ),
            ResidualUnit(
                dim // 2,
                dilation=3,
                activation=activation,
                alpha=alpha,
                scale_residual=scale_residual,
            ),
            ResidualUnit(
                dim // 2,
                dilation=9,
                activation=activation,
                alpha=alpha,
                scale_residual=scale_residual,
            ),
            get_activation(activation=activation, channels=dim // 2, alpha=alpha),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        d_in: int = 1,
        activation: str = "snake",
        alpha: float = 1.0,
        scale_residual: bool = False,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(d_in, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [
                EncoderBlock(
                    d_model,
                    stride=stride,
                    activation=activation,
                    alpha=alpha,
                    scale_residual=scale_residual,
                )
            ]

        # Create last convolution
        self.block += [
            get_activation(activation=activation, channels=d_model, alpha=alpha),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        norm: bool = False,
        activation: str = "snake",
        alpha: float = 1.0,
        scale_residual: bool = False,
    ):
        """
        Decoder block that upsamples the input and applies residual units.
        """
        super().__init__()
        if not norm:
            self.block = nn.Sequential(
                get_activation(activation=activation, channels=input_dim, alpha=alpha),
                WNConvTranspose1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    output_padding=0 if stride % 2 == 0 else 1,
                ),
                ResidualUnit(
                    output_dim,
                    dilation=1,
                    activation=activation,
                    alpha=alpha,
                    scale_residual=scale_residual,
                ),
                ResidualUnit(
                    output_dim,
                    dilation=3,
                    activation=activation,
                    alpha=alpha,
                    scale_residual=scale_residual,
                ),
                ResidualUnit(
                    output_dim,
                    dilation=9,
                    activation=activation,
                    alpha=alpha,
                    scale_residual=scale_residual,
                ),
            )
        else:
            self.block = nn.Sequential(
                get_activation(activation=activation, channels=input_dim, alpha=alpha),
                WNConvTranspose1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    output_padding=0 if stride % 2 == 0 else 1,
                ),
                nn.BatchNorm1d(output_dim),
                ResidualUnit(
                    output_dim,
                    dilation=1,
                    activation=activation,
                    alpha=alpha,
                    scale_residual=scale_residual,
                ),
                nn.BatchNorm1d(output_dim),
                ResidualUnit(
                    output_dim,
                    dilation=3,
                    activation=activation,
                    alpha=alpha,
                    scale_residual=scale_residual,
                ),
                nn.BatchNorm1d(output_dim),
                ResidualUnit(
                    output_dim,
                    dilation=9,
                    activation=activation,
                    alpha=alpha,
                    scale_residual=scale_residual,
                ),
            )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        norm: bool = False,
        activation: str = "snake",
        alpha: float = 1.0,
        scale_residual: bool = False,
        use_tanh_as_final: bool = True,
        use_bias_at_final: bool = True,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [
                DecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    norm=norm,
                    activation=activation,
                    alpha=alpha,
                    scale_residual=scale_residual,
                )
            ]

        # Add final conv layer
        layers += [
            get_activation(activation=activation, channels=output_dim, alpha=alpha),
            WNConv1d(
                output_dim, d_out, kernel_size=7, padding=3, bias=use_bias_at_final
            ),
            nn.Tanh() if use_tanh_as_final else nn.Identity(),
        ]
        self.use_tanh_as_final = use_tanh_as_final

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        if not self.use_tanh_as_final:
            x = torch.clamp(
                x, min=-1.0, max=1.0
            )  # Ensure output is within [-1, 1] range
        return x


class DACVAE(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = 64,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        sample_rate: int = 44100,
        d_in: int = 2,
        d_out: int = 2,
        weight_init: str = "xavier",
        norm: bool = False,
        activation: str = "snake",
        alpha: float = 1.0,
        gain: float = 0.02,
        scale_residual: bool = False,
        use_tanh_as_final: bool = True,
        use_bias_at_final: bool = True,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.d_in = d_in
        self.d_out = d_out

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            latent_dim,
            d_in=d_in,
            activation=activation,
            alpha=alpha,
            scale_residual=scale_residual,
        )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            d_out=d_out,
            norm=norm,
            activation=activation,
            alpha=alpha,
            scale_residual=scale_residual,
            use_tanh_as_final=use_tanh_as_final,
            use_bias_at_final=use_bias_at_final,
        )

        self.en_conv_post = WNConv1d(
            self.latent_dim, 2 * self.latent_dim, kernel_size=1
        )

        self.de_conv_pre = WNConv1d(self.latent_dim, self.latent_dim, kernel_size=1)

        self.sample_rate = sample_rate
        self.apply(lambda m: init_weights(m, init_type=weight_init, gain=gain))
        self.step = 0  # Initialize step counter for noise decay

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.en_conv_post.parameters():
            param.requires_grad = False
        print("Encoder and en_conv_post frozen")

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate
        length = audio_data.shape[-1]
        # print(f"Audio length: {length}", "math.ceil(length / self.hop_length) * self.hop_length: ", math.ceil(length / self.hop_length) * self.hop_length)
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length

        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        training: bool = True,
    ):
        x = self.encoder(audio_data)
        x = F.leaky_relu(x)
        x = self.en_conv_post(x)
        print('x shape: ', x.shape)
        m, logs = torch.split(x, self.latent_dim, dim=1)
        logs = torch.clamp(logs, min=-14.0, max=14.0)

        z = m + torch.randn_like(m) * torch.exp(logs)

        return z, m, logs

    def decode(self, z: torch.Tensor):
        z = self.de_conv_pre(z)
        z = self.decoder(z)
        return z

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = 24000,
    ):
        # print(f"Audio data shape: {audio_data.shape}")
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, m, logs = self.encode(audio_data)
        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "mu": m,
            "logs": logs,
        }


def WNConv1d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv1d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MPD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class MSD(nn.Module):
    def __init__(self, rate: int = 1, sample_rate: int = 44100):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                WNConv1d(1, 16, 15, 1, padding=7),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
                WNConv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward(self, x):
        x = AudioSignal(x, self.sample_rate)
        x.resample(self.sample_rate // self.rate)
        x = x.audio_data

        fmap = []

        for l in self.convs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: list = BANDS,
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True,
        )

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = torch.view_as_real(x.stft())
        x = rearrange(x, "b 1 f t c -> (b 1) c t f")
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class Discriminator(ml.BaseModel):
    def __init__(
        self,
        rates: list = [],
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        sample_rate: int = 44100,
        bands: list = BANDS,
        d_in: int = 1,
    ):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
        """
        super().__init__()
        discs = []
        discs += [MPD(p) for p in periods]
        discs += [MSD(r, sample_rate=sample_rate) for r in rates]
        discs += [MRD(f, sample_rate=sample_rate, bands=bands) for f in fft_sizes]
        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps


if __name__ == "__main__":
    disc = Discriminator()
    x = torch.zeros(1, 1, 44100)
    results = disc(x)
    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, r in enumerate(result):
            print(r.shape, r.mean(), r.min(), r.max())
        print()
