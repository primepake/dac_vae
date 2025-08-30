# Descript Audio Codec - VAE Variant (.dac-vae): High-Fidelity Audio Compression with Variational Autoencoder

This repository contains training and inference scripts for the Descript Audio Codec VAE variant (.dac-vae), a modified version of the [original DAC](https://github.com/descriptinc/descript-audio-codec) that replaces the RVQGAN architecture with a Variational Autoencoder while maintaining the same high-quality audio compression capabilities.

## Overview

Building on the foundation of the [original Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec), **DAC-VAE** adapts the architecture to use Variational Autoencoder principles instead of Residual Vector Quantization (RVQ).

### Key Differences from Original DAC

👉 **DAC-VAE** compresses **24 kHz audio** (instead of 44.1 kHz) using a continuous latent representation through VAE architecture

### 🔄 Architecture Changes:

- Replaces the RVQGAN's discrete codebook with VAE's continuous latent space
- Maintains the same encoder-decoder backbone architecture from the original DAC
- Swaps vector quantization layers for VAE reparameterization trick
- Preserves the multi-scale discriminator design for adversarial training

### 🎯 Inherited Features from Original DAC:

- High-fidelity neural audio compression
- Universal model for all audio domains (speech, environment, music, etc.)
- Efficient encoding and decoding
- State-of-the-art reconstruction quality

## Why VAE Instead of RVQGAN?

This fork explores an alternative approach to the original DAC's discrete coding strategy:

| Component | Original DAC (RVQGAN) | DAC-VAE (This Repo) |
|-----------|----------------------|---------------------|
| Latent Space | Discrete (VQ codes) | Continuous (Gaussian) |
| Sampling Rate | 44.1 kHz | 24 kHz |
| Quantization | Residual VQ with codebooks | VAE reparameterization |
| Training Objective | Reconstruction + VQ + Adversarial | Reconstruction + KL + Adversarial |
| Compression | Fixed bitrate (8 kbps) | Variable (KL-controlled) |

## Installation

```bash
# Clone this repository
git clone https://github.com/primepake/dac-vae.git
cd dac-vae

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Inference

```bash
python3 inference.py \
    --checkpoint checkpoint.pt \
    --config configs/configx2.yml \
    --mode encode_decode \
    --input test.wav \
    --output reconstruction.wav
```

### Training

```bash
# Single GPU training
python3 train.py --run_id factorx2

# Multi-GPU training (4 GPUs)
torchrun --nnodes=1 --nproc_per_node=4 train.py --run_id factorx2
```
## Model Architecture

DAC-VAE preserves most of the original DAC architecture with key modifications:

- **Encoder**: Same convolutional architecture as original DAC
- **Latent Layer**: VAE reparameterization (replaces VQ-VAE quantization)
- **Decoder**: Identical transposed convolution architecture  
- **Discriminator**: Same multi-scale discriminator for perceptual quality

### Configuration

The model can be configured through YAML files in the `configs/` directory:

- `configx2.yml`: Default 24kHz configuration with 2x downsampling factor
- Adjust latent dimensions, KL weight, and other hyperparameters as needed

## Training Details

### Dataset Preparation

Prepare your audio dataset with the following structure:
```
dataset/
├── train/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── val/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

### Training Command

```bash
torchrun --nnodes=1 --nproc_per_node=4 train.py \
    --run_id my_experiment \
    --config configs/configx2.yml
```

## Evaluation

Evaluate model performance using:

```bash
python3 evaluate.py \
    --checkpoint checkpoint.pt \
    --test_dir /path/to/test/audio
```

## Pretrained Models

| Model | Sample Rate | Config | Download |
|-------|-------------|---------|----------|
| dac_vae_24khz_v1 | 24 kHz | config.yml | [64 dim 3x frames](#) |
| dac_vae_24khz_v1 | 24 kHz | configx2.yml | [80 dim 2x frames](#) |


## Citation

If you use DAC-VAE, please cite both this work and the original DAC paper:

```bibtex
@misc{dacvae2024,
  title={DAC-VAE: Variational Autoencoder Adaptation of Descript Audio Codec},
  author={primepake},
  year={2024},
  url={https://github.com/primepake/dac-vae}
}

@misc{kumar2023high,
  title={High-Fidelity Audio Compression with Improved RVQGAN},
  author={Kumar, Rithesh and Seetharaman, Prem and Luebs, Alejandro and Kumar, Ishaan and Kumar, Kundan},
  journal={arXiv preprint arXiv:2306.06546},
  year={2023}
}
```

## License

This project maintains the same license as the original Descript Audio Codec. See [LICENSE](LICENSE) file for details.

## Acknowledgments

This work is built directly on top of the excellent [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) by the Descript team. We thank them for open-sourcing their high-quality implementation, which made this VAE exploration possible.

## Related Links

- [Original DAC Repository](https://github.com/descriptinc/descript-audio-codec)
- [Original DAC Paper](https://arxiv.org/abs/2306.06546)
- [Descript Audio Codec Demo](https://descript.notion.site/Descript-Audio-Codec-11389fce0ce2419891d6591a18f30bfd)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and feedback, please open an issue in this repository.