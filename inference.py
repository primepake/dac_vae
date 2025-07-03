import argparse
import os
import torch
import yaml
import numpy as np
import soundfile as sf
import librosa
from audiotools import AudioSignal
from model import DACVAE as VAE


class DACVAEInference:
    def __init__(self, checkpoint_path, config_path=None, device='cuda'):
        """
        Initialize DACVAE for inference.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            config_path (str): Path to config YAML (optional, will try to load from checkpoint)
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            raise ValueError("Config not found in checkpoint and no config_path provided")
        
        # Initialize model
        print("Initializing DACVAE model")
        self.model = VAE(**self.config['vae'])
        
        # Load weights
        if 'generator' in checkpoint:
            self.model.load_state_dict(checkpoint['generator'])
        else:
            # Try direct state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get sample rate from config
        self.sample_rate = self.config['vae']['sample_rate']
        print(f"Model loaded successfully. Sample rate: {self.sample_rate} Hz")
    
    @torch.no_grad()
    def encode(self, audio_path):
        """
        Encode an audio file to latent representation.
        
        Args:
            audio_path (str): Path to input audio file
            
        Returns:
            tuple: (z, mu, logs) - latent representation and distribution parameters
        """
        # Load audio with librosa - always converts to mono and resamples
        print(f"Loading audio from {audio_path}")
        import librosa
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        print(f"Audio loaded: shape={audio.shape}, sample_rate={sr}")
        
        # Create tensor - audio is already mono [T]
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        audio_tensor = audio_tensor.to(self.device)
        
        # Normalize to [-1, 1]
        audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
        
        # Encode
        print("Encoding audio...")
        z, mu, logs = self.model.encode(audio_tensor, self.sample_rate)
        
        return z, mu, logs
    
    @torch.no_grad()
    def decode(self, z):
        """
        Decode latent representation back to audio.
        
        Args:
            z (torch.Tensor): Latent representation
            
        Returns:
            np.ndarray: Decoded audio
        """
        print("Decoding latent representation...")
        audio_tensor = self.model.decode(z)
        
        # Convert to numpy
        audio = audio_tensor.squeeze().cpu().numpy()  # Remove batch dim and get [T] or [C, T]
        
        # If multi-channel, take first channel or average
        if audio.ndim == 2:
            audio = audio[0]  # Take first channel, or use audio.mean(axis=0) to average
        
        # Clamp to valid range
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    @torch.no_grad()
    def encode_decode(self, audio_path, output_path=None):
        """
        Full encode-decode pipeline for an audio file.
        
        Args:
            audio_path (str): Path to input audio file
            output_path (str): Path to save output audio (optional)
            
        Returns:
            tuple: (reconstructed_audio, z, mu, logs)
        """
        # Load audio with librosa - always converts to mono and resamples
        print(f"Loading audio from {audio_path}")
        import librosa
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        print(f"Audio loaded: shape={audio.shape}, sample_rate={sr}")
        
        # Create tensor - audio is already mono [T]
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        audio_tensor = audio_tensor.to(self.device)
        
        # Normalize to [-1, 1]
        audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
        
        # Forward pass through model
        print("Processing through DACVAE...")
        audio_tensor = audio_tensor[:, :, :9120]

        print('audio_tensor shape: ', audio_tensor.shape)
        out = self.model(audio_tensor, self.sample_rate)
        
        # Extract outputs
        recons_audio = out['audio'].squeeze(0).cpu().numpy()  # [1, T] or [T]
        if recons_audio.ndim == 2:
            recons_audio = recons_audio.squeeze(0)  # [T]
        z = out['z']
        mu = out['mu']
        logs = out['logs']
        print('z shape: ', z.shape)
        # Clamp output
        recons_audio = np.clip(recons_audio, -1.0, 1.0)
        
        # Save if output path provided
        if output_path:
            print(f"Saving reconstructed audio to {output_path}")
            sf.write(output_path, recons_audio, self.sample_rate)
        
        return recons_audio, z, mu, logs
    
    def get_latent_shape(self):
        """Get the shape of the latent representation for a given audio length."""
        # Create dummy input - mono audio
        dummy_audio = torch.zeros(1, 1, self.sample_rate, device=self.device)  # 1 second mono
        z, _, _ = self.model.encode(dummy_audio, self.sample_rate)
        return z.shape


def main():
    parser = argparse.ArgumentParser(description="DACVAE Audio Inference")
    parser.add_argument('--checkpoint', type=str, required=False, default="/mnt/nvme/ckpts/24khz/364k_20250702_043748/checkpoint.pt",
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default="./config.yml",
                        help='Path to config YAML (optional if config is in checkpoint)')
    parser.add_argument('--input', type=str, required=False, default='./output.wav',
                        help='Path to input audio file')
    parser.add_argument('--output', type=str, default='./test.wav',
                        help='Path to save output audio (default: input_reconstructed.wav)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to run on')
    parser.add_argument('--mode', type=str, default='encode_decode',
                        choices=['encode_decode', 'encode_only', 'decode_only'],
                        help='Inference mode')
    parser.add_argument('--latent_path', type=str, default=None,
                        help='Path to save/load latent representation')
    
    args = parser.parse_args()
    
    # Initialize model
    dac = DACVAEInference(args.checkpoint, args.config, args.device)
    
    # Set default output path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}_reconstructed.wav"
    
    if args.mode == 'encode_decode':
        # Full encode-decode pipeline
        recons_audio, z, mu, logs = dac.encode_decode(args.input, args.output)
        print(f"Reconstruction complete. Output saved to {args.output}")
        print(f"Latent shape: {z.shape}")
        
        # Optionally save latent
        if args.latent_path:
            torch.save({'z': z, 'mu': mu, 'logs': logs}, args.latent_path)
            print(f"Latent representation saved to {args.latent_path}")
    
    elif args.mode == 'encode_only':
        # Encode only
        z, mu, logs = dac.encode(args.input)
        print(f"Encoding complete. Latent shape: {z.shape}")
        
        # Save latent
        if args.latent_path:
            torch.save({'z': z, 'mu': mu, 'logs': logs}, args.latent_path)
            print(f"Latent representation saved to {args.latent_path}")
        else:
            print("Warning: No latent_path specified, latent representation not saved")
    
    elif args.mode == 'decode_only':
        # Decode only
        if not args.latent_path:
            raise ValueError("latent_path must be specified for decode_only mode")
        
        print(f"Loading latent from {args.latent_path}")
        latent_data = torch.load(args.latent_path, map_location=args.device)
        z = latent_data['z'].to(args.device)
        
        audio = dac.decode(z)
        sf.write(args.output, audio, dac.sample_rate)
        print(f"Decoding complete. Output saved to {args.output}")


if __name__ == "__main__":
    main()