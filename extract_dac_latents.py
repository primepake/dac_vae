# extract_dac_latents.py - With random decoding check

import os
import glob
import argparse
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
import yaml
import json
from collections import defaultdict
import random
import shutil

def process_single_audio(audio_path, model, sample_rate, device):
    """Process a single audio file without padding"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        
        # Convert to tensor [1, 1, T]
        audio_tensor = torch.from_numpy(audio).float()
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        # Normalize
        audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
        
        # Encode
        with torch.no_grad():
            z, mu, logs = model.encode(audio_tensor, sample_rate)
        
        return {
            'success': True,
            'z': z.cpu(),
            'mu': mu.cpu(),
            'logs': logs.cpu(),
            'duration': len(audio) / sample_rate,
            'samples': len(audio),
            'compression_ratio': audio_tensor.shape[-1] // z.shape[-1],
            'original_audio': audio  # Keep original audio for comparison
        }
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return {
            'success': False,
            'error': str(e),
            'path': audio_path
        }


def decode_and_save_sample(model, latent_data, original_audio, audio_path, tmp_dir, device):
    """Decode a latent and save both original and reconstructed audio for comparison"""
    try:
        # Extract info from path
        base_name = os.path.basename(audio_path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Create subdirectory in tmp for this sample
        sample_dir = os.path.join(tmp_dir, name_without_ext)
        os.makedirs(sample_dir, exist_ok=True)
        
        # Decode latent
        z = latent_data['z'].to(device)
        z = z.unsqueeze(0)
        print('z shape: ', z.shape)
        with torch.no_grad():
            reconstructed = model.decode(z)
        
        # Convert to numpy
        reconstructed = reconstructed.squeeze().cpu().numpy()
        if reconstructed.ndim == 2:
            reconstructed = reconstructed[0]
        reconstructed = np.clip(reconstructed, -1.0, 1.0)
        
        # Save original audio
        original_path = os.path.join(sample_dir, f"{name_without_ext}_original.wav")
        sf.write(original_path, original_audio, latent_data['sample_rate'])
        
        # Save reconstructed audio
        reconstructed_path = os.path.join(sample_dir, f"{name_without_ext}_reconstructed.wav")
        sf.write(reconstructed_path, reconstructed, latent_data['sample_rate'])
        
        # Calculate metrics
        min_len = min(len(original_audio), len(reconstructed))
        original_trimmed = original_audio[:min_len]
        reconstructed_trimmed = reconstructed[:min_len]
        
        mse = np.mean((original_trimmed - reconstructed_trimmed) ** 2)
        snr = 10 * np.log10(np.var(original_trimmed) / (mse + 1e-10))
        
        # Save info file
        info = {
            'original_path': audio_path,
            'original_duration': len(original_audio) / latent_data['sample_rate'],
            'reconstructed_duration': len(reconstructed) / latent_data['sample_rate'],
            'latent_shape': latent_data['latent_shape'],
            'compression_ratio': latent_data['compression_ratio'],
            'mse': float(mse),
            'snr': float(snr)
        }
        
        info_path = os.path.join(sample_dir, 'info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Sample saved to {sample_dir} - SNR: {snr:.2f} dB, MSE: {mse:.6f}")
        
        return True, info
        
    except Exception as e:
        print(f"Error decoding sample: {e}")
        return False, {'error': str(e)}


def extract_latents_gpu(rank, world_size, args, audio_files):
    """Extract latents on a single GPU"""
    
    # Setup device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Load DAC model
    from model import DACVAE as VAE
    
    print(f"[GPU {rank}] Loading DAC model...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model = VAE(**config['vae'])
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'generator' in checkpoint:
        model.load_state_dict(checkpoint['generator'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    sample_rate = config['vae']['sample_rate']
    
    # Split files across GPUs
    files_per_gpu = len(audio_files) // world_size
    start_idx = rank * files_per_gpu
    end_idx = start_idx + files_per_gpu if rank < world_size - 1 else len(audio_files)
    gpu_files = audio_files[start_idx:end_idx]
    
    print(f"[GPU {rank}] Processing {len(gpu_files)} files...")
    
    # Create tmp directory for this GPU
    tmp_dir = os.path.join(args.tmp_dir, f'gpu_{rank}')
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Randomly select files for decoding check
    num_samples = min(args.num_decode_samples, len(gpu_files))
    sample_indices = random.sample(range(len(gpu_files)), num_samples)
    
    # Process files one by one
    results = []
    decode_results = []
    
    for idx, audio_path in enumerate(tqdm(gpu_files, desc=f'GPU {rank}', position=rank)):
        # Process single audio
        result = process_single_audio(audio_path, model, sample_rate, device)
        
        if result['success']:
            # Create output path: a/b/c/d.wav -> a/b/c/d_latent2x.pt
            base_path = os.path.splitext(audio_path)[0]  # Remove extension
            output_path = f"{base_path}_latent2x.pt"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Extract data
            z = result['z'].squeeze(0)  # Remove batch dim
            mu = result['mu'].squeeze(0)
            logs = result['logs'].squeeze(0)
            
            # Save as torch tensor
            latent_data = {
                'z': z,
                'mu': mu,
                'logs': logs,
                'sample_rate': sample_rate,
                'compression_ratio': result['compression_ratio'],
                'original_duration': result['duration'],
                'original_samples': result['samples'],
                'latent_shape': list(z.shape),
                'original_path': audio_path
            }
            
            torch.save(latent_data, output_path)
            
            results.append({
                'path': audio_path,
                'output_path': output_path,
                'latent_shape': latent_data['latent_shape'],
                'duration': result['duration'],
                'compression_ratio': result['compression_ratio']
            })
            
            # Check if this is a sample to decode
            if idx in sample_indices:
                print(f"\n[GPU {rank}] Decoding sample {idx}: {os.path.basename(audio_path)}")
                success, decode_info = decode_and_save_sample(
                    model, latent_data, result['original_audio'], 
                    audio_path, tmp_dir, device
                )
                if success:
                    decode_results.append(decode_info)
            
            if rank == 0 and len(results) % 100 == 0:
                print(f"[GPU {rank}] Processed {len(results)} files...")
        else:
            print(f"[GPU {rank}] Failed to process: {audio_path}")
            results.append({
                'path': audio_path,
                'error': result['error'],
                'status': 'failed'
            })
    
    # Save metadata for this GPU
    metadata_path = os.path.join(args.output_dir, f'metadata_gpu{rank}.json')
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save decode results
    if decode_results:
        decode_path = os.path.join(tmp_dir, 'decode_results.json')
        with open(decode_path, 'w') as f:
            json.dump({
                'num_samples': len(decode_results),
                'samples': decode_results,
                'average_snr': np.mean([r['snr'] for r in decode_results if 'snr' in r]),
                'average_mse': np.mean([r['mse'] for r in decode_results if 'mse' in r])
            }, f, indent=2)
    
    print(f"[GPU {rank}] Completed processing {len(results)} files")
    if decode_results:
        avg_snr = np.mean([r['snr'] for r in decode_results if 'snr' in r])
        print(f"[GPU {rank}] Average SNR for decoded samples: {avg_snr:.2f} dB")


def find_audio_files(root_path, extensions=['.wav', '.flac', '.mp3']):
    """Find all audio files in root_path with various structures"""
    audio_files = []
    
    # Check if root_path is a file
    if os.path.isfile(root_path):
        if any(root_path.endswith(ext) for ext in extensions):
            return [root_path]
    
    # Search for audio files
    for ext in extensions:
        # Direct files in root
        audio_files.extend(glob.glob(os.path.join(root_path, f'*{ext}')))
        
        # Recursive search
        audio_files.extend(glob.glob(os.path.join(root_path, '**', f'*{ext}'), recursive=True))
    
    # Remove duplicates and sort
    audio_files = sorted(list(set(audio_files)))
    
    return audio_files


def merge_metadata(output_dir, tmp_dir, world_size):
    """Merge metadata from all GPUs"""
    all_results = []
    failed_files = []
    all_decode_results = []
    
    for rank in range(world_size):
        metadata_path = os.path.join(output_dir, f'metadata_gpu{rank}.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                results = json.load(f)
                for r in results:
                    if 'error' in r:
                        failed_files.append(r)
                    else:
                        all_results.append(r)
            # Remove individual metadata files
            os.remove(metadata_path)
        
        # Load decode results
        decode_path = os.path.join(tmp_dir, f'gpu_{rank}', 'decode_results.json')
        if os.path.exists(decode_path):
            with open(decode_path, 'r') as f:
                decode_data = json.load(f)
                all_decode_results.extend(decode_data['samples'])
    
    # Save merged metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'total_files': len(all_results),
            'failed_files': len(failed_files),
            'files': all_results
        }, f, indent=2)
    
    # Save failed files list if any
    if failed_files:
        failed_path = os.path.join(output_dir, 'failed_files.json')
        with open(failed_path, 'w') as f:
            json.dump(failed_files, f, indent=2)
    
    # Create summary statistics
    total_duration = sum(r['duration'] for r in all_results)
    latent_dims = defaultdict(int)
    compression_ratios = defaultdict(int)
    
    for r in all_results:
        shape_key = str(r['latent_shape'])
        latent_dims[shape_key] += 1
        compression_ratios[r['compression_ratio']] += 1
    
    summary = {
        'total_files': len(all_results),
        'failed_files': len(failed_files),
        'total_duration_hours': total_duration / 3600,
        'latent_dimensions': dict(latent_dims),
        'compression_ratios': dict(compression_ratios),
        'average_duration': total_duration / len(all_results) if all_results else 0,
        'decode_samples': len(all_decode_results)
    }
    
    if all_decode_results:
        summary['average_snr'] = np.mean([r['snr'] for r in all_decode_results if 'snr' in r])
        summary['average_mse'] = np.mean([r['mse'] for r in all_decode_results if 'mse' in r])
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(all_results)} files")
    print(f"Failed: {len(failed_files)} files")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Average duration: {summary['average_duration']:.2f} seconds")
    print(f"Compression ratios: {dict(compression_ratios)}")
    
    if all_decode_results:
        print(f"\nDecode Quality Check:")
        print(f"Samples decoded: {len(all_decode_results)}")
        print(f"Average SNR: {summary['average_snr']:.2f} dB")
        print(f"Average MSE: {summary['average_mse']:.6f}")
        print(f"Check tmp/ folder for audio comparisons")
    
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract DAC latents with multi-GPU support')
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path containing audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save metadata (latents saved alongside audio)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to DAC checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to DAC config')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--file_list', type=str, default=None,
                        help='Optional text file containing list of audio paths')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip files that already have latents')
    parser.add_argument('--tmp_dir', type=str, default='./tmp',
                        help='Directory to save decoded samples for checking')
    parser.add_argument('--num_decode_samples', type=int, default=5,
                        help='Number of random samples to decode per GPU for quality check')
    parser.add_argument('--clean_tmp', action='store_true',
                        help='Clean tmp directory before starting')
    
    args = parser.parse_args()
    
    # Clean tmp directory if requested
    if args.clean_tmp and os.path.exists(args.tmp_dir):
        print(f"Cleaning tmp directory: {args.tmp_dir}")
        shutil.rmtree(args.tmp_dir)
    
    # Create tmp directory
    os.makedirs(args.tmp_dir, exist_ok=True)
    
    # Find audio files
    if args.file_list:
        print(f"Loading file list from {args.file_list}")
        with open(args.file_list, 'r') as f:
            audio_files = [line.strip() for line in f if line.strip()]
    else:
        print(f"Searching for audio files in {args.root_path}")
        audio_files = find_audio_files(args.root_path)
    
    if not audio_files:
        print("No audio files found!")
        return
    
    # Filter out existing if requested
    if args.skip_existing:
        filtered_files = []
        for audio_path in audio_files:
            base_path = os.path.splitext(audio_path)[0]
            latent_path = f"{base_path}_latent2x.pt"

            old_latent_path = f"{base_path}_latent.pt"
            if os.path.exists(old_latent_path):
                os.remove(old_latent_path)
                print(f"Removed old latent file: {old_latent_path}")

            if not os.path.exists(latent_path):
                filtered_files.append(audio_path)
        print(f"Skipping {len(audio_files) - len(filtered_files)} existing files")
        audio_files = filtered_files
    
    print(f"Found {len(audio_files)} audio files to process")
    
    if not audio_files:
        print("No files to process!")
        return
    
    # Create output directory for metadata
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine number of GPUs
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
    
    print(f"Using {args.num_gpus} GPUs")
    print(f"Will decode {args.num_decode_samples} random samples per GPU for quality check")
    
    if args.num_gpus == 1:
        # Single GPU
        extract_latents_gpu(0, 1, args, audio_files)
    else:
        # Multi-GPU
        mp.spawn(
            extract_latents_gpu,
            args=(args.num_gpus, args, audio_files),
            nprocs=args.num_gpus,
            join=True
        )
    
    # Merge metadata
    merge_metadata(args.output_dir, args.tmp_dir, args.num_gpus)


if __name__ == '__main__':
    main()