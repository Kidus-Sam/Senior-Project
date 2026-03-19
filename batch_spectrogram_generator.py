"""
Signal Processing Pipeline: Batch Mel-Spectrogram Generator
This script converts filtered audio tracks into Mel-spectrogram images.

Based on Technical Details from Workplan:
- FFT window size: 2048
- Hop length: 512
- Mel-scale conversion with log-scale amplitude (dB)
- Output: 224x224 pixel images for CNN input
"""

import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# Signal Processing Parameters (from workplan)
SAMPLE_RATE = 22050          # Standard librosa sampling rate
N_FFT = 2048                 # FFT window size - balances time/frequency resolution
HOP_LENGTH = 512             # Hop length for STFT
N_MELS = 128                 # Number of Mel bands
DURATION = 30.0              # Clip audio to 30 seconds
IMG_SIZE = (224, 224)        # Target image size for CNN

# File paths
PROJECT_DIR = Path("c:/Users/kidus/Desktop/Senior Project 2")
METADATA_FILE = PROJECT_DIR / "fma_metadata" / "fma_metadata" / "tracks_cleaned.csv"
AUDIO_DIR = PROJECT_DIR / "fma_small" / "fma_small"  # Note: nested fma_small directory
OUTPUT_DIR = PROJECT_DIR / "processed_data" / "spectrograms"


def create_output_directory():
    """Create the output directory structure if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}")
    return OUTPUT_DIR


def get_audio_path(track_id):
    """
    Generate the physical path to an FMA audio file.
    Example: Track 123 -> folder 000/000123.mp3
    
    Args:
        track_id: Track ID from metadata
        
    Returns:
        Path object to the audio file
    """
    tid_str = str(int(track_id)).zfill(6)
    folder_name = tid_str[:3]
    return AUDIO_DIR / folder_name / f"{tid_str}.mp3"


def load_and_preprocess_audio(file_path, duration=DURATION, sr=SAMPLE_RATE):
    """
    Load audio file and preprocess to standard duration.
    
    Args:
        file_path: Path to the audio file
        duration: Target duration in seconds
        sr: Target sample rate
        
    Returns:
        Audio time series (numpy array) or None if error
    """
    try:
        # Load audio file
        audio, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)
        
        # Ensure consistent length (pad or trim)
        target_length = int(sr * duration)
        if len(audio) < target_length:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            # Trim if too long
            audio = audio[:target_length]
        
        return audio
    
    except Exception as e:
        print(f"  ✗ Error loading {file_path}: {e}")
        return None


def generate_mel_spectrogram(audio, sr=SAMPLE_RATE, n_fft=N_FFT, 
                             hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Generate a Mel-spectrogram from audio signal.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        n_mels: Number of Mel bands
        
    Returns:
        Mel-spectrogram in dB scale (numpy array)
    """
    # Compute Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=8000  # Maximum frequency
    )
    
    # Convert to log scale (dB) - improves neural network training
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def save_spectrogram_image(mel_spec_db, output_path, img_size=IMG_SIZE):
    """
    Save Mel-spectrogram as a 224x224 pixel image file.
    
    Args:
        mel_spec_db: Mel-spectrogram in dB
        output_path: Path where to save the image
        img_size: Target image dimensions (width, height)
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create figure with exact size
        dpi = 100
        figsize = (img_size[0] / dpi, img_size[1] / dpi)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Display spectrogram without axes or labels
        librosa.display.specshow(
            mel_spec_db,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            x_axis=None,
            y_axis=None,
            ax=ax,
            cmap='viridis'
        )
        
        # Remove all whitespace and axes
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save as PNG
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig)
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error saving spectrogram: {e}")
        plt.close('all')
        return False


def process_single_track(track_id, genre, output_dir):
    """
    Process a single audio track to generate its Mel-spectrogram.
    
    Args:
        track_id: Track ID from metadata
        genre: Genre label for organizing output
        output_dir: Base output directory
        
    Returns:
        Tuple (success: bool, output_path: str or None)
    """
    # Get audio file path
    audio_path = get_audio_path(track_id)
    
    if not audio_path.exists():
        return False, None
    
    # Create genre subdirectory
    genre_dir = output_dir / genre
    genre_dir.mkdir(exist_ok=True)
    
    # Define output path
    output_path = genre_dir / f"{str(int(track_id)).zfill(6)}.png"
    
    # Skip if already processed
    if output_path.exists():
        return True, str(output_path)
    
    # Load and preprocess audio
    audio = load_and_preprocess_audio(audio_path)
    if audio is None:
        return False, None
    
    # Generate Mel-spectrogram
    mel_spec_db = generate_mel_spectrogram(audio)
    
    # Save as image
    success = save_spectrogram_image(mel_spec_db, output_path)
    
    if success:
        return True, str(output_path)
    else:
        return False, None


def batch_process_spectrograms(metadata_path, output_dir, max_tracks=None):
    """
    Batch process all tracks in the cleaned metadata to generate spectrograms.
    
    Args:
        metadata_path: Path to tracks_cleaned.csv
        output_dir: Directory to save spectrograms
        max_tracks: Optional limit for testing (None = process all)
        
    Returns:
        Dictionary with processing statistics
    """
    print("\n" + "="*70)
    print("BATCH SPECTROGRAM GENERATION - SIGNAL PROCESSING PIPELINE")
    print("="*70)
    
    # Load cleaned metadata
    print(f"\nLoading metadata from: {metadata_path}")
    try:
        df = pd.read_csv(metadata_path)
        print(f"✓ Loaded {len(df)} tracks from cleaned dataset")
    except Exception as e:
        print(f"✗ Error loading metadata: {e}")
        return None
    
    # Limit for testing if specified
    if max_tracks is not None and max_tracks < len(df):
        df = df.head(max_tracks)
        print(f"  (Processing first {max_tracks} tracks for testing)")
    
    # Create output directory
    create_output_directory()
    
    # Processing statistics
    stats = {
        'total': len(df),
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'by_genre': {}
    }
    
    # Initialize genre counters
    for genre in df['primary_genre'].unique():
        stats['by_genre'][genre] = {'success': 0, 'failed': 0}
    
    print(f"\nProcessing {len(df)} tracks...")
    print(f"Signal processing parameters:")
    print(f"  - Sample rate: {SAMPLE_RATE} Hz")
    print(f"  - FFT window: {N_FFT}")
    print(f"  - Hop length: {HOP_LENGTH}")
    print(f"  - Mel bands: {N_MELS}")
    print(f"  - Duration: {DURATION}s")
    print(f"  - Output size: {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels")
    print("\nStarting batch processing...\n")
    
    # Process each track
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating spectrograms"):
        track_id = row['track_id']
        genre = row['primary_genre']
        
        # Process the track
        success, output_path = process_single_track(track_id, genre, output_dir)
        
        if success:
            stats['successful'] += 1
            stats['by_genre'][genre]['success'] += 1
        else:
            stats['failed'] += 1
            stats['by_genre'][genre]['failed'] += 1
    
    # Print summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\nTotal tracks processed: {stats['total']}")
    
    if stats['total'] > 0:
        print(f"✓ Successful: {stats['successful']} ({stats['successful']/stats['total']*100:.1f}%)")
        print(f"✗ Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
        
        print(f"\nBreakdown by genre:")
        for genre, counts in sorted(stats['by_genre'].items()):
            total = counts['success'] + counts['failed']
            if total > 0:
                print(f"  {genre:20} {counts['success']:5}/{total:5} ({counts['success']/total*100:.1f}%)")
    else:
        print("✗ No tracks found to process!")
    
    print(f"\nSpectrograms saved to: {output_dir}")
    print("="*70 + "\n")
    
    return stats


def verify_output(output_dir, sample_size=5):
    """
    Verify the generated spectrograms by checking a sample.
    
    Args:
        output_dir: Directory containing spectrograms
        sample_size: Number of files to verify
        
    Returns:
        Boolean indicating if verification passed
    """
    print("\nVerifying output files...")
    
    # Get all PNG files
    png_files = list(output_dir.rglob("*.png"))
    
    if len(png_files) == 0:
        print("✗ No spectrogram files found!")
        return False
    
    print(f"✓ Found {len(png_files)} spectrogram files")
    
    # Check sample files
    import random
    samples = random.sample(png_files, min(sample_size, len(png_files)))
    
    print(f"\nChecking {len(samples)} sample files:")
    for file_path in samples:
        file_size = file_path.stat().st_size
        print(f"  {file_path.name}: {file_size:,} bytes")
    
    return True


def main():
    """Main execution function."""
    # Check if metadata exists
    if not METADATA_FILE.exists():
        print(f"✗ Error: Metadata file not found at {METADATA_FILE}")
        print("  Please run checkpoint1_metadata_cleaner.py first.")
        return
    
    # Run batch processing
    # Set max_tracks to a number for testing, or None to process all tracks
    # Example: max_tracks=100 processes first 100 tracks
    #          max_tracks=None processes all 3,869 tracks
    stats = batch_process_spectrograms(
        metadata_path=METADATA_FILE,
        output_dir=OUTPUT_DIR,
        max_tracks=None # Change to None to process all tracks
    )
    
    if stats:
        # Verify output
        verify_output(OUTPUT_DIR)
        
        print("\n✓ Signal processing pipeline complete!")
        print(f"  Processed {stats['successful']} spectrograms successfully")
        print(f"  Ready for CNN training in Checkpoint 3")


if __name__ == "__main__":
    main()
