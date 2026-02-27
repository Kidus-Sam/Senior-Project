# Signal Processing Pipeline - Documentation

## Overview

This batch processing pipeline converts filtered FMA audio tracks into Mel-spectrogram images for CNN training. It implements the audio processing specifications from the AudioTexture workplan.

## Files Created

### 1. `batch_spectrogram_generator.py`

Main signal processing script that generates Mel-spectrograms from audio files.

**Key Features:**

- Batch processes audio tracks from cleaned metadata
- Applies FFT-based Mel-spectrogram generation
- Organizes output by genre in separate subdirectories
- Progress tracking with tqdm
- Error handling for corrupted or missing files
- Automatic file skipping (doesn't reprocess existing spectrograms)

**Technical Parameters:**

- Sample Rate: 22,050 Hz
- FFT Window Size: 2,048 samples
- Hop Length: 512 samples
- Mel Bands: 128
- Audio Duration: 30 seconds (padded/clipped)
- Output Image Size: 224×224 pixels
- Output Format: PNG

## Usage

### Basic Usage (Test Mode)

Process first 100 tracks:

```bash
conda run -p ".\.conda" python batch_spectrogram_generator.py
```

### Process All Tracks

Edit line 348 in `batch_spectrogram_generator.py`:

```python
max_tracks=None  # Change from 100 to None
```

Then run:

```bash
conda run -p ".\.conda" python batch_spectrogram_generator.py
```

## Output Structure

```
processed_data/
└── spectrograms/
    ├── Electronic/
    │   ├── 000123.png
    │   └── ...
    ├── Experimental/
    ├── Folk/
    ├── Hip-Hop/
    ├── Instrumental/
    ├── International/
    ├── Pop/
    └── Rock/
```

Each spectrogram file is named with its 6-digit track ID (e.g., `000123.png`).

## Processing Results

### Test Run (100 tracks)

- **Total tracks:** 100
- **Success rate:** 100%
- **Processing speed:** ~3-4 tracks/second
- **File size:** 60-67 KB per spectrogram

### Genre Distribution (from test)

- Folk: 49 tracks (49%)
- Rock: 17 tracks (17%)
- International: 12 tracks (12%)
- Pop: 10 tracks (10%)
- Experimental: 5 tracks (5%)
- Hip-Hop: 5 tracks (5%)
- Electronic: 2 tracks (2%)

### Full Dataset Available

- **Total verified tracks:** 3,869
- **Expected processing time:** ~15-20 minutes for all tracks

## Signal Processing Pipeline

### Step 1: Audio Loading

- Load MP3 file using librosa
- Convert to mono
- Resample to 22,050 Hz
- Clip or pad to exactly 30 seconds

### Step 2: Mel-Spectrogram Generation

- Apply Short-Time Fourier Transform (STFT)
- Convert to Mel scale (128 bands)
- Convert amplitude to decibels (log scale)
- Result: 2D array representing frequency content over time

### Step 3: Image Saving

- Render spectrogram as 224×224 pixel image
- Use 'viridis' colormap for visual clarity
- Remove axes and whitespace
- Save as PNG with 100 DPI

## Dependencies

Required packages (all installed):

- pandas - Metadata handling
- numpy - Numerical operations
- librosa - Audio processing
- soundfile - Audio I/O backend
- matplotlib - Image generation
- tqdm - Progress bars

## Integration with Project

This pipeline fulfills the Checkpoint 1 requirement:

> **Signal Processing Pipeline:** Implement the batch processing loop to convert
> the filtered audio tracks into Mel-spectrogram images. Developing the loop that
> iterates through the cleaned dataset and utilizes librosa to save spectral images
> to a dedicated /processed_data/ directory.

### Next Steps (Checkpoint 2+)

1. Integrate spectrogram display in GUI (`checkpoint1_gui_shell.py`)
2. Load spectrograms when user drops audio file
3. Train CNN model on generated spectrograms (Checkpoint 3)
4. Implement real-time inference using trained model

## Troubleshooting

### "No tracks found to process"

- Ensure `tracks_cleaned.csv` exists in `fma_metadata/fma_metadata/`
- Run `checkpoint1_metadata_cleaner.py` first

### "Module not found" errors

- Install missing dependencies:
  ```bash
  conda run -p ".\.conda" pip install librosa soundfile matplotlib tqdm
  ```

### Spectrograms look incorrect

- Check that audio files are valid MP3s
- Verify FFT parameters (N_FFT=2048, HOP_LENGTH=512)
- Ensure matplotlib backend is set to 'Agg'

### Slow processing

- Processing speed: ~3-4 tracks/second is normal
- First track takes longer (library initialization)
- Use SSD storage for faster I/O

## Performance Notes

- **Memory usage:** ~200-300 MB during processing
- **Disk space:** ~65 KB per spectrogram
  - 100 tracks = ~6.5 MB
  - 3,869 tracks = ~250 MB total
- **CPU usage:** Moderate (single-threaded)
- **GPU usage:** Not required for preprocessing

## File Organization

All spectrograms are organized by genre to facilitate:

1. Easy data loading during CNN training
2. Visual inspection by genre
3. Genre-specific analysis
4. Balanced sampling for training/validation splits

## Validation

The script includes automatic validation:

- Verifies audio file existence before processing
- Checks output file creation
- Reports success/failure statistics
- Sample file verification (5 random files)

## Credits

Based on the AudioTexture project workplan:

- Technical parameters from Section 7 (Technical Details)
- FFT specifications: 2048 window, 512 hop length
- Mel-scale conversion for human perceptual alignment
- 224×224 output for CNN compatibility
