# AudioTexture

AudioTexture is a desktop music genre classification project that turns raw audio into Mel-spectrograms and predicts one of eight genres with a trained deep learning model.

## Why This Project Was Made

Music discovery and royalty pipelines rely heavily on metadata, but manual genre tagging is often inconsistent, subjective, or missing. AudioTexture was built to provide a more objective, signal-driven approach to genre labeling by:

- transforming audio into visual spectral representations (Mel-spectrograms),
- learning genre patterns from labeled examples (FMA dataset), and
- returning interpretable predictions with confidence and top-3 alternatives.

In short: this project exists to make genre tagging more reliable for independent creators and music workflows.

## What The Project Does

- Accepts audio input through a Tkinter desktop GUI (drag-and-drop + file picker fallback).
- Generates Mel-spectrograms from audio in real time.
- Runs genre inference using a trained Keras checkpoint when available.
- Uses chunk-level voting (5-second chunks over a 30-second window) for more stable predictions.
- Displays:
  - predicted genre,
  - confidence,
  - top-3 predictions,
  - processing status and timing,
  - spectrogram visualization.
- Supports:
  - Classic and Modern UI modes,
  - batch folder inference to CSV,
  - performance snapshot view,
  - demo self-check report generation.

## Target Genres

The current model and metadata pipeline are built around these 8 classes:

- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Pop
- Rock

## Repository Structure (Key Files)

- `gui_shell.py`: Main desktop app and inference integration.
- `metadata_cleaner.py`: Filters raw metadata to target genres and verifies physical audio files.
- `batch_spectrogram_generator.py`: Generates full-track Mel-spectrogram images.
- `chunked_spectrogram_generator.py`: Generates chunk-level spectrogram dataset (+ optional augmentation).
- `build_track_level_split.py`: Creates leakage-safe train/validation manifests at track level.
- `test_colab.ipynb`: Notebook workflow for model training/evaluation experiments.
- `model_performance_snapshot.csv`: Per-genre metrics used by the GUI snapshot feature.

## Data Setup
This repository does not host the raw audio or metadata due to file size. You must download the Free Music Archive (FMA) dataset components manually:

1) Download Metadata: Download [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) and extract it.
2) Place the contents in a folder named `fma_metadata` in the project root.
3) Download Audio: * Download [fma_small.zip](https://os.unil.cloud.switch.ch/fma/fma_small.zip) (8GB, 8,000 tracks).
4) Extract it into a folder named fma_small/ in the project root.
###Verify Structure:
Your local directory should look like this for the scripts to run correctly:
```
Senior-Project/
├── fma_metadata/             # Extracted CSV files
├── fma_small/                # Folders 000 through 155 containing .mp3s
├── gui_shell.py              # Main application
└── ...
```
###Generate Processed Data:
After the files are in place, run the following to generate the missing processed_data/ folder:
1) Run `python metadata_cleaner.py` to filter the dataset.
2) Run `python chunked_spectrogram_generator.py` to create the `processed_data/` folder needed for training and inference.

###Data folders:

- `fma_metadata/`: metadata CSV files.
- `fma_small/`: audio files in FMA folder layout.
- `processed_data/`: generated spectrogram datasets and manifests.

## System Requirements

- Windows (project developed and tested on Windows)
- Python 3.10+
- Optional but recommended: conda environment 

Core Python packages used in the project:

- tensorflow
- librosa
- numpy
- pandas
- matplotlib
- pillow
- scikit-learn
- tqdm
- tkinterdnd2 (optional; enables better drag-and-drop behavior)

## Setup

If you are using the existing local environment in this repo, you can run scripts with:

```powershell
& python ".\gui_shell.py"
```

If you are creating a fresh environment, install dependencies first:

```powershell
pip install tensorflow librosa numpy pandas matplotlib pillow scikit-learn tqdm tkinterdnd2
```

## How To Use The Project

### 1. Prepare Clean Metadata

```powershell
python ".\metadata_cleaner.py"
```

This creates/updates cleaned metadata (including file existence verification) for the 8 target genres.

### 2. Generate Spectrogram Data (Optional for Training)

Full-track spectrogram generation:

```powershell
python ".\batch_spectrogram_generator.py"
```

Chunk-level dataset generation (recommended for chunk voting workflows):

```powershell
python ".\chunked_spectrogram_generator.py"
```

With augmentation enabled:

```powershell
python ".\chunked_spectrogram_generator.py" --augment
```

### 3. Build Train/Validation Split Manifest (Chunk Dataset)

```powershell
pyhton ".\build_track_level_split.py"
```

This prevents leakage by splitting at the track level before training.

### 4. Launch The Desktop App

```powershell
python ".\gui_shell.py"
```

Inside the app:

1. Drop an audio file (or click to browse).
2. Wait for processing and inference.
3. Review genre, confidence, top-3, and rendered spectrogram.
4. Optionally switch UI mode, export batch predictions, or run demo checks.

## Model + Inference Notes

- The GUI attempts to load the best available checkpoint from model checkpoint directories.
- Inference profile is selected by checkpoint family (for example engineered 3-channel preprocessing with chunk voting).
- If no suitable model is available, deterministic fallback logic is used so the interface remains functional.

## Typical Workflow For Class Demo

1. Launch GUI.
2. Run one single-file prediction.
3. Show top-3 and low-confidence behavior.
4. Run batch folder export to CSV.
5. Open Performance Snapshot.
6. Run Demo Self-Check and show generated report file.

## Troubleshooting

- If drag-and-drop is limited, install `tkinterdnd2` and use click-to-browse as fallback.
- If audio decode fails on some formats, convert to WAV/MP3 and retry.
- If no model loads, verify checkpoint files exist and are compatible with current TensorFlow/Keras.
- If spectrogram generation is slow, test with smaller subsets first before full dataset processing.

## Ethical and Practical Considerations

- Predictions are advisory, not absolute truth.
- Genre boundaries are culturally fluid; model outputs may not reflect artist intent.
- Use this system to support metadata decisions, not replace human review entirely.

## Future Improvements

- Multi-label predictions for hybrid genres.
- Mood/energy/BPM tagging.
- API/web deployment for mobile-friendly access.
- Persisting user UI preferences and extended analytics dashboards.

## Acknowledgments

- FMA Dataset: https://github.com/mdeff/fma
- Librosa: https://librosa.org/
- TensorFlow/Keras and the Python scientific stack used throughout the pipeline.
