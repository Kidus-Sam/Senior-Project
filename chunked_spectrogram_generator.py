"""
Chunked audio-to-spectrogram generator with optional audio-domain augmentation.

Outputs 5-second chunk images and a manifest CSV for downstream training.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
TRACK_DURATION = 30.0
CHUNK_DURATION = 5.0
FMAX = 8000
IMG_SIZE = (224, 224)


def get_audio_path(audio_root: Path, track_id: int) -> Path:
    tid = str(int(track_id)).zfill(6)
    return audio_root / tid[:3] / f"{tid}.mp3"


def load_audio_fixed(file_path: Path, sr: int, duration: float) -> np.ndarray:
    audio, _ = librosa.load(str(file_path), sr=sr, mono=True, duration=duration)
    target_len = int(sr * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        audio = audio[:target_len]
    return audio.astype(np.float32)


def add_noise(x: np.ndarray, noise_level: float = 0.003) -> np.ndarray:
    noise = np.random.normal(0.0, noise_level, size=x.shape).astype(np.float32)
    return (x + noise).astype(np.float32)


def augment_audio(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    out = {"orig": audio}
    out["stretch_0p95"] = librosa.effects.time_stretch(audio, rate=0.95).astype(np.float32)
    out["stretch_1p05"] = librosa.effects.time_stretch(audio, rate=1.05).astype(np.float32)
    out["pitch_m2"] = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2).astype(np.float32)
    out["pitch_p2"] = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2).astype(np.float32)
    out["noise"] = add_noise(audio)
    return out


def to_fixed_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)), mode="constant").astype(np.float32)
    return audio[:target_len].astype(np.float32)


def mel_db_from_audio(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=FMAX,
    )
    return librosa.power_to_db(mel, ref=np.max)


def save_mel_image(mel_db: np.ndarray, out_path: Path) -> None:
    dpi = 100
    fig_w = IMG_SIZE[0] / dpi
    fig_h = IMG_SIZE[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis=None,
        y_axis=None,
        cmap="viridis",
        ax=ax,
    )
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)


def generate_chunks_for_track(audio: np.ndarray) -> list[np.ndarray]:
    chunk_len = int(SAMPLE_RATE * CHUNK_DURATION)
    n_chunks = int(TRACK_DURATION / CHUNK_DURATION)
    chunks = []
    for i in range(n_chunks):
        s = i * chunk_len
        e = s + chunk_len
        chunk = audio[s:e]
        chunks.append(to_fixed_length(chunk, chunk_len))
    return chunks


def run(
    metadata_csv: Path,
    audio_root: Path,
    output_root: Path,
    include_augmentation: bool,
    max_tracks: int | None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    df = pd.read_csv(metadata_csv)
    if max_tracks is not None:
        df = df.head(max_tracks)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking tracks"):
        track_id = int(row["track_id"])
        genre = str(row["primary_genre"])
        audio_path = get_audio_path(audio_root, track_id)
        if not audio_path.exists():
            continue

        try:
            audio = load_audio_fixed(audio_path, SAMPLE_RATE, TRACK_DURATION)
        except Exception:
            continue

        versions = {"orig": audio}
        if include_augmentation:
            versions = augment_audio(audio, SAMPLE_RATE)

        for aug_name, audio_variant in versions.items():
            variant_fixed = to_fixed_length(audio_variant, int(SAMPLE_RATE * TRACK_DURATION))
            chunks = generate_chunks_for_track(variant_fixed)
            for chunk_idx, chunk in enumerate(chunks):
                mel_db = mel_db_from_audio(chunk)
                genre_dir = output_root / genre
                genre_dir.mkdir(parents=True, exist_ok=True)

                tid = str(track_id).zfill(6)
                out_name = f"{tid}_c{chunk_idx:02d}_{aug_name}.png"
                out_path = genre_dir / out_name
                save_mel_image(mel_db, out_path)

                manifest_rows.append(
                    {
                        "track_id": track_id,
                        "chunk_id": chunk_idx,
                        "augment": aug_name,
                        "genre": genre,
                        "image_path": str(out_path),
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_root / "chunk_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved {len(manifest)} chunk records -> {manifest_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate chunk-level spectrogram dataset")
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("fma_metadata/fma_metadata/tracks_cleaned.csv"),
        help="Path to cleaned tracks metadata CSV",
    )
    p.add_argument(
        "--audio-root",
        type=Path,
        default=Path("fma_small/fma_small"),
        help="Root folder containing FMA audio folders",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed_data/spectrograms_chunked"),
        help="Output folder for chunk images",
    )
    p.add_argument(
        "--augment",
        action="store_true",
        help="Enable audio-domain augmentation before chunking",
    )
    p.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Optional cap for quick tests",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        metadata_csv=args.metadata,
        audio_root=args.audio_root,
        output_root=args.output_root,
        include_augmentation=args.augment,
        max_tracks=args.max_tracks,
    )


if __name__ == "__main__":
    main()
