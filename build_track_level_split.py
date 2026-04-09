"""
Create leakage-safe train/validation splits at the track level for chunked spectrograms.

Input: processed_data/spectrograms_chunked/chunk_manifest.csv
Output: train_manifest.csv, val_manifest.csv in same directory (or custom output).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def build_track_level_split(manifest_path: Path, val_split: float, seed: int, out_dir: Path) -> None:
    df = pd.read_csv(manifest_path)

    required = {"track_id", "genre", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    track_df = df[["track_id", "genre"]].drop_duplicates().reset_index(drop=True)

    train_tracks, val_tracks = train_test_split(
        track_df,
        test_size=val_split,
        random_state=seed,
        shuffle=True,
        stratify=track_df["genre"],
    )

    train_ids = set(train_tracks["track_id"].tolist())
    val_ids = set(val_tracks["track_id"].tolist())

    train_manifest = df[df["track_id"].isin(train_ids)].copy()
    val_manifest = df[df["track_id"].isin(val_ids)].copy()

    overlap = set(train_manifest["track_id"]).intersection(set(val_manifest["track_id"]))
    if overlap:
        raise RuntimeError(f"Leakage detected: {len(overlap)} overlapping track IDs")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_manifest.csv"
    val_path = out_dir / "val_manifest.csv"

    train_manifest.to_csv(train_path, index=False)
    val_manifest.to_csv(val_path, index=False)

    print(f"Saved train manifest: {train_path} ({len(train_manifest)} rows)")
    print(f"Saved val manifest:   {val_path} ({len(val_manifest)} rows)")
    print(f"Train tracks: {train_manifest['track_id'].nunique()} | Val tracks: {val_manifest['track_id'].nunique()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build track-level train/val split for chunked data")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("processed_data/spectrograms_chunked/chunk_manifest.csv"),
        help="Path to chunk manifest CSV",
    )
    p.add_argument("--val-split", type=float, default=0.2, help="Validation ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("processed_data/spectrograms_chunked"),
        help="Output directory for split manifests",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_track_level_split(args.manifest, args.val_split, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
