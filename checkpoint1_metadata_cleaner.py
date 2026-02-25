"""
Checkpoint 1 - Deliverable 1: FMA Metadata Cleaning Script
This script cleans and filters the FMA metadata for the 8 target genres.
Target genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock
"""

import pandas as pd
import os
import json
import ast
from pathlib import Path

# Define the 8 target genres
TARGET_GENRES = [
    'Electronic',
    'Experimental',
    'Folk',
    'Hip-Hop',
    'Instrumental',
    'International',
    'Pop',
    'Rock'
]

def load_metadata(metadata_path):
    """
    Load the FMA metadata CSV file.
    
    Args:
        metadata_path: Path to the raw_tracks.csv file
        
    Returns:
        DataFrame containing the metadata
    """
    print(f"Loading metadata from {metadata_path}...")
    try:
        # Use raw_tracks.csv which has simpler structure
        if 'raw_tracks' in metadata_path or 'tracks.csv' in metadata_path:
            if 'raw_tracks' not in metadata_path:
                metadata_path = metadata_path.replace('tracks.csv', 'raw_tracks.csv')
        
        df = pd.read_csv(metadata_path)
        print(f"✓ Loaded {len(df)} total tracks")
        return df
    except Exception as e:
        print(f"✗ Error loading metadata: {e}")
        return None


def parse_genres(genre_string):
    """
    Parse the genre string and extract genre titles.
    
    Args:
        genre_string: String representation of genre list
        
    Returns:
        List of genre titles or empty list if parsing fails
    """
    if pd.isna(genre_string) or genre_string == '[]':
        return []
    
    try:
        # Try to parse as Python literal first
        genres_list = ast.literal_eval(genre_string)
        if isinstance(genres_list, list):
            return [g.get('genre_title', '') for g in genres_list if isinstance(g, dict)]
    except:
        pass
    
    return []


def filter_by_genres(df):
    """
    Filter metadata to only include target genres.
    Extracts the primary (first) genre from each track.
    
    Args:
        df: DataFrame containing the metadata
        
    Returns:
        Filtered DataFrame with only target genres
    """
    print("\nFiltering by target genres...")
    
    if 'track_genres' not in df.columns:
        print("✗ Error: 'track_genres' column not found in metadata")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Extract primary genre from track_genres
    df['primary_genre'] = df['track_genres'].apply(lambda x: parse_genres(x)[0] if parse_genres(x) else None)
    
    # Filter for target genres
    filtered_df = df[df['primary_genre'].isin(TARGET_GENRES)].copy()
    print(f"✓ Found {len(filtered_df)} tracks in target genres")
    
    return filtered_df


def remove_missing_values(df):
    """
    Remove tracks with missing or null genre labels.
    
    Args:
        df: DataFrame containing the metadata
        
    Returns:
        DataFrame with rows containing missing values removed
    """
    print("\nRemoving tracks with missing values...")
    
    initial_count = len(df)
    df_clean = df.dropna(subset=['primary_genre'])
    removed = initial_count - len(df_clean)
    
    if removed > 0:
        print(f"✓ Removed {removed} tracks with missing genre labels")
    else:
        print(f"✓ No missing values found")
    
    return df_clean


def generate_summary(df):
    """
    Generate a summary of the cleaned dataset.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        Dictionary containing summary statistics
    """
    print("\n" + "="*60)
    print("CLEANED DATASET SUMMARY")
    print("="*60)
    
    print(f"\nTotal tracks: {len(df)}")
    print(f"\nGenre distribution:")
    
    genre_counts = df['primary_genre'].value_counts()
    for genre in TARGET_GENRES:
        count = genre_counts.get(genre, 0)
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  {genre:20} {count:6} tracks ({percentage:5.1f}%)")
    
    print("\n" + "="*60)
    
    return {
        'total_tracks': len(df),
        'genre_distribution': genre_counts.to_dict(),
        'target_genres': TARGET_GENRES
    }


def save_cleaned_data(df, output_path):
    """
    Save the cleaned dataset to a CSV file.
    
    Args:
        df: Cleaned DataFrame
        output_path: Path where to save the cleaned CSV
        
    Returns:
        Boolean indicating success
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"\n✓ Cleaned metadata saved to {output_path}")
        return True
    except Exception as e:
        print(f"\n✗ Error saving cleaned metadata: {e}")
        return False

def get_audio_path(audio_dir, track_id):
    """
    Generate the physical path to an FMA audio file.
    Example: Track 123 -> folder 000, file 000123.mp3
    """
    tid_str = str(int(track_id)).zfill(6)
    folder_name = tid_str[:3]
    return os.path.join(audio_dir, folder_name, f"{tid_str}.mp3")

def verify_physical_files(df, audio_dir):
    """
    Check if the .mp3 file for each track actually exists on disk.
    Args:
        df: DataFrame containing the metadata
        audio_dir: Path to the 'fma_small' or 'fma_full' audio directory
    Returns:
        DataFrame containing only tracks with existing audio files
    """
    print(f"\nVerifying physical files in {audio_dir}...")
    
    if not os.path.exists(audio_dir):
        print(f"✗ Error: Audio directory not found at {audio_dir}")
        return df

    initial_count = len(df)
    
    # Create a boolean mask for existing files
    def check_file(tid):
        path = get_audio_path(audio_dir, tid)
        return os.path.exists(path)

    # Note: 'track_id' is the column name in raw_tracks.csv
    df['file_exists'] = df['track_id'].apply(check_file)
    
    df_verified = df[df['file_exists'] == True].copy()
    df_verified.drop(columns=['file_exists'], inplace=True)
    
    removed = initial_count - len(df_verified)
    if removed > 0:
        print(f"✓ Removed {removed} tracks with missing physical .mp3 files")
    else:
        print(f"✓ All {initial_count} tracks verified on disk")
        
    return df_verified

def clean_and_filter_fma_metadata(metadata_path, audio_dir, output_path=None):
    """
    Main function updated to include file verification.
    """
    # 1. Load metadata
    df = load_metadata(metadata_path)
    if df is None: return None

    # 2. Filter by target genres
    df_filtered = filter_by_genres(df)
    if df_filtered is None: return None

    # 3. Remove tracks with missing genre labels
    df_clean = remove_missing_values(df_filtered)

    # 4. NEW: Verify physical .mp3 existence
    df_verified = verify_physical_files(df_clean, audio_dir)

    # 5. Generate summary and save
    generate_summary(df_verified)
    
    if output_path is None:
        output_dir = os.path.dirname(metadata_path)
        output_path = os.path.join(output_dir, 'tracks_cleaned.csv')
    
    save_cleaned_data(df_verified, output_path)
    return df_verified

if __name__ == "__main__":
    # Update these paths to your local Windows environment
    metadata_base = "c:\\Users\\kidus\\Desktop\\Senior Project 2\\fma_metadata\\fma_metadata"
    metadata_path = os.path.join(metadata_base, "raw_tracks.csv")
    
    # Path to where your 000, 001... folders are located
    audio_dir = "c:\\Users\\kidus\\Desktop\\Senior Project 2\\fma_small" 
    
    cleaned_df = clean_and_filter_fma_metadata(metadata_path, audio_dir)