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


def clean_and_filter_fma_metadata(metadata_path, output_path=None):
    """
    Main function to clean and filter FMA metadata.
    
    Args:
        metadata_path: Path to the input tracks.csv file
        output_path: Optional path to save the cleaned CSV (default: cleaned_tracks.csv in same directory)
        
    Returns:
        Cleaned DataFrame or None if error
    """
    
    # Load metadata
    df = load_metadata(metadata_path)
    if df is None:
        return None
    
    # Filter by target genres
    df_filtered = filter_by_genres(df)
    if df_filtered is None:
        return None
    
    # Remove missing values
    df_clean = remove_missing_values(df_filtered)
    
    # Generate summary
    summary = generate_summary(df_clean)
    
    # Save cleaned data
    if output_path is None:
        output_dir = os.path.dirname(metadata_path)
        output_path = os.path.join(output_dir, 'tracks_cleaned.csv')
    
    save_cleaned_data(df_clean, output_path)
    
    return df_clean


# Example usage
if __name__ == "__main__":
    # Path to the FMA metadata - using raw_tracks.csv
    metadata_path = "c:\\Users\\kidus\\Desktop\\Senior Project 2\\fma_metadata\\fma_metadata\\raw_tracks.csv"
    
    # Run the cleaning process
    cleaned_df = clean_and_filter_fma_metadata(metadata_path)
    
    if cleaned_df is not None:
        print("\n✓ Metadata cleaning completed successfully!")
        print(f"Cleaned dataset has {len(cleaned_df)} tracks ready for training.")
    else:
        print("\n✗ Metadata cleaning failed.")
