"""
FakeReddit Multimodal Fake News Detection - Preprocessing Pipeline

This script downloads, validates, and preprocesses the FakeReddit dataset
for binary (fake/real) classification. It handles image validation,
text preprocessing, and prepares data for local training.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import requests
from tqdm import tqdm
from urllib.parse import urlparse
import kagglehub
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')


# Configuration
SAMPLE_SIZE = 10000
IMAGE_SIZE = (256, 256)
IMAGE_DIR = "images"
PROCESSED_DATA_DIR = "processed_data"
DATASET_NAME = "vanshikavmittal/fakeddit-dataset"
MAX_TEXT_LENGTH = 128  # For BERT tokenization


def setup_directories():
    """Create necessary directories for output."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print(f"Created directories: {IMAGE_DIR}, {PROCESSED_DATA_DIR}")


def download_dataset():
    """Download the FakeReddit dataset using kagglehub."""
    print(f"Downloading dataset: {DATASET_NAME}...")
    try:
        path = kagglehub.dataset_download(DATASET_NAME)
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise


def find_dataset_file(dataset_path):
    """Find the multimodal TSV file in the downloaded dataset."""
    dataset_path = Path(dataset_path)
    
    # Common file patterns in FakeReddit dataset
    possible_files = [
        "multimodal_only_samples/multimodal_train.tsv",
        "multimodal_train.tsv",
        "train.tsv",
        "multimodal_only_samples/train.tsv"
    ]
    
    for file_pattern in possible_files:
        full_path = dataset_path / file_pattern
        if full_path.exists():
            print(f"Found dataset file: {full_path}")
            return str(full_path)
    
    # If not found, search for TSV files
    tsv_files = list(dataset_path.rglob("*.tsv"))
    if tsv_files:
        print(f"Found TSV files: {[str(f) for f in tsv_files]}")
        # Prefer files with 'multimodal' in name
        multimodal_files = [f for f in tsv_files if 'multimodal' in str(f).lower()]
        if multimodal_files:
            return str(multimodal_files[0])
        return str(tsv_files[0])
    
    raise FileNotFoundError(f"Could not find dataset TSV file in {dataset_path}")


def load_dataset(dataset_path):
    """Load the dataset TSV file."""
    tsv_file = find_dataset_file(dataset_path)
    print(f"Loading dataset from: {tsv_file}")
    
    df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    return df


def inspect_dataset(df):
    """Inspect dataset structure and columns."""
    print("\n=== Dataset Inspection ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check for required columns
    required_cols = ['id', 'clean_title', 'image_url', 'hasImage']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\nWarning: Missing columns: {missing_cols}")
    
    # Check for label columns
    label_cols = [col for col in df.columns if 'label' in col.lower() or 'way' in col.lower()]
    print(f"\nLabel columns found: {label_cols}")
    
    if '2_way_label' in df.columns:
        print(f"\n2_way_label distribution:")
        print(df['2_way_label'].value_counts())
    elif '2way_label' in df.columns:
        print(f"\n2way_label distribution:")
        print(df['2way_label'].value_counts())
        # Rename for consistency
        df = df.rename(columns={'2way_label': '2_way_label'})
    
    return df


def filter_and_sample_data(df):
    """Filter rows with images and sample 10K rows maintaining class distribution."""
    print("\n=== Filtering and Sampling Data ===")
    
    # Filter rows with images
    initial_count = len(df)
    df = df[df['hasImage'] == True].copy()
    print(f"Rows with hasImage=True: {len(df)}")
    
    # Filter rows with non-empty image URLs
    df = df[df['image_url'].notna()].copy()
    df = df[df['image_url'] != ''].copy()
    df = df[df['image_url'] != 'nan'].copy()
    print(f"Rows with valid image_url: {len(df)}")
    
    # Check for 2_way_label
    if '2_way_label' not in df.columns:
        raise ValueError("2_way_label column not found. Available columns: " + str(list(df.columns)))
    
    # Remove rows with missing labels
    df = df[df['2_way_label'].notna()].copy()
    print(f"Rows with valid 2_way_label: {len(df)}")
    
    # Check class distribution
    print(f"\nClass distribution before sampling:")
    print(df['2_way_label'].value_counts())
    
    # Sample 10K rows with stratified sampling
    if len(df) >= SAMPLE_SIZE:
        df_sampled, _ = train_test_split(
            df,
            test_size=len(df) - SAMPLE_SIZE,
            stratify=df['2_way_label'],
            random_state=42
        )
        df_sampled = df_sampled.reset_index(drop=True)
        print(f"\nSampled {len(df_sampled)} rows")
    else:
        df_sampled = df.reset_index(drop=True)
        print(f"\nDataset has {len(df_sampled)} rows (less than {SAMPLE_SIZE}), using all rows")
    
    print(f"\nClass distribution after sampling:")
    print(df_sampled['2_way_label'].value_counts())
    
    return df_sampled


def is_valid_image_url(url, timeout=5):
    """Check if image URL is valid and accessible."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()
            return 'image' in content_type
        return False
    except:
        return False


def download_image(url, save_path, timeout=10):
    """Download image from URL and save to path."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        return False


def download_and_validate_images(df):
    """Download images from URLs and validate them."""
    print("\n=== Downloading and Validating Images ===")
    
    valid_indices = []
    failed_downloads = 0
    invalid_urls = 0
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def process_row(row):
        image_url = row['image_url']
        image_id = row['id']
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        
        # Skip if already downloaded
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                img.verify()
                return row.name, True
            except:
                os.remove(image_path)
        
        # Check URL validity
        if not is_valid_image_url(image_url):
            return row.name, False
        
        # Download image
        if download_image(image_url, image_path):
            try:
                img = Image.open(image_path)
                img.verify()
                return row.name, True
            except:
                if os.path.exists(image_path):
                    os.remove(image_path)
                return row.name, False
        return row.name, False

    print(f"Downloading images with parallel processing (max_workers=10)...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            idx, success = future.result()
            if success:
                valid_indices.append(idx)
            else:
                failed_downloads += 1
    
    print(f"\nDownload Statistics:")
    print(f"  Total rows: {len(df)}")
    print(f"  Successful downloads: {len(valid_indices)}")
    print(f"  Failed downloads: {failed_downloads}")
    # print(f"  Invalid URLs: {invalid_urls}") # invalid_urls logic moved inside function, simplified reporting
    
    # Filter dataframe to only valid images
    df_valid = df.loc[valid_indices].copy().reset_index(drop=True)
    print(f"\nValid rows after image download: {len(df_valid)}")
    
    return df_valid


def validate_image_file(image_path):
    """Validate that an image file exists and is not corrupted."""
    try:
        if not os.path.exists(image_path):
            return False
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False


def preprocess_images(df):
    """Validate, resize, and convert images to standard format."""
    print("\n=== Preprocessing Images ===")
    
    valid_indices = []
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing images"):
        image_id = row['id']
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        
        try:
            # Open and validate image
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # Save processed image
            img_resized.save(image_path, 'JPEG', quality=95)
            
            # Verify saved image
            img_verify = Image.open(image_path)
            img_verify.verify()
            valid_indices.append(idx)
            
        except Exception as e:
            # Remove corrupted image
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
            continue
    
    print(f"\nImage Preprocessing Statistics:")
    print(f"  Total images: {len(df)}")
    print(f"  Successfully processed: {len(valid_indices)}")
    print(f"  Failed: {len(df) - len(valid_indices)}")
    
    # Filter dataframe to only valid images
    df_valid = df.loc[valid_indices].copy().reset_index(drop=True)
    print(f"\nValid rows after image preprocessing: {len(df_valid)}")
    
    return df_valid


def preprocess_text(df):
    """Validate and prepare text data for BERT tokenization."""
    print("\n=== Preprocessing Text ===")
    
    # Check for clean_title column
    if 'clean_title' not in df.columns:
        raise ValueError("clean_title column not found")
    
    # Remove rows with null or empty text
    initial_count = len(df)
    df = df[df['clean_title'].notna()].copy()
    df = df[df['clean_title'] != ''].copy()
    
    # Convert to string and strip whitespace
    df['clean_title'] = df['clean_title'].astype(str).str.strip()
    
    # Remove rows that became empty after stripping
    df = df[df['clean_title'] != ''].copy()
    
    print(f"Text Preprocessing Statistics:")
    print(f"  Initial rows: {initial_count}")
    print(f"  Valid text rows: {len(df)}")
    print(f"  Removed: {initial_count - len(df)}")
    
    # Text length statistics
    text_lengths = df['clean_title'].str.len()
    print(f"\nText Length Statistics:")
    print(f"  Mean: {text_lengths.mean():.2f}")
    print(f"  Median: {text_lengths.median():.2f}")
    print(f"  Min: {text_lengths.min()}")
    print(f"  Max: {text_lengths.max()}")
    print(f"  Rows > {MAX_TEXT_LENGTH} chars: {(text_lengths > MAX_TEXT_LENGTH).sum()}")
    
    return df.reset_index(drop=True)


def generate_statistics(df):
    """Generate comprehensive dataset statistics."""
    stats = {
        'total_samples': len(df),
        'class_distribution': df['2_way_label'].value_counts().to_dict(),
        'class_balance': {
            'real': int((df['2_way_label'] == 0).sum()),
            'fake': int((df['2_way_label'] == 1).sum())
        },
        'text_statistics': {
            'mean_length': float(df['clean_title'].str.len().mean()),
            'median_length': float(df['clean_title'].str.len().median()),
            'min_length': int(df['clean_title'].str.len().min()),
            'max_length': int(df['clean_title'].str.len().max())
        },
        'image_statistics': {
            'target_size': IMAGE_SIZE,
            'format': 'JPEG',
            'color_mode': 'RGB'
        },
        'preprocessing_parameters': {
            'sample_size': SAMPLE_SIZE,
            'max_text_length': MAX_TEXT_LENGTH,
            'image_size': IMAGE_SIZE
        }
    }
    
    return stats


def save_preprocessed_data(df, stats):
    """Save preprocessed data and metadata."""
    print("\n=== Saving Preprocessed Data ===")
    
    # Add image_path column
    df['image_path'] = df['id'].apply(lambda x: f"{IMAGE_DIR}/{x}.jpg")
    
    # Select columns to save
    columns_to_save = ['id', 'clean_title', '2_way_label', 'image_path']
    df_save = df[columns_to_save].copy()
    
    # Save CSV
    csv_path = os.path.join(PROCESSED_DATA_DIR, "preprocessed_fakeddit.csv")
    df_save.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")
    
    # Save metadata
    metadata_path = os.path.join(PROCESSED_DATA_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    # Save requirements for Colab
    requirements_path = os.path.join(PROCESSED_DATA_DIR, "requirements.txt")
    with open('requirements.txt', 'r') as f:
        requirements_content = f.read()
    with open(requirements_path, 'w') as f:
        f.write(requirements_content)
    print(f"Saved requirements to: {requirements_path}")
    
    return csv_path, metadata_path


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("FakeReddit Multimodal Preprocessing Pipeline")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    # Download dataset
    dataset_path = download_dataset()
    
    # Load dataset
    df = load_dataset(dataset_path)
    
    # Inspect dataset
    df = inspect_dataset(df)
    
    # Filter and sample
    df = filter_and_sample_data(df)
    
    # Download and validate images
    df = download_and_validate_images(df)
    
    # Preprocess images
    df = preprocess_images(df)
    
    # Preprocess text
    df = preprocess_text(df)
    
    # Generate statistics
    stats = generate_statistics(df)
    
    # Save preprocessed data
    csv_path, metadata_path = save_preprocessed_data(df, stats)
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"\nFinal dataset size: {len(df)}")
    print(f"Class distribution:")
    print(df['2_way_label'].value_counts())
    print(f"\nOutput files:")
    print(f"  CSV: {csv_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Images: {IMAGE_DIR}/ ({(len(df))} images)")
    print("\nReady for training!")


if __name__ == "__main__":
    main()

