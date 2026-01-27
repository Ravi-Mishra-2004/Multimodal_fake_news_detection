"""
Dataset Setup Script for FakeReddit Multimodal Fake News Detection

This script provides helper functions and classes for loading and preparing
the preprocessed FakeReddit dataset for model training (local or Colab).
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
import json


class FakedditDataset(Dataset):
    """
    PyTorch Dataset class for FakeReddit multimodal data.
    
    Returns:
        - input_ids: BERT tokenized text input IDs
        - attention_mask: BERT attention mask
        - image: Preprocessed image tensor
        - label: Binary label (0=Real, 1=Fake)
    """
    
    def __init__(self, csv_path, images_dir, text_field='clean_title', 
                 label_field='2_way_label', max_length=128):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to preprocessed CSV file
            images_dir: Directory containing preprocessed images
            text_field: Name of text column in CSV
            label_field: Name of label column in CSV
            max_length: Maximum sequence length for BERT tokenization
        """
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.text_field = text_field
        self.label_field = label_field
        self.max_length = max_length
        
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Image transforms (matching preprocessing)
        # Note: Images are already resized to 256x256 and converted to RGB
        # Image transforms (matching preprocessing)
        # Note: Images are already resized to 256x256 and converted to RGB
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get text
        text = str(self.df.iloc[idx][self.text_field])
        
        # Get label
        label = int(self.df.iloc[idx][self.label_field])
        
        # Get image path
        image_id = self.df.iloc[idx]['id']
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # If image fails to load, create a black image
            print(f"Warning: Failed to load image {image_path}: {e}")
            image = torch.zeros((3, 256, 256))
        
        # Tokenize text
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_metadata(metadata_path):
    """Load dataset metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def create_dataloaders(csv_path, images_dir, batch_size=16, val_split=0.2, 
                       test_split=0.1, random_seed=42):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        csv_path: Path to preprocessed CSV file
        images_dir: Directory containing images
        batch_size: Batch size for DataLoader
        val_split: Validation set split ratio
        test_split: Test set split ratio
        random_seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    from sklearn.model_selection import train_test_split
    
    # Load full dataset
    df = pd.read_csv(csv_path)
    
    # Split into train, val, test
    df_train, df_temp = train_test_split(
        df, 
        test_size=(val_split + test_split), 
        stratify=df['2_way_label'],
        random_state=random_seed
    )
    
    df_val, df_test = train_test_split(
        df_temp,
        test_size=test_split / (val_split + test_split),
        stratify=df_temp['2_way_label'],
        random_state=random_seed
    )
    
    # Save splits to temporary CSV files
    train_csv = csv_path.replace('.csv', '_train.csv')
    val_csv = csv_path.replace('.csv', '_val.csv')
    test_csv = csv_path.replace('.csv', '_test.csv')
    
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    
    # Create datasets
    train_dataset = FakedditDataset(train_csv, images_dir)
    val_dataset = FakedditDataset(val_csv, images_dir)
    test_dataset = FakedditDataset(test_csv, images_dir)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def get_class_weights(csv_path):
    """
    Calculate class weights for balanced loss function.
    
    Args:
        csv_path: Path to preprocessed CSV file
    
    Returns:
        torch.Tensor of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    df = pd.read_csv(csv_path)
    labels = df['2_way_label'].to_numpy()
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return torch.tensor(class_weights, dtype=torch.float32)


# Example usage and model architecture template
if __name__ == "__main__":
    print("FakeReddit Dataset Setup Helper")
    print("=" * 60)
    print("\nThis script provides helper functions for dataset loading.")
    print("\nUsage:")
    print("  from dataset_setup import FakedditDataset, create_dataloaders, get_class_weights")
    print("\nSee train_model.py for training example.")

