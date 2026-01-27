"""
Training Script for FakeReddit Multimodal Fake News Detection

Train a multimodal model (BERT + ResNet) for binary fake news classification.
This script is designed to run on local Mac (CPU or MPS if available).
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from transformers import BertModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import json
from dataset_setup import FakedditDataset, create_dataloaders, get_class_weights


# Check for available device
def get_device():
    """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class MultimodalFakeNewsClassifier(nn.Module):
    """
    Multimodal classifier combining BERT for text and ResNet50 for images.
    """
    
    def __init__(self, num_classes=2, dropout=0.3, fine_tune=False):
        super().__init__()
        
        # Text branch (BERT)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        if fine_tune:
            # Unfreeze last layer of BERT
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.bert.encoder.layer[-1].parameters():
                param.requires_grad = True
        else:
            # Freeze BERT layers
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.text_fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        
        # Image branch (ResNet50)
        resnet = models.resnet50(weights='IMAGENET1K_V2')
        
        if fine_tune:
            # Unfreeze last block of ResNet
            for param in resnet.parameters():
                param.requires_grad = False
            for param in resnet.layer4.parameters():
                param.requires_grad = True
        else:
            # Freeze ResNet layers
            for param in resnet.parameters():
                param.requires_grad = False
        
        self.image_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.image_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128)
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64)
        )
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, input_ids, attention_mask, image):
        # Text features
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_pooled = text_outputs.pooler_output
        text_features = self.text_fc(text_pooled)
        
        # Image features
        image_features = self.image_backbone(image)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_fc(image_features)
        
        # Fusion (concatenate)
        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        
        return output


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return total_loss / len(val_loader), accuracy, precision, recall, f1, all_preds, all_labels


def train_model(csv_path, images_dir, num_epochs=10, batch_size=16, 
                learning_rate=1e-4, val_split=0.2, test_split=0.1, fine_tune=False):
    """Main training function."""
    import os
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Setup model save path
    model_save_path = os.path.join(os.getcwd(), 'best_model.pth')
    results_save_path = os.path.join(os.getcwd(), 'training_results.json')
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=csv_path,
        images_dir=images_dir,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
        random_seed=42
    )
    
    # Get class weights
    class_weights = get_class_weights(csv_path)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")
    
    # Initialize model
    print("\nInitializing model...")
    model = MultimodalFakeNewsClassifier(num_classes=2, fine_tune=fine_tune)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        print(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'history': history
                }, model_save_path)
                print(f"✓ Saved best model (Val F1: {val_f1:.4f}) to {model_save_path}")
            except Exception as e:
                print(f"Warning: Failed to save model: {e}")
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    best_epoch = num_epochs - 1  # Default to last epoch
    if os.path.exists(model_save_path):
        try:
            checkpoint = torch.load(model_save_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_epoch = checkpoint['epoch']
            print(f"Loaded best model from epoch {best_epoch+1} (Val F1: {checkpoint['val_f1']:.4f})")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}. Using current model state.")
    else:
        print(f"Warning: Model file not found at {model_save_path}. Using current model state.")
        # Use the model as-is (it's already the best from training)
    
    # Test
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_precision, test_recall, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\n{'='*60}")
    print("Test Results")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Real', 'Fake']))
    
    # Save results
    results = {
        'test_metrics': {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1_score': float(test_f1)
        },
        'history': history,
        'best_epoch': best_epoch
    }
    
    try:
        with open(results_save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_save_path}")
    except Exception as e:
        print(f"Warning: Failed to save results: {e}")
    
    if os.path.exists(model_save_path):
        print(f"Best model saved to {model_save_path}")
    else:
        print(f"Warning: Model file not found at {model_save_path}")
    
    return model, history, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multimodal fake news classifier')
    parser.add_argument('--csv', type=str, default='processed_data/preprocessed_fakeddit.csv',
                        help='Path to preprocessed CSV file')
    parser.add_argument('--images', type=str, default='images',
                        help='Path to images directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Unfreeze last layers of BERT and ResNet for fine-tuning')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FakeReddit Multimodal Fake News Detection - Training")
    print("="*60)
    
    train_model(
        csv_path=args.csv,
        images_dir=args.images,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        test_split=args.test_split,
        fine_tune=args.fine_tune
    )

