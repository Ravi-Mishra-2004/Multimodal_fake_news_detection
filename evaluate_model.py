"""
Evaluate the trained model on the test set or custom data.
Can load from checkpoint or use a freshly trained model.
Supports both CSV evaluation and single prediction mode.
"""

import torch
import argparse
from train_model import get_device, validate
from dataset_setup import create_dataloaders, get_class_weights
from train_model import MultimodalFakeNewsClassifier
from sklearn.metrics import classification_report
import os


def evaluate(csv_path, images_dir, model_path=None, batch_size=16, 
             val_split=0.2, test_split=0.1):
    """Evaluate the model on test set."""
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
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
    
    # Initialize model
    print("\nInitializing model...")
    model = MultimodalFakeNewsClassifier(num_classes=2)
    model = model.to(device)
    
    # Load model if path provided
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"Loaded model from epoch {checkpoint['epoch']+1}")
        if 'val_f1' in checkpoint:
            print(f"Model validation F1: {checkpoint['val_f1']:.4f}")
    else:
        print("Using randomly initialized model (train first to get trained weights)")
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Evaluate on test set
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
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--csv', type=str, default='processed_data/preprocessed_fakeddit.csv',
                        help='Path to preprocessed CSV file')
    parser.add_argument('--images', type=str, default='images',
                        help='Path to images directory')
    parser.add_argument('--model', type=str, default='best_model.pth',
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FakeReddit Multimodal Fake News Detection - Evaluation")
    print("="*60)
    
    evaluate(
        csv_path=args.csv,
        images_dir=args.images,
        model_path=args.model if os.path.exists(args.model) else None,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split
    )

