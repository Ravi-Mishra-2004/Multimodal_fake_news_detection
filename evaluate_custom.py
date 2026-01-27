"""
Evaluate model on custom CSV data.

CSV should have columns: id, clean_title (or text), image_path, 2_way_label (optional)
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset_setup import FakedditDataset
from train_model import MultimodalFakeNewsClassifier, get_device, validate
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import os


def evaluate_custom_csv(csv_path, images_dir, model_path='best_model.pth', batch_size=16):
    """Evaluate model on custom CSV file."""
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load CSV
    print(f"\nLoading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Check required columns
    required_cols = ['clean_title', 'image_path']
    if 'clean_title' not in df.columns and 'text' in df.columns:
        df['clean_title'] = df['text']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check if labels exist
    has_labels = '2_way_label' in df.columns
    
    # Create dataset
    print("Creating dataset...")
    dataset = FakedditDataset(csv_path, images_dir, text_field='clean_title')
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = MultimodalFakeNewsClassifier(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get class weights (if labels exist)
    if has_labels:
        from dataset_setup import get_class_weights
        class_weights = get_class_weights(csv_path)
        class_weights = class_weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluate
    print("\nEvaluating...")
    if has_labels:
        loss, accuracy, precision, recall, f1, preds, labels = validate(
            model, dataloader, criterion, device
        )
        
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(labels, preds, target_names=['Real', 'Fake']))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(labels, preds))
    else:
        # Just predictions
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                
                outputs = model(input_ids, attention_mask, images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Add predictions to dataframe
        df['prediction'] = ['Fake' if p == 1 else 'Real' for p in all_preds]
        df['fake_probability'] = [p[1] for p in all_probs]
        df['real_probability'] = [p[0] for p in all_probs]
        df['confidence'] = [max(p) for p in all_probs]
        
        # Save results
        output_path = csv_path.replace('.csv', '_predictions.csv')
        df.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print("Predictions Complete")
        print(f"{'='*60}")
        print(f"Total predictions: {len(df)}")
        print(f"Real: {(df['prediction'] == 'Real').sum()}")
        print(f"Fake: {(df['prediction'] == 'Fake').sum()}")
        print(f"\nResults saved to: {output_path}")
    
    return df if not has_labels else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on custom CSV data')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV file with columns: clean_title, image_path, (optional: 2_way_label)')
    parser.add_argument('--images', type=str, default='images',
                       help='Base directory for images (if image_path is relative)')
    parser.add_argument('--model', type=str, default='best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file {args.csv} not found!")
        exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        exit(1)
    
    print("="*60)
    print("Custom Data Evaluation")
    print("="*60)
    
    evaluate_custom_csv(
        csv_path=args.csv,
        images_dir=args.images,
        model_path=args.model,
        batch_size=args.batch_size
    )

