"""
Export model for deployment in different formats.
"""

import torch
import os
from inference import MultimodalFakeNewsClassifier, get_device


def export_pytorch(model_path, output_path='exported_model.pth'):
    """Export model in PyTorch format (lightweight, includes only state dict)."""
    device = get_device()
    
    # Load model
    model = MultimodalFakeNewsClassifier(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Save only state dict (smaller file)
    torch.save(model.state_dict(), output_path)
    print(f"✓ Exported PyTorch model to {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    return output_path


def export_onnx(model_path, output_path='exported_model.onnx', dummy_text="sample text", dummy_image_size=(1, 3, 256, 256)):
    """Export model to ONNX format (for production deployment)."""
    try:
        import torch.onnx
        device = get_device()
        
        # Load model
        model = MultimodalFakeNewsClassifier(num_classes=2)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(device)
        
        # Create dummy inputs
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded = tokenizer.encode_plus(
            dummy_text,
            add_special_tokens=True,
            max_length=80,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        dummy_input_ids = encoded['input_ids'].to(device)
        dummy_attention_mask = encoded['attention_mask'].to(device)
        dummy_image = torch.randn(dummy_image_size).to(device)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask, dummy_image),
            output_path,
            input_names=['input_ids', 'attention_mask', 'image'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'image': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=11
        )
        
        print(f"✓ Exported ONNX model to {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        return output_path
    except ImportError:
        print("⚠ ONNX export requires: pip install onnx")
        return None
    except Exception as e:
        print(f"⚠ ONNX export failed: {e}")
        return None


def create_model_info(model_path, output_path='model_info.json'):
    """Create model information file."""
    import json
    
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    
    info = {
        'model_type': 'MultimodalFakeNewsClassifier',
        'num_classes': 2,
        'architecture': {
            'text_backbone': 'BERT-base-uncased',
            'image_backbone': 'ResNet50',
            'fusion': 'Concatenation + FC layers'
        },
        'training_info': {
            'best_epoch': checkpoint.get('epoch', 'unknown'),
            'val_f1': checkpoint.get('val_f1', 'unknown')
        },
        'input_specs': {
            'text': {
                'max_length': 80,
                'tokenizer': 'bert-base-uncased'
            },
            'image': {
                'size': [256, 256],
                'channels': 3,
                'normalization': 'ImageNet stats'
            }
        },
        'output': {
            'classes': ['Real', 'Fake'],
            'format': 'probabilities'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✓ Created model info file: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export model for deployment')
    parser.add_argument('--model', type=str, default='best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--format', type=str, choices=['pytorch', 'onnx', 'both', 'info'],
                       default='both', help='Export format')
    parser.add_argument('--output_dir', type=str, default='exports',
                       help='Output directory for exported files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Model Export Tool")
    print("="*60)
    print(f"Input model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print("="*60 + "\n")
    
    if args.format in ['pytorch', 'both']:
        export_pytorch(args.model, os.path.join(args.output_dir, 'model.pth'))
    
    if args.format in ['onnx', 'both']:
        export_onnx(args.model, os.path.join(args.output_dir, 'model.onnx'))
    
    if args.format in ['info', 'both']:
        create_model_info(args.model, os.path.join(args.output_dir, 'model_info.json'))
    
    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)

