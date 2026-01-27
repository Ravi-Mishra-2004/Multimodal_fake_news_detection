"""
Inference script for FakeReddit Multimodal Fake News Detection

Use this to predict fake/real on new text + image pairs.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import BertModel, BertTokenizer
from PIL import Image
import os
import argparse


class MultimodalFakeNewsClassifier(nn.Module):
    """Multimodal classifier combining BERT for text and ResNet50 for images."""
    
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        
        # Text branch (BERT)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
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
        
        # Fusion
        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        
        return output


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model(model_path, device):
    """Load the trained model."""
    model = MultimodalFakeNewsClassifier(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, device):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")


def preprocess_text(text, device, max_length=128):
    """Preprocess text for model input."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    return input_ids, attention_mask


def predict(model, text, image_path, device):
    """Make a prediction on text + image."""
    # Preprocess inputs
    input_ids, attention_mask = preprocess_text(text, device)
    image_tensor = preprocess_image(image_path, device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Map class to label
    label = "Fake" if predicted_class == 1 else "Real"
    fake_prob = probabilities[0][1].item()
    real_prob = probabilities[0][0].item()
    
    return {
        'prediction': label,
        'confidence': confidence,
        'fake_probability': fake_prob,
        'real_probability': real_prob,
        'class': predicted_class
    }


def main():
    parser = argparse.ArgumentParser(description='Predict fake news from text and image')
    parser.add_argument('--model', type=str, default='best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--text', type=str, required=True,
                       help='Text/title to analyze')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file')
    
    args = parser.parse_args()
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)
    print("Model loaded successfully!")
    
    # Make prediction
    print(f"\nAnalyzing text: '{args.text}'")
    print(f"Image: {args.image}")
    print("\n" + "="*60)
    
    try:
        result = predict(model, args.text, args.image, device)
        
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"\nProbabilities:")
        print(f"  Real: {result['real_probability']*100:.2f}%")
        print(f"  Fake: {result['fake_probability']*100:.2f}%")
        print("="*60)
        
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()

