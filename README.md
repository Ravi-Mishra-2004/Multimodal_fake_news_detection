# FakeReddit Multimodal Fake News Detection

A complete pipeline for multimodal fake news detection using text and images. This project includes data preprocessing, model training, inference tools, and a web application.

## 🎯 Overview

This project implements a multimodal fake news detection system using:
- **Text**: BERT-base-uncased for text encoding
- **Images**: ResNet50 for image feature extraction
- **Fusion**: Concatenation + fully connected layers for binary classification (Real/Fake)

**Model Performance:**
- Test Accuracy: **78.43%**
- Test F1-Score: **0.7846**
- Test Precision: 0.7854
- Test Recall: 0.7843
- Best Model: Epoch 8 (Val F1: 0.8172)
- Dataset Size: 6,621 samples
- Training: Fine-tuned BERT (last layer) + ResNet50 (last block)

## 📁 Project Structure

```
FN_mm/
├── preprocess_fakeddit.py      # Data preprocessing pipeline
├── dataset_setup.py            # Dataset loading utilities
├── train_model.py              # Model training script
├── inference.py                # Command-line inference tool
├── evaluate_model.py           # Test set evaluation
├── evaluate_custom.py          # Custom data evaluation
├── export_model.py             # Model export tool
├── app.py                      # Flask web application
├── best_model.pth              # Trained model (524MB)
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # Web UI
├── static/
│   ├── css/style.css           # Styling
│   └── js/main.js              # Frontend logic
├── images/                     # Preprocessed images
├── processed_data/             # Preprocessed dataset
└── uploads/                    # Temporary uploads (web app)
```

## 🚀 Quick Start

### 1. Installation

**Note**: Python 3.11 or 3.12 recommended (Python 3.14 has known compatibility issues with PyTorch)

```bash
# Clone the repository
git clone <your-repo-url>
cd FN_mm

# Create virtual environment (recommended)
python3.11 -m venv venv  # or python3.12
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing

```bash
python preprocess_fakeddit.py
```

This will:
- Download FakeReddit dataset via kagglehub
- Filter and sample 10,000 rows (stratified)
- Download and validate images
- Preprocess images (256x256 RGB)
- Save preprocessed data to `processed_data/`

### 3. Train Model

```bash
# Default training (frozen layers, 10 epochs)
python train_model.py

# Fine-tuned training (recommended for better performance)
python train_model.py --fine_tune --epochs 10

# Custom parameters
python train_model.py --fine_tune --epochs 20 --batch_size 32 --lr 0.0001
```

### 4. Start Web Application

```bash
python app.py
```

Open browser: `http://localhost:5001`

## 📖 Usage

### Web Application

1. Start server: `python app.py`
2. Open `http://localhost:5001`
3. Enter news text/headline
4. Upload an image
5. Click "Check for Fake News"
6. View prediction with confidence scores

### Command Line Inference

```bash
python inference.py --text "Your news headline" --image path/to/image.jpg
```

**Output:**
```
Prediction: Fake
Confidence: 85.23%
Probabilities:
  Real: 14.77%
  Fake: 85.23%
```

### Evaluate Custom Data

```bash
python evaluate_custom.py --csv custom_data.csv --images images/
```

**CSV Format:**
- `clean_title` or `text`: News text
- `image_path`: Path to image file
- `2_way_label` (optional): Ground truth (0=Real, 1=Fake)

### Model Export

```bash
# Export PyTorch format
python export_model.py --format pytorch

# Export ONNX format (requires: pip install onnx)
python export_model.py --format onnx

# Export both
python export_model.py --format both
```

## 🔧 API Endpoints

### POST `/predict`
Predict fake news from text and image.

**Request:**
- `text` (form): News text/headline
- `image` (file): Image file (PNG, JPG, JPEG, GIF, WEBP)

**Response:**
```json
{
  "success": true,
  "prediction": "Fake",
  "confidence": 85.23,
  "fake_probability": 85.23,
  "real_probability": 14.77
}
```

### GET `/health`
Check server and model status.

## 📊 Dataset

- **Source**: [FakeReddit Dataset](https://www.kaggle.com/datasets/vanshikavmittal/fakeddit-dataset)
- **Task**: Binary classification (0=Real, 1=Fake)
- **Modalities**: Text (Reddit titles) + Images
- **Sample Size**: 10,000 rows (stratified sampling)

## 🏗️ Model Architecture

- **Text Branch**: BERT-base-uncased (feature extraction + optional fine-tuning)
- **Image Branch**: ResNet50 (feature extraction + optional fine-tuning)
- **Fusion**: Concatenation + fully connected layers
- **Output**: Binary classification (Real/Fake)

**Fine-tuning Options:**
- Default: All pre-trained layers frozen (feature extraction only)
- With `--fine_tune`: Last BERT layer + last ResNet block unfrozen
- Recommended: Use fine-tuning for better performance

**Parameters:**
- Total: 134,376,066
- Trainable (frozen): 1,385,794
- Trainable (fine-tuned): ~15M (includes last BERT layer + ResNet block)

## 📝 Training Options

```bash
python train_model.py --help

Options:
  --csv PATH          Path to preprocessed CSV
  --images PATH       Path to images directory
  --epochs N          Number of epochs (default: 10)
  --batch_size N      Batch size (default: 16)
  --lr FLOAT          Learning rate (default: 0.0001)
  --val_split FLOAT   Validation split (default: 0.2)
  --test_split FLOAT  Test split (default: 0.1)
  --fine_tune         Enable fine-tuning of last layers (recommended)
```

## 🖥️ Device Support

- **Apple Silicon Macs**: Uses MPS (Metal Performance Shaders) for GPU acceleration
- **Intel Macs**: Uses CPU
- **CUDA**: Automatically detected if available
- Script automatically selects best available device

## 🔍 Preprocessing Details

### Image Processing
- URL validation and download
- Resize to 256x256 pixels
- Convert to RGB format
- JPEG storage

### Text Processing
- Uses `clean_title` column
- BERT tokenization (max_length=128)
- Handles null/empty values

### Data Sampling
- Stratified sampling on `2_way_label`
- Maintains class distribution
- 10,000 rows (or all available if less)

## 🐛 Troubleshooting

### Python 3.14 Timeout Error
If you encounter `TimeoutError` when importing torch (especially on Python 3.14):
- **Solution 1**: Use Python 3.11 or 3.12 (recommended)
- **Solution 2**: Update packages: `pip install --upgrade torch torchvision`
- **Solution 3**: This is a Python 3.14 compatibility issue, not a code error
- The code works fine on Python 3.11/3.12

### Model Not Loading
- Ensure `best_model.pth` exists
- Check file permissions

### Image Upload Fails
- Max file size: 16MB
- Supported formats: PNG, JPG, JPEG, GIF, WEBP
- Check `uploads/` directory permissions

### Slow Training/Inference
- First prediction loads model (~2-3 seconds)
- Subsequent predictions faster (~0.5-1 second)
- Use GPU (MPS/CUDA) for acceleration
- Reduce batch size if memory issues

### Kaggle Download Fails
- Set up Kaggle API credentials
- See: https://github.com/Kaggle/kaggle-api

### Memory Issues
- Reduce `SAMPLE_SIZE` in preprocessing
- Reduce `batch_size` in training
- Use smaller model architecture

## 🚀 Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

### Environment Variables

```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
```

### Docker (Example)

```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "app:app"]
```

## 📈 Performance Tips

- First prediction: ~2-3 seconds (model loading)
- Subsequent predictions: ~0.5-1 second
- Batch processing: ~0.1-0.2 seconds per sample
- Export to ONNX for faster inference
- Use GPU acceleration when available

## 🔒 Security Notes

- File upload size limit: 16MB
- Allowed file types: PNG, JPG, JPEG, GIF, WEBP
- Uploaded files auto-deleted after processing
- Input validation on server side

## 📚 References

- [FakeReddit Dataset](https://github.com/entitize/Fakeddit)
- [Kaggle Notebook Reference](https://www.kaggle.com/code/hustzx/fakeddit-multimodal-fake-news-classification)

## 📄 License

This project is provided as-is. Please refer to the original FakeReddit dataset license for data usage terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.
