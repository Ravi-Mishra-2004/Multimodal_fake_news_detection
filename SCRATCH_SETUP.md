# Scratch Setup Guide: FakeNews_MultiModal

This guide provides the step-by-step instructions to set up the project, preprocess the data, train the model from scratch, and run the web application.

## Prerequisites

- **Python 3.11 or 3.12**: This is required for compatibility with BERT and PyTorch on macOS.
- **Kaggle API Credentials**: Required to download the FakeReddit dataset.

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
# Create a virtual environment with Python 3.11
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m venv venv_311

# Activate the environment
source venv_311/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 2: Kaggle API Setup (If not already configured)

The preprocessing script downloads data from Kaggle. You need a `kaggle.json` file in `~/.kaggle/`.

1. Go to your Kaggle account settings.
2. Click "Create New API Token".
4. Move the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`.
5. Run: `chmod 600 ~/.kaggle/kaggle.json`

## Step 3: Data Preprocessing

Run the preprocessing script to download the dataset, sample rows, and process images:

```bash
python preprocess_fakeddit.py
```

*This will create the `processed_data/` and `images/` directories.*

## Step 4: Model Training

Train the multimodal model (BERT + ResNet). For better performance, use the `--fine_tune` flag:

```bash
# Basic training
python train_model.py --epochs 10

# Recommended: Fine-tuned training (last layers unfrozen)
python train_model.py --fine_tune --epochs 10
```

*This will save the best model weights to `best_model.pth`.*

## Step 5: Start the Web Application

Once `best_model.pth` is generated, start the Flask app:

```bash
python app.py
```

Access the UI at: `http://localhost:5001`
