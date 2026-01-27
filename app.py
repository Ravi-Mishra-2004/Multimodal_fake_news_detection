"""
Flask web application for FakeReddit Multimodal Fake News Detection

Upload an image and enter text to check if it's fake or real news.
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from inference import get_device, load_model, predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model once at startup
device = get_device()
model = None

def load_app_model():
    """Load the model for the app."""
    global model
    model_path = 'best_model.pth'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, device)
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file {model_path} not found!")

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    """Handle prediction request."""
    try:
        # Check if text is provided
        if 'text' not in request.form or not request.form['text'].strip():
            return jsonify({'error': 'Text is required'}), 400
        
        text = request.form['text'].strip()
        
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'Image is required'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, WEBP'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Make prediction
            result = predict(model, text, filepath, device)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'prediction': result['prediction'],
                'confidence': round(result['confidence'] * 100, 2),
                'fake_probability': round(result['fake_probability'] * 100, 2),
                'real_probability': round(result['real_probability'] * 100, 2)
            })
        
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

if __name__ == '__main__':
    load_app_model()
    print(f"\n{'='*60}")
    print("FakeReddit Fake News Detection Web App")
    print(f"{'='*60}")
    print(f"Server starting on http://localhost:5001")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    app.run(debug=True, host='0.0.0.0', port=5001)

