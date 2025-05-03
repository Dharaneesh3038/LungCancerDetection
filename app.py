from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
model = load_model('lung_cancer_resnet50_model.keras')  # Load the trained model
img_height, img_width = 224, 224  # Match training dimensions

# Define class labels (adjust based on your training data)
class_labels = ['Benign cases', 'Malignant cases', 'Normal cases']  # Update if 4 classes

def preprocess_image(img):
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Load and preprocess image
        img = Image.open(file.stream).convert('RGB')
        processed_img = preprocess_image(img)
        
        # Make prediction
        prediction = model.predict(processed_img)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        return jsonify({
            'class': predicted_class,
            'confidence': f'{confidence:.2f}%'
        })

@app.route('/metrics', methods=['GET'])
def metrics():
    # Placeholder metrics (replace with actual values from your model evaluation)
    precision = 0.87  # Example macro precision
    recall = 0.84     # Example macro recall
    f1_score = 0.85   # Example macro F1 score
    confusion_matrix = [
        [50, 5, 2],  # Benign: [TP, FP, FP]
        [3, 45, 4],  # Malignant: [FP, TP, FP]
        [1, 2, 48]   # Normal: [FP, FP, TP]
    ]
    
    return jsonify({
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': confusion_matrix
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)