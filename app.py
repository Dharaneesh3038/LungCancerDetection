from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
model = load_model('resnet50_model.keras')  # Load the trained model
img_height, img_width = 224, 224  # Match training dimensions

# Define class labels
class_labels = ['Benign', 'Malignant', 'Normal']

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
            'confidence': f'{confidence:.2f}'
        })

@app.route('/metrics', methods=['GET'])
def metrics():
    # Load metrics and confusion matrix
    metrics_data = np.load('metrics.npy', allow_pickle=True).item()
    
    return jsonify({
        'precision': metrics_data['precision'],
        'recall': metrics_data['recall'],
        'f1_score': metrics_data['f1_score'],
        'confusion_matrix': metrics_data['confusion_matrix'].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)