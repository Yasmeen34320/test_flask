import os
from flask import Flask, jsonify, request
from PIL import Image
import pytesseract
import subprocess
import cv2
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load the model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Route for OCR
@app.route('/api/perform_ocr/<language>', methods=['PUT'])
def perform_ocr(language):
    try:
        # Get the image file from the request
        file = request.files['image']
        custom_config = r'--oem 3 --psm 6 -l ara'
        
        # Save the image to a temporary file
        temp_image_path = 'temp_image.png'
        file.save(temp_image_path)

        # Open the image using PIL
        image = Image.open(temp_image_path)

        # Perform OCR using Tesseract
        text = pytesseract.image_to_string(image)

        # Return the extracted text
        return jsonify(text)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for color detection
@app.route('/api/perform_color', methods=['PUT'])
def perform_color():
    try:
        file = request.files['image']
        temp_image_path = 'edittt.jpg'  # Temporary file path
        file.save(temp_image_path)

        img = cv2.imread(temp_image_path)  # Read the image using OpenCV

        if img is None:
            return jsonify({'error': 'Failed to read image'}), 500

        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Detect color logic here...

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for image captioning
@app.route('/api/image_caption', methods=['PUT'])
def image_caption():
    try:
        file = request.files['image']
        image = Image.open(file)

        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        caption = predict_caption(image)

        return jsonify(caption)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Function to predict image caption
def predict_caption(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    captions = [caption.strip() for caption in captions]
    return captions[0] if captions else "No caption generated"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False)
