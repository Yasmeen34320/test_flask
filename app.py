import os
from flask import Flask, jsonify, request
from PIL import Image
import pytesseract
import subprocess
# from google.colab.patches import cv2_imshow
import cv2
import numpy as np
#for imgee
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
#############
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
print ("runnninggggg")
# Run the command to install tesseract-ocr-ara
#subprocess.run(['sudo', 'apt-get', 'install', 'tesseract-ocr-ara'])
#for imgeee
# Load the model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#####3
app = Flask(__name__)

# # Set the path to Tesseract executable (update this with your actual Tesseract path)
#pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

@app.route('/api/perform_ocr/<language>', methods=['PUT'])
def perform_ocr(language):
    try:
        # Get the image file from the request
        file = request.files['image']
        custom_config = r'--oem 3 --psm 6 -l ara'
        # Save the image to a temporary file (you can customize the path as needed)
        temp_image_path = 'temp_image.png'
        file.save(temp_image_path)

        # Open the image using PIL
        image = Image.open(temp_image_path)
        print(language)
        # Perform OCR using Tesseract
        print("??????????????")
        text = pytesseract.image_to_string(image)

        # if language == 'en':
        #     text = pytesseract.image_to_string(image)
        #     print(text)
        # else:
        #     print("d?????????")
        #     text = pytesseract.image_to_string(image,config=custom_config)
            
        print(text+'dddddddddddddddddddddddd')
        print("Sdsd")

        # Return the extracted text
        return jsonify(text)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

color_ranges = {
    'Red': ([0, 50, 50], [10, 255, 255]),
    'Green': ([40, 20, 50], [90, 255, 255]),
    'Blue': ([100, 50, 50], [130, 255, 255]),
    'Yellow': ([20, 100, 100], [30, 255, 255]),
    'Purple': ([130, 50, 50], [160, 255, 255]),
    'Orange': ([10, 100, 100], [20, 255, 255]),
    'Cyan': ([85, 100, 100], [105, 255, 255]),
    'Magenta': ([160, 100, 100], [180, 255, 255]),
    'Pink': ([140, 50, 50], [160, 255, 255]),
    'Brown': ([0, 50, 10], [30, 255, 120]),
    'Lime': ([50, 50, 50], [70, 255, 255]),
    'Teal': ([80, 50, 50], [100, 255, 255]),
    'Olive': ([30, 50, 50], [50, 255, 255]),
    'Maroon': ([0, 50, 50], [10, 255, 120]),
    'Navy': ([100, 50, 50], [130, 255, 120]),
    'Turquoise': ([70, 100, 100], [90, 255, 255]),
    'Violet': ([130, 50, 50], [160, 255, 120]),
    'Indigo': ([90, 50, 50], [110, 255, 120]),
    'Beige': ([20, 50, 120], [30, 150, 255]),
    'Mint': ([50, 50, 120], [70, 150, 255]),
    'Black': ([0, 0, 0], [180, 255, 30]),
}

def get_contour_area(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w * h 

@app.route('/')
def hello():
    return "<h1>Hello, World!! </h1>"

@app.route('/api/perform_color', methods=['PUT'])
def perform_color():
    try:
        print("its stuckkkk")
        file = request.files['image']
        print("Received file:", file.filename)
      #  temp_image_path = 'D:/vs code/flutter/grd_project/grd_projecttt/lib/api/temp_image2.jpg'  # Temporary file path
        temp_image_path = 'edittt.jpg'  # Temporary file path
        
        file.save(temp_image_path)

        # Check if the file exists
        if os.path.exists(temp_image_path):
            print("File saved successfully:", temp_image_path)
        else:
            print("Failed to save file!")

        img = cv2.imread(temp_image_path)  # Read the image using OpenCV

        if img is not None:
            print("Image read successfully!")
        else:
            print("Failed to read image!")

        img = cv2.resize(img, (640, 480))

        input_image_cpy = img.copy()

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
       
        # Initialize variables to track the largest object
        largest_object_area = 0
        largest_object_color = None
        largest_object_coordinates = None

        # Function to update the largest object information
        def update_largest_object(contour, color):
            nonlocal largest_object_area, largest_object_color, largest_object_coordinates
            contour_area = get_contour_area(contour)
            if contour_area > largest_object_area:
                largest_object_area = contour_area
                largest_object_color = color
                largest_object_coordinates = cv2.boundingRect(contour)

        # Loop through each color range
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through the contours and find the largest object for each color
            for cnt in contours:
                contour_area = get_contour_area(cnt)
                if contour_area > 1000:  # You can adjust the area threshold as needed
                    update_largest_object(cnt, color)

        # If an object is detected, draw a rectangle and return the color
        if largest_object_area > 0:
            x, y, w, h = largest_object_coordinates
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, largest_object_color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            return jsonify( largest_object_color), 200
        else:
            return jsonify({'error': 'No object detected'}), 500

    except Exception as e:
        return jsonify({'errordfd': str(e)}), 500


# Define route for image captioning
@app.route('/api/image_caption', methods=['PUT'])
def image_caption():
    try:
        # Get image file from request
        file = request.files['image']

        # Open image and preprocess
        image = Image.open(file)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        # Perform image captioning
        caption = predict_caption(image)
        print('cptionnnnn' + caption)
        # Return caption
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

if __name__ == '__main__':
    app.run(debug=True, port=8000)

