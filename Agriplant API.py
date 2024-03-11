from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import os
import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image
from keras.models import load_model
import warnings
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load your trained model "Corn Model"
model_path = 'modules/backed/corn.h5'
model = load_model(model_path)

# load the Paddy efficientNet model
model_Paddy = load_model('modules/backed/efficiemtnet.h5')

# Load Paddy ResNet34 Model
model_paddy = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
model_paddy.fc = nn.Linear(512, 10)

# Load the saved model state dictionary
model_paddy.load_state_dict(torch.load('modules/backed/paddy.pt', map_location=torch.device('cpu')))

# Set the model in evaluation mode
model_paddy.eval()


def transform_image(image_bytes):
    # Define the transformations
    my_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Open the image using PIL
    img = Image.open(image_bytes)

    # Apply the transformations
    preprocessed_image = my_transforms(img)

    # Add a batch dimension to match the model's input shape
    preprocessed_image = preprocessed_image.unsqueeze(0)

    return preprocessed_image


# Allow files with extension png, jpg and jpeg
ALLOWED_EXT = {'jpg', 'jpeg', 'png'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXT


# Function for image preprocessing
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))  # Resize the image to match model input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    return img_array


# Function for making predictions
def predict_disease_corn(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence


@app.route('/corn', methods=['POST'])
def predict_corn():
    try:
        # Get the image file from the request
        file = request.files['file']

        # Preprocess the image
        img = preprocess_image(file)

        # Make predictions
        predicted_class, confidence = predict_disease_corn(img)

        # Map the predicted class to the corresponding disease
        class_names = [
            'Blight',
            'Common Rust',
            'Gray Leaf Spot',
            'Healthy'
        ]
        predicted_disease = class_names[predicted_class]

        # Return the prediction result
        result = {'predicted_disease': predicted_disease, 'confidence': confidence}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/paddy', methods=['POST', 'GET'])
def predict_paddy():
    global disease, result
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format

            # preprocessing method
            img = transform_image(file)

            # Perform inference using the model
            with torch.no_grad():
                output = model_paddy(img)
                _, predicted = torch.max(output, 1)
                class_x = predicted.item()

            disease_mapping = {
                0: "Bacterial Leaf Blight",
                1: "Bacterial Leaf Streak",
                2: "Bacterial Panicle Blight",
                3: "Blast",
                4: "Brown Spot",
                5: "Dead Heart",
                6: "Downy Mildew",
                7: "Hispa",
                8: "Normal",
                9: "Tungro"
            }

            paddy_disease = disease_mapping.get(class_x, "Unknown Disease")

            confidence_rounded = float(predicted) * 10

            # Return the prediction result
            result = {'predicted_disease': paddy_disease, 'confidence': confidence_rounded}
        return jsonify(result)
    else:
        return "Unable to read the file. Please check file extension"


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
