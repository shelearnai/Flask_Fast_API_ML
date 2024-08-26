from flask import Flask, request, jsonify, render_template
from tensorflow import keras
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import io

# Initialize the Flask app
app = Flask(__name__)

def get_best_model():
    model = keras.models.load_model('rps_model.h5', compile=False)
    model.make_predict_function()  # Necessary
    print('Model loaded. Start serving...')
    return model

# Route to serve the index.html
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for image classification
@app.route('/predictimage', methods=['POST'])
def classify_image():
    classlabel = ['paper', 'rock', 'scissors']
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Load the image from the request
    image_file = request.files['image']

    # Convert the uploaded file to a BytesIO object
    image_bytes = io.BytesIO(image_file.read())

    # Load and preprocess the image
    img = Image.open(image_bytes)
    img = img.resize((150, 150))  # Resize the image to match the input size expected by the model
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension

    model = get_best_model()
    classes = model.predict(img_arr)
    label_position = np.argmax(classes)
    pred_value = classlabel[label_position]

    return jsonify({"predicted_class": pred_value})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
