from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import tensorflow as tf
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__, static_folder='docs', template_folder='docs')

# Load the pre-trained model
model = tf.keras.models.load_model('handwritten.h5')

@app.route('/')
def home():
    # Render index.html from the docs directory
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    
    # Decode and convert the image to grayscale
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')
    
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))

    # Convert to NumPy array and invert the colors (if necessary)
    image = np.array(image)
    image = np.invert(image)  # Invert to ensure the digit is black on white background

    # Normalize the image to the range [0, 1]
    image = image / 255.0

    # Reshape to match model input
    image = image.reshape(1, 28, 28, 1)

    # Predict the digit
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    return jsonify({'prediction': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)