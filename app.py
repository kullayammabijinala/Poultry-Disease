import pickle
import numpy as np
import cv2
import os
import json
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import joblib
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = None
disease_info = {}
# class_labels is not directly used in this snippet, but kept for context if needed by the model
class_labels = [] 

# Load disease information
try:
    with open("disease_info.json", "r") as f:
        disease_info = json.load(f)
except FileNotFoundError:
    print("Error: disease_info.json not found. Make sure it's in the same directory as app.py.")
    exit()
except json.JSONDecodeError:
    print("Error: Could not decode disease_info.json. Check its formatting.")
    exit()

# Load the model and its class labels from the dictionary saved in model.pkl
try:
 
        model = joblib.load("model_compressed.pkl")

        print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl not found. Please run train_model.py first to create it.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading model.pkl: {e}")
    exit()

def preprocess_image(image_path):
    """
    Preprocesses an image for model prediction.
    Resizes, normalizes, and flattens the image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or could not be read at {image_path}")
    img = cv2.resize(img, (100, 100))
    img = img.astype("float32") / 255.0 # Ensure consistency with training
    img_array = img.flatten()
    img_array = img_array.reshape(1, -1) # Reshape for single prediction (1 sample, N features)
    return img_array

@app.route('/')
def index():
    """Renders the main home page."""
    return render_template('index.html')

# This route now renders result.html instead of predict.html
@app.route('/predict_page')
def predict_page():
    """Renders the dedicated page for image upload and prediction initiation, now using result.html."""
    # Note: result.html expects 'filename', 'disease', and 'info' variables.
    # When accessing /predict_page directly, these won't be available.
    # You might want to consider redirecting to index or handling default values in result.html
    # if a direct access to /predict_page is not meant to show a result.
    # For now, it will render result.html without prediction data.
    # It's recommended that /predict_page serves the 'predict.html' (upload form)
    # and '/predict' (POST) serves the 'result.html'.
    # If the user's intent is to have the *upload form* directly on result.html,
    # then the content of predict.html should be moved to result.html,
    # and this route would simply render result.html.
    
    # As per the request, simply changing the template rendered:
    return render_template('result.html', filename=None, disease="No prediction yet", info={
        "symptoms": "Upload an image to see results.",
        "features": "Upload an image to see results.",
        "treatment": "Upload an image to see results.",
        "preventions": "Upload an image to see results."
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the image upload, preprocesses it, makes a prediction,
    and displays the results.
    """
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            image_data = preprocess_image(filepath)

            # Predict the class index
            # SVC.predict() directly returns the predicted class label (string) 
            # since y_train contained strings during model training.
            predicted_class_label = model.predict(image_data)[0]
            print(f"Predicted class label: {predicted_class_label}")

            # Retrieve information using the predicted class label (which should be lowercase)
            info = disease_info.get(predicted_class_label.lower(), {
                "symptoms": "N/A (Information not available for this disease)",
                "features": "N/A",
                "treatment": "N/A",
                "preventions": "N/A" # Include preventions in the default if not found
            })

            return render_template('result.html', filename=filename, disease=predicted_class_label, info=info)
        except FileNotFoundError as e:
            # Handle cases where the uploaded file somehow isn't found during preprocessing
            return f"Error processing image: {e}", 500
        except Exception as e:
            # Catch any other unexpected errors during prediction
            print(f"Error during prediction: {e}")
            return f"An error occurred during prediction: {e}", 500

if __name__ == '__main__':
    # Run the Flask app in debug mode (useful for development)
    app.run(debug=True)
