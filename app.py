from flask import Flask, request, render_template_string, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load Keras model
MODEL_PATH = "best_model.keras"
model = load_model(MODEL_PATH)

# Class Labels
class_labels = [
    "rov", "trash_fabric", "plant", "trash_rubber", "trash_metal",
    "animal_fish", "animal_eel", "trash_etc", "trash_fishing_gear",
    "trash_paper", "trash_wood", "animal_starfish", "animal_shells",
    "animal_crab", "animal_etc", "trash_plastic"
]

# Image Preprocessing Function
def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Image Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; }
            .container { width: 50%; margin: auto; padding: 20px; border: 1px solid #ccc; border-radius: 10px; background: #f9f9f9; }
            input { margin-top: 10px; }
        </style>
    </head>
    <body>
    <div class="container">
        <h2>Sea Debris Image Classifier</h2>
        <form action="/" method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required><br>
            <input type="submit" value="Upload & Classify">
        </form>
        {% if prediction %}
            <h3>Prediction: {{ prediction }}</h3>
            <p>Confidence: {{ confidence }}</p>
        {% endif %}
        {% if error %}
            <p style="color:red;">{{ error }}</p>
        {% endif %}
    </div>
    </body>
    </html>
    """
    if request.method == "POST":
        if "file" not in request.files:
            return render_template_string(html_template, error="No file uploaded.")

        file = request.files["file"]

        if file.filename == "":
            return render_template_string(html_template, error="No file selected.")

        try:
            img_array = preprocess_image(file)
            predictions = model.predict(img_array)
            predicted_label = class_labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            return render_template_string(html_template, prediction=predicted_label, confidence=f"{confidence:.2f}%")

        except Exception as e:
            return render_template_string(html_template, error=f"Error processing image: {str(e)}")

    return render_template_string(html_template)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
