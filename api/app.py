# app.py
from model.gradcam import generate_gradcam
from api.logger import log_event

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")))
from inference import predict

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Import your inference function
from inference import predict

app = Flask(__name__)

# -----------------------
# CONFIG
# -----------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# -----------------------
# HELPERS
# -----------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_explanation(prediction):
    """
    Simple rule-based explanation (Digital Health Insight Layer)
    """
    if prediction == "PNEUMONIA":
        return "Model detected lung opacity patterns consistent with possible pneumonia. Clinical follow-up is recommended."
    else:
        return "No significant abnormal lung patterns detected. Result appears within normal range."


# -----------------------
# ROUTES
# -----------------------

@app.route("/gradcam/<filename>")
def get_gradcam(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))



@app.route("/")
def home():
    return jsonify({"message": "Medical AI API is running"})


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        try:
            result = predict(filepath)
            gradcam_path = generate_gradcam(filepath)

            response = {
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "explanation": generate_explanation(result["prediction"]),
                "gradcam_image": gradcam_path
            }
            log_event(response)

            return jsonify(response)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type"}), 400


# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)