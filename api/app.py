# app.py
import os
import uuid

from flask import Flask, jsonify, request, send_file, url_for
from werkzeug.utils import secure_filename

from api.logger import log_event
from model.gradcam import generate_gradcam
from model.inference import predict

app = Flask(__name__)

# -----------------------
# CONFIG
# -----------------------
UPLOAD_FOLDER = "/tmp/uploads"
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


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = str(uuid.uuid4()) + "_" + filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)


        file.save(filepath)

        try:
            result = predict(filepath)
            gradcam_path = generate_gradcam(filepath)

            response = {
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "explanation": generate_explanation(result["prediction"]),
                "gradcam_image": url_for(
                    "get_gradcam",
                    filename=os.path.basename(gradcam_path),
                    _external=True,
                ),
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
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
