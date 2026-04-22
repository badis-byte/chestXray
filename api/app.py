# app.py
import logging
import os
import time
import uuid

from flask import Flask, jsonify, request, send_file, url_for
from werkzeug.utils import secure_filename

from api.logger import log_event
from model.gradcam import generate_gradcam
from model.inference import get_model, predict

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# -----------------------
# CONFIG
# -----------------------
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def warm_up_model():
    start = time.perf_counter()
    get_model()
    logger.info("Model warm-up completed in %.2fs", time.perf_counter() - start)


warm_up_model()

# -----------------------
# HELPERS
# -----------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def wants_gradcam(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
    request_started_at = time.perf_counter()

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
        logger.info("Saved upload to %s", filepath)

        try:
            predict_started_at = time.perf_counter()
            result = predict(filepath)
            logger.info(
                "Prediction finished in %.2fs for %s",
                time.perf_counter() - predict_started_at,
                filename,
            )

            gradcam_url = None
            include_gradcam = wants_gradcam(request.form.get("include_gradcam"))
            gradcam_started_at = time.perf_counter()
            if include_gradcam:
                gradcam_path = generate_gradcam(filepath)
                gradcam_url = url_for(
                    "get_gradcam",
                    filename=os.path.basename(gradcam_path),
                    _external=True,
                )
                logger.info(
                    "Grad-CAM finished in %.2fs for %s",
                    time.perf_counter() - gradcam_started_at,
                    filename,
                )
            else:
                logger.info("Grad-CAM skipped for %s", filename)

            response = {
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "explanation": generate_explanation(result["prediction"]),
                "gradcam_requested": include_gradcam,
                "gradcam_image": gradcam_url,
            }
            log_event(response)
            logger.info(
                "Request completed in %.2fs for %s",
                time.perf_counter() - request_started_at,
                filename,
            )

            return jsonify(response)

        except Exception as e:
            logger.exception("Prediction request failed for %s", filename)
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
