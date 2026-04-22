# inference.py

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# -----------------------
# CONFIG
# -----------------------
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "1"))

if DEVICE.type == "cpu":
    torch.set_num_threads(TORCH_NUM_THREADS)
    torch.set_num_interop_threads(1)

# Update this path if needed
MODEL_PATH = os.path.join(os.path.dirname(__file__), "resnet18_best.pth")


# Class names (must match training folder names EXACTLY)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# -----------------------
# TRANSFORM (same as val)
# -----------------------
inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------
# LOAD MODEL
# -----------------------
def load_model():
    model = models.resnet18(weights=None)

    # Replace final layer (must match training)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model


# Load once globally (efficient for API)
model = None

def get_model():
    global model
    if model is None:
        model = load_model()
    return model


# -----------------------
# PREDICT FUNCTION
# -----------------------
def predict(image_path):
    """
    Input: path to image
    Output: dict with prediction + confidence
    """
    model= get_model()
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Apply transform
    image = inference_transform(image).unsqueeze(0).to(DEVICE)

    # Forward pass
    with torch.inference_mode():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    prediction = CLASS_NAMES[predicted.item()]
    confidence = confidence.item()

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }


# -----------------------
# TEST RUN (optional)
# -----------------------
if __name__ == "__main__":
    test_image = "IM-0001-0001.jpeg" 
    result = predict(test_image)
    print(result)
