# evaluate.py

import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from inference import model, inference_transform, DEVICE

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "data"
BATCH_SIZE = 32

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# -----------------------
# DATASET
# -----------------------
test_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/test",
    transform=inference_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------
# EVALUATION LOOP
# -----------------------
def evaluate():
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # -----------------------
    # METRICS
    # -----------------------
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary")
    rec = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    cm = confusion_matrix(all_labels, all_preds)

    # -----------------------
    # OUTPUT
    # -----------------------
    print("\n==============================")
    print("📊 TEST RESULTS")
    print("==============================")

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClass mapping:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{i} -> {name}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }


# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    evaluate()