# gradcam.py

import torch
import numpy as np
import cv2
from torchvision import models
from PIL import Image
from model.inference import model, inference_transform, DEVICE

# Hook storage
gradients = None
activations = None

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def forward_hook(module, input, output):
    global activations
    activations = output


# Attach hooks to last conv layer of ResNet18
target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)


def generate_gradcam(image_path):
    global gradients, activations

    image = Image.open(image_path).convert("RGB")
    input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    # Forward
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)

    # Backward
    model.zero_grad()
    output[0, pred_class].backward()

    # Compute weights
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations_ = activations.squeeze(0)

    for i in range(len(pooled_gradients)):
        activations_[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations_, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    # Convert to image
    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img

    output_path = image_path.replace(".", "_gradcam.")
    cv2.imwrite(output_path, superimposed_img)

    return output_path