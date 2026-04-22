import os

import cv2
import numpy as np
import torch
from PIL import Image

from model.inference import DEVICE, get_model, inference_transform


def generate_gradcam(image_path):
    model = get_model()
    gradients = None
    activations = None

    def forward_hook(_, __, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(_, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    target_layer = model.layer4[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        model.zero_grad(set_to_none=True)
        output[0, pred_class].backward()

        if gradients is None or activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture model state.")

        pooled_gradients = gradients.mean(dim=[0, 2, 3])
        weighted_activations = activations.squeeze(0).clone()
        weighted_activations *= pooled_gradients[:, None, None]

        heatmap = weighted_activations.mean(dim=0).clamp(min=0).cpu().numpy()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image at {image_path}")

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = np.clip((heatmap * 0.4) + img, 0, 255).astype(np.uint8)

        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_gradcam{ext}"
        cv2.imwrite(output_path, superimposed_img)

        return output_path
    finally:
        forward_handle.remove()
        backward_handle.remove()
