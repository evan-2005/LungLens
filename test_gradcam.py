import torch
from app import CNNModel, predict_image
from PIL import Image
import numpy as np

# Create a grayscale dummy image (1 channel)
img = Image.fromarray(np.uint8(np.random.rand(224, 224) * 255))

try:
    print("Testing prediction and Grad-CAM with Grayscale Image...")
    result, heatmap = predict_image(img, "Pneumonia")
    print("Prediction result:", result)
    print("Heatmap shape (should be HxWx3):", heatmap.shape if heatmap is not None else None)
except Exception as e:
    import traceback
    traceback.print_exc()
