import torch
from app import predict_image
from PIL import Image
import numpy as np

# Create a grayscale dummy image (1 channel)
img = Image.fromarray(np.uint8(np.random.rand(224, 224) * 255))

try:
    print("Testing prediction and Segmentation overlay...")
    findings_text, prob_dict, heatmap_update = predict_image(img, "Pneumonia")
    print("=== Diagnostic Findings ===")
    print(findings_text)
    print("=== Class Confidence ===")
    print(prob_dict)

    # gr.update(value=...) returns a dict-like update; pull the image back out.
    superimposed = heatmap_update.get("value") if isinstance(heatmap_update, dict) else heatmap_update
    if superimposed is not None:
        superimposed_np = np.array(superimposed)
        print("Superimposed Image shape (should be 224x224x3):", superimposed_np.shape)
        print("Verification Successful!")
    else:
        print("Error: Superimposed image is None!")
except Exception as e:
    import traceback
    traceback.print_exc()
