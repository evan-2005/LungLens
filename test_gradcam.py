import torch
from app import predict_image
from PIL import Image
import numpy as np

# Create a grayscale dummy image (1 channel)
img = Image.fromarray(np.uint8(np.random.rand(224, 224) * 255))

try:
    print("Testing prediction and Segmentation overlay...")
    findings_text, superimposed, upload_view_update, results_view_update = predict_image(img, "Pneumonia")
    print("=== Diagnostic Findings ===")
    print(findings_text)
    
    if superimposed is not None:
        superimposed_np = np.array(superimposed)
        print("Superimposed Image shape (should be 224x224x3):", superimposed_np.shape)
        print("Verification Successful!")
    else:
        print("Error: Superimposed image is None!")
except Exception as e:
    import traceback
    traceback.print_exc()
