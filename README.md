# LungLens

**LungLens** is an advanced, Apple-inspired clinical diagnostic web application designed to perform joint classification and semantic segmentation of chest X-rays across four distinct categories: **Normal**, **Pneumonia**, **Tuberculosis**, and **Covid-19**.

Powered by a robust **Multi-Task U-Net** deep learning architecture and wrapped in a beautiful, highly polished Gradio interface, LungLens provides both clinical probability metrics and **precise segmentation mask outlines** (with fallback support for pre-trained Grad-CAM localization) to ensure the AI's diagnostic reasoning is completely transparent to clinicians.

---

## 🌟 Key Features

- **Clinical-Grade AI Inference**: Built on a multi-task semantic segmentation architecture (Multi-Task U-Net) trained for thoracic pathology detection.
- **Precise Lesion Segmentation**: Automatically segments and overlays the exact localized pathology regions (e.g., fluid consolidation, focal lesions, cavities) as contoured masks rather than fuzzy saliency maps.
- **Human-Readable Findings**: Translates complex probabilistic outputs into plain, clinical English sentences.
- **Sleek Apple-Aesthetic UI**: Features an ultra-minimalist design with SF Pro typography, smooth animations, and a strict monochrome palette accented by medical blue.
- **Built-in Training Dashboard**: A dedicated interface allowing users to natively train the Multi-Task U-Net on local or downloaded datasets directly from the browser without touching a line of code.

---

## 💾 Datasets Used

LungLens uses a combination of high-quality, open-source medical datasets pulled automatically via KaggleHub during training:

1. **Tuberculosis Chest X-ray Database**
   - *Source*: [tawsifurrahman/tuberculosis-tb-chest-xray-dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
   - *Contents*: Confirmed Tuberculosis positive scans and normal healthy scans.
2. **Pneumonia X-Ray Images**
   - *Source*: [pcbreviglieri/pneumonia-xray-images](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images)
   - *Contents*: Thousands of pediatric and adult scans exhibiting viral and bacterial pneumonia alongside normal lungs.
3. **COVID-19 Chest X-ray Positive Tests**
   - *Source*: [raddar/ricord-covid19-xray-positive-tests](https://www.kaggle.com/datasets/raddar/ricord-covid19-xray-positive-tests)
   - *Contents*: Confirmed positive COVID-19 chest scans.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system along with `pip`.

### 2. Clone the Repository
```bash
git clone https://github.com/evan-2005/LungLens.git
cd LungLens
```

### 3. Install Dependencies
Install the required Python libraries. It is recommended to use a virtual environment.
```bash
pip install torch torchvision gradio opencv-python numpy scikit-learn kagglehub
```
*(Note: If you have a dedicated NVIDIA GPU, make sure to install the CUDA-enabled version of PyTorch from [pytorch.org](https://pytorch.org/) to significantly speed up training.)*

### 4. Launch the App
Run the main application file from your terminal:
```bash
python app.py
```
Open your web browser and navigate to **[http://127.0.0.1:7860](http://127.0.0.1:7860)**.

---

## 🚀 How to Use

### Diagnostic Inference (Testing)
1. Navigate to the **Diagnostic Inference** tab.
2. Drag and drop a chest X-ray image into the upload zone.
3. Click **Diagnose**.
4. The system will reveal the findings, probabilities, and the segmented pathology mask outline overlay.

### Model Training (Improving Accuracy)
When you first run the app, it will automatically load the fallback pre-trained DenseNet-121 classifier (`chest_model_4class.pth`) and threshold its Grad-CAM outputs to show segmentation outlines. To train the full Multi-Task U-Net segmentation model:
1. Navigate to the **Model Training** tab.
2. Adjust the **Dataset Size** slider (e.g., 2000+ images) and set **Epochs** to 5 or more.
3. Click **Start Training**. The terminal logs will update on the right side of the screen.
4. Once training finishes, the new segmentation model is automatically saved as `chest_segmentation_model.pth` and loaded for direct feedforward segmentation inference.

### Adding Custom Datasets
You can easily train the model on your own X-ray images!
1. Place your images inside the `custom_dataset` folder.
2. Ensure the file names or their parent folder names contain the disease name (`normal`, `pneumonia`, `tuberculosis`, or `covid`).
3. Click **Start Training** in the dashboard. The script will automatically scan your custom folder and incorporate them into the training pipeline.
