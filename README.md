# LungLens (formerly Chestist)

**LungLens** is an advanced, Apple-inspired clinical diagnostic web application designed to classify chest X-rays into three distinct categories: **Normal**, **Pneumonia**, and **Tuberculosis (TB)**. 

Powered by a robust **DenseNet-121** deep learning architecture and wrapped in a beautiful, highly polished Gradio interface, LungLens provides both clinical probability metrics and **Grad-CAM visual heatmaps** to ensure the AI's diagnostic reasoning is completely transparent to clinicians.

---

## 🌟 Key Features

- **Clinical-Grade AI Inference**: Built on a pre-trained PyTorch DenseNet-121 model fine-tuned for thoracic pathology detection.
- **Grad-CAM Visual Explainability**: Automatically generates a heat map highlighting the exact localized areas of the lungs that the neural network used to determine its diagnosis (e.g., fluid consolidation, focal lesions).
- **Human-Readable Findings**: Translates complex probabilistic outputs into plain, clinical English sentences.
- **Sleek Apple-Aesthetic UI**: Features an ultra-minimalist design with frosted glassmorphism, SF Pro typography, smooth animations, and a strict monochrome palette accented by medical blue.
- **Built-in Training Dashboard**: A dedicated interface allowing users to natively retrain the neural network on local or downloaded datasets directly from the browser without touching a line of code.

---

## 💾 Datasets Used

LungLens uses a combination of high-quality, open-source medical datasets pulled automatically via KaggleHub during training:

1. **Tuberculosis (TB) Chest X-ray Database**
   - *Source*: [tawsifurrahman/tuberculosis-tb-chest-xray-dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
   - *Contents*: Confirmed TB positive scans and normal healthy scans.
2. **Pneumonia X-Ray Images**
   - *Source*: [pcbreviglieri/pneumonia-xray-images](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images)
   - *Contents*: Thousands of pediatric and adult scans exhibiting viral and bacterial pneumonia alongside normal lungs.

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
4. The system will slide over to reveal the findings, the probabilities, and the diagnostic heatmap. 

### Model Training (Improving Accuracy)
When you first run the app, it initializes a rapid "fast-mode" model on a tiny subset of images so you don't have to wait to see the UI. To make the model highly accurate:
1. Navigate to the **Model Training** tab.
2. Adjust the **Dataset Size** slider (e.g., 2000+ images) and set **Epochs** to 5 or more.
3. Click **Start Training**. The terminal logs will update on the right side of the screen.
4. Once training finishes, the new, highly accurate model is automatically saved as `chest_model_3class.pth` and loaded for inference.

### Adding Custom Datasets
You can easily train the model on your own X-ray images! 
1. Place your images inside the `custom_dataset` folder.
2. Ensure the file names or their parent folder names contain the disease name (`normal`, `pneumonia`, or `tb`/`tuberculosis`).
3. Click **Start Training** in the dashboard. The script will automatically scan your custom folder and incorporate them into the training pipeline.
