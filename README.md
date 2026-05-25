# Tomato Disease Classification using CNN & Transfer Learning (VGG16)

A deep learning project that classifies tomato leaf diseases from images using a custom-built Convolutional Neural Network (CNN) and VGG16 transfer learning.

---

## Problem Statement

Tomato crops are vulnerable to multiple diseases that reduce yield and quality. Manual identification is slow and error-prone. This project automates disease detection by classifying tomato leaf images into 10 disease/health categories.

---

## Dataset

**Source:** [PlantVillage Tomato Dataset on Google Drive](https://drive.google.com/drive/folders/1-7Ha8VR0aHjOx1SFPLdJZamI1qLsTj0r?usp=drive_link)

The dataset is organized into three splits:

```
Tomato/
├── train/
├── validation/
└── test/
```

Each split contains subdirectories named by class (e.g., `Tomato_healthy`, `Tomato_Early_blight`, etc.).

**Classes (10 total):**
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Spider Mites (Two-spotted spider mite)
- Tomato Target Spot
- Tomato Tomato Yellow Leaf Curl Virus
- Tomato Tomato Mosaic Virus
- Tomato Healthy

---

## Project Structure

```
├── ANN_Tomato.ipynb       # Main notebook: EDA, preprocessing, training, evaluation
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Class distribution across train/val/test splits
- Image size and dimension analysis
- Pixel intensity and RGB channel distribution
- Sample image visualization
- Corrupted image detection

### 2. Preprocessing & Augmentation
- Resize all images to 224×224 (custom CNN) or 160×160 (VGG16)
- Normalize pixel values (rescale to [0, 1])
- Augmentation: rotation, zoom, flips, brightness adjustment, shear, width/height shifts
- Class weight computation to handle class imbalance

### 3. Model 1 — Custom CNN
- Multiple convolutional blocks with BatchNormalization and MaxPooling
- Dropout regularization
- Adam optimizer (lr=0.0003) with gradient clipping
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### 4. Model 2 — VGG16 Transfer Learning
- Pretrained on ImageNet; top layers removed
- Base model frozen; last 6 layers fine-tuned
- Custom head: GlobalAveragePooling → BatchNorm → Dense(128) → Dropout → Softmax
- Adam optimizer (lr=1e-5)

### 5. Evaluation
- Test accuracy and loss
- Confusion matrix (heatmap)
- Classification report: Precision, Recall, F1-score per class
- Training/validation accuracy and loss curves

---

## Results

| Model       | Input Size | Epochs | Notes                         |
|-------------|------------|--------|-------------------------------|
| Custom CNN  | 224×224    | ≤25    | EarlyStopping, ModelCheckpoint|
| VGG16       | 160×160    | ≤10    | Fine-tuned last 6 layers      |

> Actual accuracy values will appear after running the notebook with the dataset.

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/tomato-disease-classification.git
cd tomato-disease-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from the [Google Drive link](https://drive.google.com/drive/folders/1-7Ha8VR0aHjOx1SFPLdJZamI1qLsTj0r?usp=drive_link) and place it in the project root so the path is:
```
Tomato/train/
Tomato/validation/
Tomato/test/
```

### 4. Run the notebook
```bash
jupyter notebook ANN_Tomato.ipynb
```

---

## Requirements

See `requirements.txt` for full list. Key dependencies:
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- Pillow
- pandas, numpy, matplotlib, seaborn

---

## License

This project is for educational purposes. Dataset credit: [PlantVillage](https://plantvillage.psu.edu/).
