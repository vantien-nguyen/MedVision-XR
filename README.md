# PulmoVision 🫁📸  
**AI-Powered Chest X-Ray Classification for COVID-19 and Lung Diseases**

PulmoVision is a deep learning project that uses **transfer learning with EfficientNetB0** to classify chest X-ray images into multiple pulmonary conditions, including **COVID-19**, **Normal**, **Viral Pneumonia**, and **Lung Opacity**.

The goal of this project is to demonstrate how modern convolutional neural networks can assist in **medical image analysis** and **clinical decision support**.

---

## 📌 Features

- Multi-class chest X-ray classification (4 classes)
- Transfer learning with **EfficientNetB0**
- Implemented using **TensorFlow / Keras**
- Trained and evaluated on the **COVID-19 Radiography Database**
- Optimized for **Google Colab (GPU)** and local execution
- Clean, reproducible training pipeline

---

## 🗂 Dataset

**COVID-19 Radiography Database**  
📎 Source: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

### Classes Used
- COVID-19
- Normal
- Viral Pneumonia
- Lung Opacity

### Dataset Structure

```
data/
├── COVID/
├── NORMAL/
├── Viral Pneumonia/
└── Lung Opacity/
```

---

## 🧠 Model Architecture

- **Base Model:** EfficientNetB0 (ImageNet pretrained)
- **Input Size:** 224 × 224 × 3
- **Pooling:** Global Max Pooling
- **Regularization:** L1 + L2
- **Dropout:** 0.45
- **Optimizer:** Adamax
- **Loss:** Categorical Cross-Entropy

```text
EfficientNetB0 (frozen)
→ Batch Normalization
→ Dense (256, ReLU)
→ Dropout (0.45)
→ Dense (Softmax, 4 classes)
```

--- Useful commands ---
# Activate base conda
```
source ~/miniforge3/bin/activate
```

# Create and activate conda environment
```
conda create -n covid_ai python=3.11
conda activate covid_ai
pip install -r requirements.txt
```