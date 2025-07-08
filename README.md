# Multi-Task ViT and CNN Architectures for Diabetic Retinopathy Severity Classification

## Team Members

- Ata Güneş: [GitHub](https://github.com/AtaGn) 
- Esra Cesur: [GitHub](https://github.com/esracesur4)

## Project Overview

Diabetic Retinopathy (DR) is a leading cause of blindness. Early and accurate diagnosis is crucial for intervention.  
This project presents five progressively advanced deep learning models for classifying DR severity from retinal fundus images:

- EfficientNet-based CNNs  
- Vision Transformer (ViT) architectures  
- Multi-task learning (classification + regression + ordinal)  
- Two-stage cascaded classification pipelines  
- Focal Loss to handle class imbalance  

---

## 🔗 Colab Notebook

Run or explore the notebook:  
[Google Colab Notebook](https://colab.research.google.com/drive/1jKgU-lRVS0r3xyfWo-FZ18Kfub5JIQ6T?usp=sharing)

---

## Dataset

![image](https://github.com/user-attachments/assets/4f96cbb1-2015-4e16-bcf2-5022fda74cae)

- **Source**: Kaggle (Diabetic Retinopathy Classification #3)
- **Training Images**: 2,197  
- **Test Images**: 1,465  
- **Classes**:
  - `0`: No DR  
  - `1`: Mild DR  
  - `2`: Moderate DR  
  - `3`: Severe DR  
  - `4`: Proliferative DR

![image](https://github.com/user-attachments/assets/5fbf8b5d-04c6-4041-b9a3-4c500121eb29)

---

## Exploratory Data Analysis (EDA)

- Distribution showed strong class imbalance, requiring special strategies like Focal Loss and two-stage pipelines.
- Visual inspections and class-wise EDA were performed to guide preprocessing choices.

---

## Image Preprocessing & Augmentation

| Technique                   | Purpose                                            |
|----------------------------|----------------------------------------------------|
| CLAHE                      | Enhance micro-lesions and vessels                  |
| Green Channel Extraction   | Highlight vascular contrast                        |
| Gaussian Blur (Ben Graham) | Sharpen edge-level features                        |
| Resizing & Normalization   | Prepare inputs for model compatibility             |
| Data Augmentation          | Flip, rotate, affine for generalization            |

---

## Model Architectures

### 🔹 Model 1: EfficientNetB3
- Simple 5-class classifier
- Green channel preprocessing
- CrossEntropy loss

### 🔹 Model 2: Three-Headed EfficientNetB4
- Multi-task learning with classification, regression, and ordinal heads
- Combiner model for final prediction
- Weighted CrossEntropy, MSE, and BCEWithLogits

### 🔹 Model 3: EfficientNetB4 + Focal Loss
- Same architecture as Model 2 but focused only on classification
- Improved rare class prediction

### 🔹 Model 4: Two-Stage ViT
- Stage 1: Binary ViT model (No DR vs DR)
- Stage 2: 4-class ViT model for DR severity
- Ben Graham method + CLAHE

### 🔹 Model 5: Two-Stage ViT + Three-Headed Stage 2
- Final & best-performing model
- Multi-task heads + Focal Loss in Stage 2
- Trimmed mean for regression output smoothing

---

## Evaluation Metrics

| Metric        | Description                                      |
|---------------|--------------------------------------------------|
| **Accuracy**  | Used only for Model 1                            |
| **Cohen’s Kappa** | Primary metric for Models 2–5 (handles imbalance) |
| **Soft-Voting** | Ensemble from K-Fold predictions               |
| **Trimmed Mean** | For regression heads in Model 2               |

---

## Results Summary

| Model | Kaggle Accuracy | Best Validation Kappa      |
|-------|------------------|----------------------------|
| 1     | 0.78020          | Accuracy ≈ 0.99            |
| 2     | 0.83412          | 0.8948 – 0.9268            |
| 3     | 0.84163          | 0.8882 – 0.9167            |
| 4     | 0.85870          | Stage 1: 0.9499 – 0.9773   |
| 5     | **0.86075**      | Stage 1: 0.9590 – 0.9818, Stage 2: 0.5841 – 0.7438 |

---

## Key Insights

-  **Focal Loss** effectively addresses class imbalance  
-  **ViT models** outperform CNNs for global feature recognition  
-  **Two-stage pipelines** allow DR detection and grading to be decoupled  
-  **Multi-task learning** enriches feature representations  


