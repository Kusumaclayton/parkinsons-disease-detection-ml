# Parkinson’s Disease Detection using Machine Learning

## Overview

This project implements a machine learning-based system for early detection of Parkinson’s Disease using two complementary modalities: voice signal features and spiral drawing patterns. The approach combines gradient boosting (XGBoost) for tabular voice data and ensemble learning (Random Forest) for image-based spiral analysis to improve prediction reliability.

The pipeline covers data preprocessing, feature scaling, model training, and evaluation using standard classification metrics. The goal is to provide a non-invasive, cost-effective method for early screening.

---

## Key Highlights

* Multi-modal approach combining voice and image-based features
* End-to-end pipeline: preprocessing, training, evaluation
* Models: XGBoost (voice), Random Forest (spiral images)
* Evaluation metrics: accuracy, confusion matrix, classification report
* Designed for reproducibility using publicly available datasets

---

## Technologies

* Python
* NumPy, Pandas
* OpenCV
* Scikit-learn
* XGBoost

---

## Dataset

This project uses publicly available datasets:

Voice Dataset:
https://www.kaggle.com/datasets/nidaguler/parkinsons-data-set

Spiral Drawings Dataset:
https://www.kaggle.com/datasets/kmader/parkinsons-drawings

**Dataset Details (from source):**

* Voice dataset: ~195 samples (147 Parkinson’s, 48 healthy), 24 features including frequency, jitter, shimmer, and noise ratios
* Spiral dataset: labeled images of healthy vs Parkinson-affected drawings used for classification

Note: Datasets are not included in this repository due to size constraints. Please download them manually.

---

## Expected Dataset Structure

dataset/
├── parkinsons.csv
├── spiral/
│   ├── healthy/
│   ├── parkinson/

---

## Installation and Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/parkinsons-disease-detection-ml.git
cd parkinsons-disease-detection-ml
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Download and place datasets

* Download datasets from the links above
* Place them inside the `dataset/` directory

### 4. Run the project

```
python code/main.py
```

---

## Methodology

1. Load and preprocess voice dataset (feature scaling applied)
2. Train XGBoost classifier on voice features
3. Load spiral images and perform resizing and flattening
4. Train Random Forest classifier on image data
5. Evaluate both models using classification metrics
6. Combine insights for overall prediction

---

## Results

* Achieved reliable classification performance using ensemble models
* Demonstrated effectiveness of combining voice and visual biomarkers
* Evaluation performed using accuracy and confusion matrix for both models

(Note: Results may vary depending on dataset split and preprocessing.)

---

## Project Context

This project was developed as part of a Bachelor of Technology (Electronics and Communication Engineering) final year project. It focuses on applying machine learning techniques in the healthcare domain for early disease detection.

---

## Contributors

* Kusuma Clayton
* Kampati Naithika Sree
* Hemant Katta

---

## License

This project is licensed under the MIT License.

---

## Reproducibility Note

The implementation is based on the methodology described in the project report. The project can be reproduced using the provided code and publicly available datasets.
