# Credit Scoring Model

## Overview
This project uses machine learning to predict credit risk based on an applicant's financial and personal information. It was developed as part of the CodeAlpha Machine Learning Internship.

## Dataset
- **Source:** UCI Statlog German Credit Data
- **Records:** 1,000
- **Features:** 20
- **Target:** Good credit risk or bad credit risk

## Machine Learning Models
- Logistic Regression
- Random Forest Classifier

## Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 78.00% | 75.50% |
| Precision | 66.67% | 67.74% |
| Recall | 53.33% | 35.00% |
| F1-score | 59.26% | 46.15% |
| ROC-AUC | 80.40% | 79.36% |

Logistic Regression achieved the higher accuracy, recall, F1-score, and ROC-AUC on the current test split.

## Technologies
- Python
- pandas and NumPy
- scikit-learn
- Matplotlib

## Project Files
- `load_data.py` — data processing, model training, and evaluation
- `predict.py` — predicts credit risk for a sample applicant
- `credit_scoring_model.pkl` — saved trained model
- `german.data` — dataset
- `german.data-numeric` — numeric version of the dataset
- `german.doc` — dataset documentation
- `random_forest_confusion_matrix.png` — Random Forest evaluation
- `roc_curve_comparison.png` — ROC curve comparison

## How to Run

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Train and evaluate the models:

```bash
python3 load_data.py
```

Run a sample prediction:

```bash
python3 predict.py
```

## Disclaimer
This project is for educational purposes. Its predictions should not be used as the sole basis for real credit decisions.
