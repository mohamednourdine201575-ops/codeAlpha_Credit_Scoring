import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ==========================================
# 1. LOAD DATA
# ==========================================

data = pd.read_csv(
    "german.data",
    sep=" ",
    header=None
)


# ==========================================
# 2. COLUMN NAMES
# ==========================================

columns = [
    "checking_account",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
    "risk"
]

data.columns = columns


# ==========================================
# 3. BASIC DATA CHECKS
# ==========================================

print("First 5 rows:")
print(data.head())

print("\nColumn names:")
print(data.columns.tolist())

print("\nDataset shape:")
print(data.shape)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate rows:")
print(data.duplicated().sum())

print("\nCredit risk distribution:")
print(data["risk"].value_counts())


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

X = data.drop("risk", axis=1)
y = data["risk"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ==========================================
# 6. DEFINE FEATURES FOR PREPROCESSING
# ==========================================

numerical_features = [
    "duration",
    "credit_amount",
    "installment_rate",
    "residence",
    "age",
    "existing_credits",
    "dependents"
]

categorical_features = [
    "checking_account",
    "credit_history",
    "purpose",
    "savings",
    "employment",
    "personal_status",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]


# ==========================================
# 7. PREPROCESSING
# ==========================================

numerical_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing pipeline created successfully!")


# ==========================================
# 8. LOGISTIC REGRESSION MODEL
# ==========================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Train the model
logistic_model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")
# ==========================================
# 9. MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Make predictions
y_pred = logistic_model.predict(X_test)

# Probability of class 2 (Bad)
y_proba = logistic_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=2)
recall = recall_score(y_test, y_pred, pos_label=2)
f1 = f1_score(y_test, y_pred, pos_label=2)

# ROC-AUC using probability for class 2
roc_auc = roc_auc_score(
    (y_test == 2).astype(int),
    y_proba
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[1, 2])

print("\n===== Logistic Regression Results =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)
# ==========================================
# 10. RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)

# Train
random_forest_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

# Predictions
rf_pred = random_forest_model.predict(X_test)

# Probability of Bad class (2)
rf_proba = random_forest_model.predict_proba(X_test)[:, 1]

# Metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, pos_label=2)
rf_recall = recall_score(y_test, rf_pred, pos_label=2)
rf_f1 = f1_score(y_test, rf_pred, pos_label=2)

rf_roc_auc = roc_auc_score(
    (y_test == 2).astype(int),
    rf_proba
)

rf_cm = confusion_matrix(y_test, rf_pred, labels=[1, 2])

print("\n===== Random Forest Results =====")
print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1-Score:  {rf_f1:.4f}")
print(f"ROC-AUC:   {rf_roc_auc:.4f}")

print("\nConfusion Matrix:")
print(rf_cm)
# ==========================================
# 11. MODEL COMPARISON
# ==========================================

print("\n===== MODEL COMPARISON =====")

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy,
        rf_accuracy
    ],
    "Precision": [
        precision,
        rf_precision
    ],
    "Recall": [
        recall,
        rf_recall
    ],
    "F1-Score": [
        f1,
        rf_f1
    ],
    "ROC-AUC": [
        roc_auc,
        rf_roc_auc
    ]
})

print(comparison.to_string(index=False))
# ==========================================
# 12. VISUALIZATION
# ==========================================

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# Confusion Matrix - Random Forest
ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_pred,
    labels=[1, 2],
    display_labels=["Good", "Bad"]
)

plt.title("Random Forest - Confusion Matrix")
plt.savefig("random_forest_confusion_matrix.png")
plt.show()


# ROC Curve - Both Models
RocCurveDisplay.from_predictions(
    (y_test == 2).astype(int),
    y_proba,
    name="Logistic Regression"
)

RocCurveDisplay.from_predictions(
    (y_test == 2).astype(int),
    rf_proba,
    name="Random Forest"
)

plt.title("ROC Curve - Credit Risk Models")
plt.savefig("roc_curve_comparison.png")
plt.show()
# ==========================================
# 13. SAVE BEST MODEL
# ==========================================

import joblib

joblib.dump(
    logistic_model,
    "credit_scoring_model.pkl"
)

print("\nBest model saved as credit_scoring_model.pkl")