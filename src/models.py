"""
Tahap 4: Model Training.

Tiga model yang dibandingkan:
  1. Decision Tree Classifier
  2. Gaussian Naive Bayes
  3. Support Vector Machine (SVM)
"""

import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from .config import RANDOM_STATE, CV_FOLDS

warnings.filterwarnings("ignore")


def _build_models():
    """Definisikan ketiga model."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }


def train_all(data):
    """
    Latih ketiga model dan hitung metrik evaluasi.

    Args:
        data: dict hasil dari preprocessing.

    Returns:
        dict results — per model berisi model, prediksi, probabilitas, metrik.
    """
    print("\n" + "=" * 65)
    print("  TAHAP 4: MODEL TRAINING & EVALUATION")
    print("=" * 65)

    models = _build_models()
    results = {}

    X_train = data["X_train"]
    X_test = data["X_test"]
    X_train_s = data["X_train_scaled"]
    X_test_s = data["X_test_scaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    le = data["label_encoder"]

    for name, model in models.items():
        print(f"\n  >> Training: {name}")
        print("  " + "-" * 40)

        # SVM memerlukan data yang sudah di-scale
        is_svm = name == "SVM"
        Xtr = X_train_s if is_svm else X_train
        Xte = X_test_s if is_svm else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]

        # Metrik
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Cross-Validation
        cv_scores = cross_val_score(
            model, Xtr, y_train, cv=CV_FOLDS, scoring="accuracy"
        )

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

        print(f"    Accuracy   : {acc:.4f}")
        print(f"    Precision  : {prec:.4f}")
        print(f"    Recall     : {rec:.4f}")
        print(f"    F1-Score   : {f1:.4f}")
        print(f"    CV (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print()
        print(classification_report(
            y_test, y_pred, target_names=le.classes_, zero_division=0
        ))

    return results
