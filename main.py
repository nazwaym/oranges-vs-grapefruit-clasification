"""
============================================================================
  Oranges vs Grapefruit Classification
  =====================================
  Membandingkan tiga model klasifikasi:
    1. Decision Tree
    2. Naive Bayes (Gaussian)
    3. Support Vector Machine (SVM)

  Dataset : https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit
  Features: diameter, weight, red, green, blue
  Target  : name (orange / grapefruit)
============================================================================

Usage:
    python main.py
"""

from src.data_loader import load_data
from src.eda import run_eda
from src.preprocessing import preprocess
from src.models import train_all
from src.evaluation import evaluate_and_visualize, print_summary


def main():
    # Tahap 1 — Load & Explore
    df = load_data()

    # Tahap 2 — EDA
    run_eda(df)

    # Tahap 3 — Preprocessing
    data = preprocess(df)

    # Tahap 4 — Training
    results = train_all(data)

    # Tahap 5 — Visualisasi Evaluasi
    evaluate_and_visualize(results, data)

    # Tahap 6 — Kesimpulan
    print_summary(results, data)


if __name__ == "__main__":
    main()
