"""
Tahap 5 & 6: Evaluasi Model & Visualisasi Perbandingan.

Visualisasi yang dihasilkan:
  06 - Confusion Matrices (3 model)
  07 - ROC Curves
  08 - Metrics Comparison Bar Chart
  09 - Feature Importance (Decision Tree)
  10 - Final Summary Heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc

from .config import MODEL_COLORS, FEATURES, save_fig


def evaluate_and_visualize(results, data):
    """
    Buat seluruh visualisasi evaluasi model.

    Args:
        results: dict hasil dari train_all().
        data:    dict hasil dari preprocess().
    """
    print("\n" + "=" * 65)
    print("  TAHAP 5: VISUALISASI PERBANDINGAN MODEL")
    print("=" * 65)

    y_test = data["y_test"]
    le = data["label_encoder"]

    _plot_confusion_matrices(results, y_test, le)
    _plot_roc_curves(results, y_test)
    _plot_metrics_comparison(results)
    _plot_feature_importance(results)
    _plot_final_summary(results, y_test)


def print_summary(results, data):
    """Cetak tabel ringkasan performa ke terminal."""
    print("\n" + "=" * 65)
    print("  TAHAP 6: KESIMPULAN")
    print("=" * 65)

    y_test = data["y_test"]

    rows = []
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        rows.append({
            "Model": name,
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1"],
            "AUC-ROC": auc(fpr, tpr),
            "CV Mean": r["cv_mean"],
        })

    df_summary = pd.DataFrame(rows)
    print("\n  -- Tabel Perbandingan Performa --")
    print(df_summary.to_string(index=False, float_format="{:.4f}".format))

    best = max(results, key=lambda k: results[k]["f1"])
    print(f"\n  [BEST] Model Terbaik (F1-Score): {best}")
    print(f"         F1-Score : {results[best]['f1']:.4f}")
    print(f"         Accuracy : {results[best]['accuracy']:.4f}")

    print("\n" + "=" * 65)
    print("  [DONE] Semua output tersimpan di folder /output/")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════
# Private visualization functions
# ══════════════════════════════════════════════════════════════════════════

def _plot_confusion_matrices(results, y_test, le):
    """06 — Confusion matrix side-by-side 3 model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Confusion Matrix - Perbandingan 3 Model",
        fontweight="bold", fontsize=15,
    )

    for i, (name, r) in enumerate(results.items()):
        cm = confusion_matrix(y_test, r["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=1, linecolor="white", cbar_kws={"shrink": 0.8},
        )
        axes[i].set_title(
            f"{name}\n(Acc: {r['accuracy']:.4f})", fontweight="bold"
        )
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.tight_layout()
    save_fig("06_confusion_matrices")


def _plot_roc_curves(results, y_test):
    """07 — ROC curve overlay 3 model."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title(
        "ROC Curve - Perbandingan 3 Model", fontweight="bold", fontsize=14
    )

    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr, color=MODEL_COLORS[name], linewidth=2.5,
            label=f"{name} (AUC = {roc_auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig("07_roc_curves")


def _plot_metrics_comparison(results):
    """08 — Grouped bar chart perbandingan metrik."""
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "CV Mean"]
    model_list = list(results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        "Perbandingan Metrik Evaluasi - 3 Model",
        fontweight="bold", fontsize=15,
    )

    x = np.arange(len(metric_names))
    width = 0.22

    for i, name in enumerate(model_list):
        r = results[name]
        vals = [r["accuracy"], r["precision"], r["recall"],
                r["f1"], r["cv_mean"]]
        bars = ax.bar(
            x + i * width, vals, width, label=name,
            color=MODEL_COLORS[name], edgecolor="white",
            linewidth=1, alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig("08_metrics_comparison")


def _plot_feature_importance(results):
    """09 — Feature importance dari Decision Tree."""
    dt_model = results["Decision Tree"]["model"]
    importances = dt_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Feature Importance - Decision Tree", fontweight="bold", fontsize=15
    )

    bars = ax.barh(
        range(len(FEATURES)), importances[indices],
        color=[
            "#2ECC71" if importances[indices[j]] == max(importances)
            else "#95E1B7"
            for j in range(len(FEATURES))
        ],
        edgecolor="white", linewidth=1, height=0.5,
    )

    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(
        [FEATURES[i] for i in indices], fontweight="bold"
    )
    ax.set_xlabel("Importance")
    ax.invert_yaxis()

    for bar, imp in zip(bars, importances[indices]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{imp:.4f}", va="center", fontweight="bold", fontsize=10,
        )

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_fig("09_feature_importance")


def _plot_final_summary(results, y_test):
    """10 — Heatmap ringkasan performa akhir."""
    model_names = list(results.keys())
    metric_labels = [
        "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "CV Mean"
    ]

    cell_data = []
    for name in model_names:
        r = results[name]
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        roc_auc = auc(fpr, tpr)
        cell_data.append([
            r["accuracy"], r["precision"], r["recall"],
            r["f1"], roc_auc, r["cv_mean"],
        ])

    cell_array = np.array(cell_data)

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle(
        "Ringkasan Performa Model - Final Comparison",
        fontweight="bold", fontsize=15,
    )

    sns.heatmap(
        cell_array, annot=True, fmt=".4f", cmap="RdYlGn",
        xticklabels=metric_labels, yticklabels=model_names,
        linewidths=2, linecolor="white", ax=ax,
        vmin=0.5, vmax=1.0,
        cbar_kws={"label": "Score", "shrink": 0.8},
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")

    plt.tight_layout()
    save_fig("10_final_summary")
