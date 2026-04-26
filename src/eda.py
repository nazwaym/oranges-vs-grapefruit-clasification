"""
Tahap 2: Exploratory Data Analysis (EDA).

Visualisasi yang dihasilkan:
  01 - Class Distribution (Bar + Pie)
  02 - Feature Distributions (Histogram + KDE)
  03 - Box Plots
  04 - Correlation Heatmap
  05 - Scatter Plots
"""

import matplotlib.pyplot as plt
import seaborn as sns

from .config import CLASS_PALETTE, FEATURES, save_fig


def run_eda(df):
    """Jalankan seluruh visualisasi EDA."""
    print("\n" + "=" * 65)
    print("  TAHAP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 65)

    _plot_class_distribution(df)
    _plot_feature_distributions(df)
    _plot_boxplots(df)
    _plot_correlation_heatmap(df)
    _plot_scatter_plots(df)

    print("  [OK] EDA selesai.")


# ─────────────────────────────────────────────────────────────────────────
def _plot_class_distribution(df):
    """01 — Bar chart + Pie chart distribusi kelas."""
    class_dist = df["name"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Distribusi Kelas: Oranges vs Grapefruit",
        fontweight="bold", fontsize=15,
    )

    # Bar
    bars = axes[0].bar(
        class_dist.index, class_dist.values,
        color=[CLASS_PALETTE[c] for c in class_dist.index],
        edgecolor="white", linewidth=1.5, width=0.5,
    )
    for bar, val in zip(bars, class_dist.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=12,
        )
    axes[0].set_title("Jumlah per Kelas", fontweight="bold")
    axes[0].set_ylabel("Jumlah")
    axes[0].set_ylim(0, class_dist.values.max() * 1.15)

    # Pie
    wedges, texts, autotexts = axes[1].pie(
        class_dist.values, labels=class_dist.index, autopct="%1.1f%%",
        colors=[CLASS_PALETTE[c] for c in class_dist.index],
        startangle=90, explode=(0.03, 0.03), shadow=True,
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    axes[1].set_title("Proporsi Kelas", fontweight="bold")

    plt.tight_layout()
    save_fig("01_class_distribution")


# ─────────────────────────────────────────────────────────────────────────
def _plot_feature_distributions(df):
    """02 — Histogram + KDE setiap fitur per kelas."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Distribusi Fitur berdasarkan Kelas (Histogram + KDE)",
        fontweight="bold", fontsize=15,
    )
    axes_flat = axes.flatten()

    for i, feat in enumerate(FEATURES):
        ax = axes_flat[i]
        for label, color in CLASS_PALETTE.items():
            subset = df[df["name"] == label][feat]
            ax.hist(subset, bins=40, alpha=0.45, color=color,
                    label=label, density=True)
            subset.plot.kde(ax=ax, color=color, linewidth=2)
        ax.set_title(feat.capitalize(), fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.legend()

    axes_flat[-1].axis("off")  # hide empty subplot
    plt.tight_layout()
    save_fig("02_feature_distributions")


# ─────────────────────────────────────────────────────────────────────────
def _plot_boxplots(df):
    """03 — Box plot setiap fitur per kelas."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle("Box Plot Fitur per Kelas", fontweight="bold", fontsize=15)

    for i, feat in enumerate(FEATURES):
        sns.boxplot(
            data=df, x="name", y=feat, ax=axes[i],
            palette=CLASS_PALETTE, linewidth=1.2, fliersize=2,
        )
        axes[i].set_title(feat.capitalize(), fontweight="bold")
        axes[i].set_xlabel("")

    plt.tight_layout()
    save_fig("03_boxplots")


# ─────────────────────────────────────────────────────────────────────────
def _plot_correlation_heatmap(df):
    """04 — Heatmap korelasi seluruh data."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Correlation Heatmap", fontweight="bold", fontsize=15)

    corr = df[FEATURES].corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
        ax=ax, vmin=-1, vmax=1, linewidths=0.5, square=True,
    )

    plt.tight_layout()
    save_fig("04_correlation_heatmap")


# ─────────────────────────────────────────────────────────────────────────
def _plot_scatter_plots(df):
    """05 — Scatter plot pasangan fitur utama."""
    pairs = [("diameter", "weight"), ("red", "green"), ("red", "blue")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Scatter Plot Fitur Utama", fontweight="bold", fontsize=15)

    for i, (fx, fy) in enumerate(pairs):
        for label, color in CLASS_PALETTE.items():
            subset = df[df["name"] == label]
            axes[i].scatter(
                subset[fx], subset[fy], c=color, label=label,
                alpha=0.3, s=8, edgecolors="none",
            )
        axes[i].set_xlabel(fx.capitalize())
        axes[i].set_ylabel(fy.capitalize())
        axes[i].set_title(
            f"{fx.capitalize()} vs {fy.capitalize()}", fontweight="bold"
        )
        axes[i].legend(markerscale=3)

    plt.tight_layout()
    save_fig("05_scatter_plots")
