"""
Konfigurasi global: style, warna, dan konstanta.
"""

import os
import sys
import io
import matplotlib.pyplot as plt

# ── Fix Windows console encoding ─────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

# ── Matplotlib Style ─────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica"],
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 16,
})

# ── Color Palette ────────────────────────────────────────────────────────
ORANGE_COLOR = "#FF8C00"
GRAPE_COLOR = "#9B59B6"
CLASS_PALETTE = {"orange": ORANGE_COLOR, "grapefruit": GRAPE_COLOR}

MODEL_COLORS = {
    "Decision Tree": "#2ECC71",
    "Naive Bayes": "#3498DB",
    "SVM": "#E74C3C",
}

# ── Paths ────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "citrus.csv")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Feature list ─────────────────────────────────────────────────────────
FEATURES = ["diameter", "weight", "red", "green", "blue"]

# ── Constants ────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5


def save_fig(name):
    """Save current figure to output directory."""
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  [OK] Saved: {path}")
    return path
