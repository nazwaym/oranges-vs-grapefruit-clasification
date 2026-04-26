"""
Tahap 1: Data Loading & Exploration.
"""

import pandas as pd
from .config import DATA_PATH


def load_data():
    """
    Load dataset dari CSV dan tampilkan ringkasan eksplorasi awal.

    Returns:
        pd.DataFrame: Dataset yang sudah dimuat.
    """
    print("=" * 65)
    print("  TAHAP 1: DATA LOADING & EXPLORATION")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)

    # Dimensi data
    print(f"\n  Jumlah Data  : {len(df)} baris")
    print(f"  Jumlah Fitur : {df.shape[1]} kolom")
    print(f"  Kolom        : {list(df.columns)}")

    # Sampel data
    print("\n  -- 5 Data Pertama --")
    print(df.head().to_string(index=False))

    # Statistik deskriptif
    print("\n  -- Statistik Deskriptif --")
    print(df.describe().round(2).to_string())

    # Missing values
    print("\n  -- Missing Values --")
    missing = df.isnull().sum()
    print(missing.to_string())
    if missing.sum() == 0:
        print("  [OK] Tidak ada missing values.")

    # Distribusi kelas
    print("\n  -- Distribusi Kelas --")
    class_dist = df["name"].value_counts()
    print(class_dist.to_string())
    print(f"  Rasio: {class_dist.values[0]}:{class_dist.values[1]}")

    return df
