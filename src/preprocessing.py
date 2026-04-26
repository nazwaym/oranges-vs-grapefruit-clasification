"""
Tahap 3: Data Preprocessing.

Langkah:
  - Label Encoding (target)
  - Train-Test Split (80:20, stratified)
  - Feature Scaling (StandardScaler)
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .config import FEATURES, TEST_SIZE, RANDOM_STATE


def preprocess(df):
    """
    Lakukan preprocessing lengkap.

    Returns:
        dict berisi X_train, X_test, X_train_scaled, X_test_scaled,
        y_train, y_test, label_encoder, scaler.
    """
    print("\n" + "=" * 65)
    print("  TAHAP 3: DATA PREPROCESSING")
    print("=" * 65)

    # 1. Label Encoding
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["name"])   # grapefruit=0, orange=1
    print(f"\n  Label Encoding : {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X = df[FEATURES].values
    y = df["label"].values

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  Train set      : {X_train.shape[0]} samples")
    print(f"  Test  set      : {X_test.shape[0]} samples")

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  Scaling        : StandardScaler applied")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": le,
        "scaler": scaler,
    }
