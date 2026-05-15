"""
automate_Muhammad-Zaenal-Arifin.py
Skrip otomatisasi preprocessing dataset Wine Quality Red.
Menghasilkan data yang siap dilatih.

Author : Muhammad Zaenal Arifin
Dataset: Wine Quality Red (UCI ML Repository)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# KONSTANTA
# ──────────────────────────────────────────────
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
RAW_FILE    = "winequality_raw.csv"
OUTPUT_DIR  = "winequality_preprocessing"
QUALITY_THR = 7          # quality >= threshold → label 1 (good)
TEST_SIZE   = 0.2
RANDOM_STATE = 42


# ──────────────────────────────────────────────
# FUNGSI-FUNGSI PREPROCESSING
# ──────────────────────────────────────────────

def load_data(url: str = DATA_URL, sep: str = ";") -> pd.DataFrame:
    """Memuat dataset dari URL atau file lokal."""
    print("[1/7] Memuat dataset ...")
    try:
        df = pd.read_csv(url, sep=sep)
    except Exception:
        # fallback ke file lokal
        df = pd.read_csv(RAW_FILE, sep=sep)
    print(f"      Shape raw: {df.shape}")
    # Simpan raw
    df.to_csv(RAW_FILE, index=False)
    return df


def binarize_target(df: pd.DataFrame,
                    quality_col: str = "quality",
                    threshold: int = QUALITY_THR) -> pd.DataFrame:
    """Konversi skor quality ke label biner (0 / 1)."""
    print("[2/7] Binarisasi target ...")
    df = df.copy()
    df["quality_label"] = (df[quality_col] >= threshold).astype(int)
    df = df.drop(columns=[quality_col])
    dist = df["quality_label"].value_counts().to_dict()
    print(f"      Distribusi: bad={dist.get(0,0)}, good={dist.get(1,0)}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat."""
    print("[3/7] Menghapus duplikat ...")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"      Dihapus: {before - len(df)} baris | Sisa: {len(df)}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Mengisi missing values kolom numerik dengan median."""
    print("[4/7] Menangani missing values ...")
    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        print("      Tidak ada missing values.")
        return df
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"      {col}: diisi median = {median_val:.4f}")
    return df


def handle_outliers(df: pd.DataFrame,
                    exclude_col: str = "quality_label") -> pd.DataFrame:
    """Menangani outlier dengan IQR capping (winsorizing)."""
    print("[5/7] Menangani outlier (IQR capping) ...")
    feature_cols = [c for c in df.columns if c != exclude_col]
    for col in feature_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_out > 0:
            print(f"      {col}: {n_out} outlier di-cap ke [{lower:.4f}, {upper:.4f}]")
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def scale_features(df: pd.DataFrame,
                   target_col: str = "quality_label"):
    """Standarisasi fitur numerik; mengembalikan (df_scaled, scaler)."""
    print("[6/7] Standarisasi fitur (StandardScaler) ...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled = X_scaled.copy()
    df_scaled[target_col] = y.values
    return df_scaled, scaler


def split_and_save(df: pd.DataFrame,
                   target_col: str = "quality_label",
                   output_dir: str = OUTPUT_DIR,
                   test_size: float = TEST_SIZE,
                   random_state: int = RANDOM_STATE) -> None:
    """Train-test split dan simpan hasil preprocessing."""
    print("[7/7] Split data dan simpan ...")
    os.makedirs(output_dir, exist_ok=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Gabungkan kembali
    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    test_df = X_test.copy()
    test_df[target_col] = y_test.values

    # Simpan file
    df.to_csv(os.path.join(output_dir, "winequality_preprocessed.csv"), index=False)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"      Train: {train_df.shape} → {output_dir}/train.csv")
    print(f"      Test : {test_df.shape} → {output_dir}/test.csv")
    print(f"      Full : {df.shape} → {output_dir}/winequality_preprocessed.csv")


# ──────────────────────────────────────────────
# PIPELINE UTAMA
# ──────────────────────────────────────────────

def run_preprocessing() -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing.
    Mengembalikan DataFrame yang sudah siap dilatih.
    """
    print("=" * 55)
    print("  AUTOMATE PREPROCESSING - Wine Quality Red")
    print("  Author: Muhammad Zaenal Arifin")
    print("=" * 55)

    df = load_data()
    df = binarize_target(df)
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = handle_outliers(df)
    df, scaler = scale_features(df)
    split_and_save(df)

    print("\n✓ Preprocessing selesai! Data siap untuk pelatihan.")
    print(f"  Shape akhir: {df.shape}")
    return df


if __name__ == "__main__":
    result = run_preprocessing()
