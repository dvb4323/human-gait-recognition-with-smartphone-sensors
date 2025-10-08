# ============================================
# 📁 File: src/feature_engineering.py
# 📦 Mục đích: Trích xuất đặc trưng miền thời gian và tần số
# ============================================

import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.fft import fft


def extract_time_domain_features(window, feature_names=None):
    """
    Trích xuất đặc trưng miền thời gian từ 1 cửa sổ dữ liệu (numpy array)
    window.shape = (window_size, n_features)
    """
    features = {}
    if feature_names is None:
        feature_names = [f"sensor_{i}" for i in range(window.shape[1])]

    # --- Các đặc trưng thống kê ---
    for i, name in enumerate(feature_names):
        col = window[:, i]
        features[f"{name}_mean"] = np.mean(col)
        features[f"{name}_std"] = np.std(col)
        features[f"{name}_min"] = np.min(col)
        features[f"{name}_max"] = np.max(col)
        features[f"{name}_median"] = np.median(col)
        features[f"{name}_rms"] = np.sqrt(np.mean(col ** 2))

        # Entropy
        p = np.abs(col) / np.sum(np.abs(col)) if np.sum(np.abs(col)) != 0 else np.ones_like(col) / len(col)
        features[f"{name}_entropy"] = entropy(p)

    return features


def extract_frequency_domain_features(window, feature_names=None):
    """
    Trích xuất đặc trưng miền tần số (sử dụng FFT)
    """
    features = {}
    if feature_names is None:
        feature_names = [f"sensor_{i}" for i in range(window.shape[1])]

    fft_vals = np.abs(fft(window, axis=0))
    for i, name in enumerate(feature_names):
        col_fft = fft_vals[:, i]
        features[f"{name}_fft_mean"] = np.mean(col_fft)
        features[f"{name}_fft_std"] = np.std(col_fft)
        features[f"{name}_fft_energy"] = np.sum(col_fft ** 2) / len(col_fft)

    return features


def extract_features(X_windows, y_windows, feature_names=None, include_frequency=False):
    """
    X_windows: numpy array (num_samples, window_size, n_features)
    y_windows: numpy array (num_samples,)
    """
    all_features = []
    for window in X_windows:
        feats = extract_time_domain_features(window, feature_names)
        if include_frequency:
            feats.update(extract_frequency_domain_features(window, feature_names))
        all_features.append(feats)

    feature_df = pd.DataFrame(all_features)
    feature_df["label"] = y_windows
    return feature_df
