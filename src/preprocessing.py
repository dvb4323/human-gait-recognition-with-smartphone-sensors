"""
File: src/preprocessing.py
Mô tả: Làm sạch dữ liệu, chuẩn hóa, và phân đoạn thành cửa sổ.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils import sliding_window
import os


def clean_data(X):
    """
    Xử lý ngoại lệ hoặc giá trị thiếu (ở đây UCI HAR gần như sạch).
    Nếu có thể áp dụng thêm median filter, clipping, hoặc interpolation.
    """
    X = np.asarray(X)
    X = np.nan_to_num(X)  # thay NaN bằng 0
    return X


def scale_data(X_train, X_test):
    """
    Chuẩn hóa dữ liệu (StandardScaler: z-score normalization)
    """
    X_train_arr = np.asarray(X_train)
    X_test_arr = np.asarray(X_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_arr)
    X_test_scaled = scaler.transform(X_test_arr)
    return X_train_scaled, X_test_scaled, scaler


def segment_data(X_train, y_train, X_test, y_test, window_size=128, overlap=0.5):
    """
    Phân đoạn dữ liệu thành cửa sổ trượt (cho DL hoặc feature extraction).
    """

    def to_numpy(x):
        if hasattr(x, "values"):
            return x.values
        return np.asarray(x)

    Xtr = to_numpy(X_train)
    Xte = to_numpy(X_test)
    ytr = to_numpy(y_train).flatten()
    yte = to_numpy(y_test).flatten()

    n_samples_tr, n_features_tr = Xtr.shape
    if n_features_tr % window_size == 0:
        num_channels = n_features_tr // window_size
        X_train_w = Xtr.reshape(n_samples_tr, num_channels, window_size).transpose(0, 2, 1)
        y_train_w = ytr
    else:
        X_train_w, y_train_w = sliding_window(Xtr, ytr, window_size, overlap)

    n_samples_te, n_features_te = Xte.shape
    if n_features_te % window_size == 0:
        num_channels_te = n_features_te // window_size
        X_test_w = Xte.reshape(n_samples_te, num_channels_te, window_size).transpose(0, 2, 1)
        y_test_w = yte
    else:
        X_test_w, y_test_w = sliding_window(Xte, yte, window_size, overlap)

    print(f"✅ Dữ liệu sau khi windowing:")
    print(f"Train windows: {X_train_w.shape}, Test windows: {X_test_w.shape}")
    return X_train_w, y_train_w, X_test_w, y_test_w