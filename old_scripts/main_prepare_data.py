"""
File: main_prepare_data.py
Mô tả: Script chính chạy Bước 1 – Chuẩn bị dữ liệu.
"""
import os
import joblib
import numpy as np
from old_scripts.data_loader import download_uci_har_dataset, load_har_data
from old_scripts.preprocessing import clean_data, scale_data, segment_data

# ========================== # 0. Tạo folder cần thiết # ==========================
os.makedirs("../data/raw", exist_ok=True)
os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../artifacts", exist_ok=True)

# ==========================
# 1. Tải và đọc dữ liệu
# ==========================
download_uci_har_dataset()
X_train, y_train, X_test, y_test = load_har_data()

# ==========================
# 2. Làm sạch & chuẩn hóa
# ==========================
X_train = clean_data(X_train)
X_test = clean_data(X_test)

X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

# Lưu scaler để dùng khi inference
joblib.dump(scaler, "../artifacts/scaler.pkl")
print("✅ Lưu scaler tại artifacts/scaler.pkl")

# ==========================
# 3. Phân đoạn (Windowing)
# ==========================
X_train_w, y_train_w, X_test_w, y_test_w = segment_data(
    X_train_scaled, y_train, X_test_scaled, y_test, window_size=128, overlap=0.5
)

# ==========================
# 4. Lưu dữ liệu processed
# ==========================
np.savez_compressed(
    "../data/processed/har_data_windows.npz",
    X_train=X_train_w, y_train=y_train_w,
    X_test=X_test_w, y_test=y_test_w
)

print("🎉 Hoàn tất bước 1 – dữ liệu đã được xử lý và lưu tại data/processed/har_data_windows.npz")
