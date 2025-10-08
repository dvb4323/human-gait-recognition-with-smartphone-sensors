# ============================================
# 📁 File: main_train_deep.py
# 🎯 Mục đích: Chạy huấn luyện mô hình học sâu
# ============================================

import os
import numpy as np
from src.deep_learning_model import train_cnn_model

# Đường dẫn dữ liệu từ bước 1
processed_path = "data/processed/har_data_windows.npz"
print(f"📂 Đang tải dữ liệu từ {processed_path} ...")

# Đọc dữ liệu cửa sổ đã lưu
data = np.load(processed_path)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
print(f"✅ X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"✅ X_test: {X_test.shape}, y_test: {y_test.shape}")

num_classes = len(np.unique(y_train))
train_cnn_model(X_train, y_train, X_test, y_test, num_classes)
