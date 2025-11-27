# ============================================
# 📁 File: main_feature_engineering.py
# 🎯 Mục đích: Chạy bước 2 - Trích xuất đặc trưng
# ============================================

import os
import numpy as np
from old_scripts.feature_engineering import extract_features
from old_scripts.visualization import generate_all_visualizations  # 🆕 Thêm module mới

# Đường dẫn dữ liệu từ bước 1
processed_path = "../data/processed/har_data_windows.npz"
print(f"📂 Đang tải dữ liệu từ {processed_path} ...")

# Đọc dữ liệu cửa sổ đã lưu
data = np.load(processed_path)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
print(f"✅ X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"✅ X_test: {X_test.shape}, y_test: {y_test.shape}")

# Trích xuất đặc trưng
print("⚙️ Đang trích xuất đặc trưng...")
train_features = extract_features(X_train, y_train, include_frequency=True)
test_features = extract_features(X_test, y_test, include_frequency=True)

# Lưu kết quả
features_dir = "../data/features"
os.makedirs(features_dir, exist_ok=True)
train_csv = os.path.join(features_dir, "train_features.csv")
test_csv = os.path.join(features_dir, "test_features.csv")
train_features.to_csv(train_csv, index=False)
test_features.to_csv(test_csv, index=False)

print(f"✅ Đã lưu đặc trưng tại {features_dir}/")
print(f"🧾 train_features: {train_features.shape}, test_features: {test_features.shape}")

# 🔹 Sinh biểu đồ trực quan
print("📊 Đang tạo biểu đồ trực quan...")
generate_all_visualizations(train_csv, save_dir="../reports/figures/train")
generate_all_visualizations(test_csv, save_dir="../reports/figures/test")
