# ============================================
# 📁 File: main_train_classical.py
# 🎯 Mục đích: Chạy huấn luyện mô hình cổ điển
# ============================================

import os
from old_scripts.classical_model import train_classical_models

# Đường dẫn dữ liệu đặc trưng
features_dir = "../data/features"
train_csv = os.path.join(features_dir, "train_features.csv")
test_csv = os.path.join(features_dir, "test_features.csv")

# Huấn luyện
train_classical_models(train_csv, test_csv)
