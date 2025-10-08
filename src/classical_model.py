# ============================================
# 📁 File: src/classical_model.py
# 🎯 Mục đích: Huấn luyện mô hình học máy cổ điển (RF, SVM)
# ============================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


def train_classical_models(train_csv, test_csv, model_dir="data/models"):
    """
    Huấn luyện và đánh giá các mô hình học máy cổ điển.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Đọc dữ liệu
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    # Chuẩn hóa dữ liệu đầu vào
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 🔹 Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    print("\n=== 🟩 Random Forest Evaluation ===")
    print(classification_report(y_test, y_pred_rf))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    joblib.dump(rf, os.path.join(model_dir, "random_forest.pkl"))

    # 🔹 Support Vector Machine (tùy chọn)
    svm = SVC(kernel="rbf", C=10, gamma="scale")
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)

    print("\n=== 🟦 Support Vector Machine Evaluation ===")
    print(classification_report(y_test, y_pred_svm))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
    joblib.dump(svm, os.path.join(model_dir, "svm_rbf.pkl"))

    # 🔹 Lưu scaler
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print(f"✅ Mô hình đã lưu tại {model_dir}/")

