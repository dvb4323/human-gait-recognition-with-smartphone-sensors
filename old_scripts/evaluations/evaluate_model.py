"""
File: evaluate_model.py
Vị trí: src/evaluations/evaluate_model.py

Mục đích:
 - Đánh giá mô hình CNN đã huấn luyện
 - Tạo biểu đồ trực quan: confusion matrix, loss/accuracy
 - Xuất báo cáo đánh giá ra file
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# ==============================
# 🔧 Cấu hình đường dẫn
# ==============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

# ==============================
# 📥 Nạp dữ liệu
# ==============================
processed_path = os.path.join(DATA_DIR, "processed", "har_data_windows.npz")
data = np.load(processed_path)
X_test, y_test = data["X_test"], data["y_test"]

print(f"✅ Loaded test data: {X_test.shape}, labels: {y_test.shape}")

# ==============================
# 🧠 Nạp mô hình CNN đã lưu
# ==============================
model_path = os.path.join(MODEL_DIR, "cnn_final.h5")
model = load_model(model_path)
print(f"✅ Loaded model from: {model_path}")

# ==============================
# 📈 Dự đoán và tính toán metrics
# ==============================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n=== 🧠 CNN Evaluation ===")
report = classification_report(y_test, y_pred, digits=2, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

# Lưu báo cáo ra file
report_csv_path = os.path.join(REPORT_DIR, "cnn_evaluation_report.csv")
report_df.to_csv(report_csv_path, index=True)
print(f"\n📄 Saved evaluation report to: {report_csv_path}")

# ==============================
# 🔍 Confusion Matrix
# ==============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - CNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

cm_path = os.path.join(REPORT_DIR, "cnn_confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"📊 Saved confusion matrix plot to: {cm_path}")

# ==============================
# 📉 Training history
# ==============================
history_path = os.path.join(MODEL_DIR, "training_history.csv")
if os.path.exists(history_path):
    history = pd.read_csv(history_path)
    if "epoch" not in history.columns:
        history.insert(0, "epoch", np.arange(1, len(history) + 1))
    plt.figure(figsize=(10, 4))

    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Biểu đồ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["accuracy"], label="Train Acc")
    plt.plot(history["epoch"], history["val_accuracy"], label="Val Acc")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    history_plot_path = os.path.join(REPORT_DIR, "cnn_training_curves.png")
    plt.savefig(history_plot_path)
    plt.close()
    print(f"📈 Saved training history plots to: {history_plot_path}")
else:
    print("⚠️ No training history file found at:", history_path)

print("\n✅ Evaluation completed successfully!")
