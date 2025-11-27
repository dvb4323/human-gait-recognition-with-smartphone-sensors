"""
File: src/data_loader.py
Mô tả: Tải và đọc UCI HAR Dataset từ nguồn công khai.
"""
import os
import urllib.request
import zipfile
import pandas as pd


UCI_HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATA_PATH = os.path.join("data", "raw")
EXTRACTED_PATH = os.path.join(DATA_PATH, "UCI HAR Dataset")


def download_uci_har_dataset():
    """Tải dataset UCI HAR nếu chưa tồn tại."""
    os.makedirs(DATA_PATH, exist_ok=True)
    zip_path = os.path.join(DATA_PATH, "UCI_HAR_Dataset.zip")

    if not os.path.exists(EXTRACTED_PATH):
        print("📥 Đang tải UCI HAR Dataset...")
        urllib.request.urlretrieve(UCI_HAR_URL, zip_path)
        print("✅ Tải xong, giải nén...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_PATH)
        print("✅ Giải nén hoàn tất.")
    else:
        print("✅ Dataset đã có sẵn.")


def load_har_data():
    """Đọc dữ liệu train/test từ thư mục UCI HAR Dataset."""
    print("📂 Đang đọc dữ liệu...")

    # Đường dẫn
    train_path = os.path.join(EXTRACTED_PATH, "train")
    test_path = os.path.join(EXTRACTED_PATH, "test")

    # Đọc tín hiệu Accelerometer & Gyroscope (3 trục)
    def load_signals(folder, dataset_type):
        signals = []
        for signal in ["body_acc_x", "body_acc_y", "body_acc_z",
                       "body_gyro_x", "body_gyro_y", "body_gyro_z"]:
            filename = os.path.join(folder, "Inertial Signals", f"{signal}_{dataset_type}.txt")
            data = pd.read_csv(filename, sep=r'\s+', header=None)
            signals.append(data)
        return signals

    X_train_signals = load_signals(train_path, "train")
    X_test_signals = load_signals(test_path, "test")

    # Gộp trục lại → (num_samples, num_features)
    X_train = pd.concat(X_train_signals, axis=1)
    X_test = pd.concat(X_test_signals, axis=1)

    # Đọc nhãn
    y_train = pd.read_csv(os.path.join(train_path, "y_train.txt"), header=None, names=["label"])
    y_test = pd.read_csv(os.path.join(test_path, "y_test.txt"), header=None, names=["label"])

    print(f"✅ X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test
