import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Đường dẫn đến thư mục dataset
base_path = r"C:\Users\Binh\Desktop\Code\human-gait-recognition-with-smartphone-sensors\data\raw\A_DeviceMotion_data\A_DeviceMotion_data"

# Lấy danh sách các thư mục (mỗi thư mục là một hoạt động)
activities = os.listdir(base_path)

activity_data = []
activity_labels = []

for act in activities:
    act_path = os.path.join(base_path, act)
    if not os.path.isdir(act_path):
        continue
    for file in os.listdir(act_path):
        if file.endswith(".csv"):
            file_path = os.path.join(act_path, file)
            df = pd.read_csv(file_path)
            df["activity"] = act.split("_")[0]  # trích tên hoạt động (dws, ups, sit, ...)
            df["subject"] = file.split(".")[0]
            activity_data.append(df)

# Gộp toàn bộ thành 1 DataFrame
data = pd.concat(activity_data, ignore_index=True)

# Kiểm tra tổng quan
print(data.head())
print("\nSố lượng mẫu theo hoạt động:")
print(data["activity"].value_counts())

# Vẽ biểu đồ phân bố mẫu
plt.figure(figsize=(8,5))
sns.countplot(data=data, x="activity", order=data["activity"].value_counts().index)
plt.title("Phân bố số lượng mẫu theo hoạt động")
plt.xlabel("Hoạt động")
plt.ylabel("Số mẫu")
plt.show()

# Lọc ra các cột cảm biến chính
acc_cols = ["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]
gyro_cols = ["rotationRate.x", "rotationRate.y", "rotationRate.z"]
gravity_cols = ["gravity.x", "gravity.y", "gravity.z"]

# Tính giá trị trung bình theo hoạt động
acc_mean = data.groupby("activity")[acc_cols].mean()
gyro_mean = data.groupby("activity")[gyro_cols].mean()

# --- Biểu đồ 1: Trung bình các trục accelerometer theo hoạt động ---
plt.figure(figsize=(8,5))
acc_mean.plot(kind='bar')
plt.title("Trung bình gia tốc theo trục và hoạt động")
plt.ylabel("Giá trị trung bình (m/s²)")
plt.xlabel("Hoạt động")
plt.legend(title="Trục")
plt.tight_layout()
plt.show()

# --- Biểu đồ 2: Trung bình tốc độ quay theo trục và hoạt động ---
plt.figure(figsize=(8,5))
gyro_mean.plot(kind='bar')
plt.title("Trung bình tốc độ quay theo trục và hoạt động")
plt.ylabel("Tốc độ quay trung bình (rad/s)")
plt.xlabel("Hoạt động")
plt.legend(title="Trục")
plt.tight_layout()
plt.show()

# --- Biểu đồ 3: Độ biến thiên (Standard deviation) của gia tốc ---
acc_std = data.groupby("activity")[acc_cols].std()
plt.figure(figsize=(8,5))
acc_std.plot(kind='bar')
plt.title("Độ biến thiên của gia tốc theo hoạt động")
plt.ylabel("Độ lệch chuẩn")
plt.xlabel("Hoạt động")
plt.legend(title="Trục")
plt.tight_layout()
plt.show()