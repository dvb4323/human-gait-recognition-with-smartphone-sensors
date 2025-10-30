import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn đến thư mục HAPT Dataset sau khi giải nén
DATASET_PATH = r"C:\Users\Binh\Desktop\Code\human-gait-recognition-with-smartphone-sensors\data\raw\HAPT"

# Các file cần đọc
signals_path = os.path.join(DATASET_PATH, "RawData")
X_train = pd.read_csv(os.path.join(DATASET_PATH, "Train", "X_train.txt"), delim_whitespace=True, header=None)
y_train = pd.read_csv(os.path.join(DATASET_PATH, "Train", "y_train.txt"), header=None, names=["activity"])
subject_train = pd.read_csv(os.path.join(DATASET_PATH, "Train", "subject_id_train.txt"), header=None, names=["subject"])

X_test = pd.read_csv(os.path.join(DATASET_PATH, "Test", "X_test.txt"), delim_whitespace=True, header=None)
y_test = pd.read_csv(os.path.join(DATASET_PATH, "Test", "y_test.txt"), header=None, names=["activity"])
subject_test = pd.read_csv(os.path.join(DATASET_PATH, "Test", "subject_id_test.txt"), header=None, names=["subject"])

# Gộp dữ liệu train + test
X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
subjects = pd.concat([subject_train, subject_test], axis=0).reset_index(drop=True)

# Tạo DataFrame tổng hợp
data = pd.concat([subjects, y], axis=1)
print("✅ Dữ liệu HAPT đã đọc thành công.")
print("Số mẫu:", len(data))
print("Số người tham gia:", data['subject'].nunique())
print("Số nhãn hoạt động:", data['activity'].nunique())

plt.figure(figsize=(10,6))
sns.countplot(x="activity", data=data, palette="tab10")
plt.title("Phân bố số mẫu theo hoạt động (HAPT Dataset)")
plt.xlabel("Mã hoạt động")
plt.ylabel("Số mẫu")
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(x="subject", data=data, palette="viridis")
plt.title("Phân bố số mẫu theo người tham gia")
plt.xlabel("Người tham gia")
plt.ylabel("Số mẫu")
plt.show()

import numpy as np

# Đọc 1 file ví dụ accelerometer và gyroscope
acc_x = pd.read_csv(os.path.join(DATASET_PATH, "Train", "Inertial Signals", "body_acc_x_train.txt"), delim_whitespace=True, header=None)
gyro_x = pd.read_csv(os.path.join(DATASET_PATH, "Train", "Inertial Signals", "body_gyro_x_train.txt"), delim_whitespace=True, header=None)

# Chọn 1 mẫu ngẫu nhiên
sample_idx = 10
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(acc_x.iloc[sample_idx].values)
plt.title("Gia tốc (Accelerometer X) – mẫu {}".format(sample_idx))
plt.ylabel("Acceleration (g)")

plt.subplot(2,1,2)
plt.plot(gyro_x.iloc[sample_idx].values)
plt.title("Tốc độ góc (Gyroscope X) – mẫu {}".format(sample_idx))
plt.ylabel("Angular Velocity (rad/s)")
plt.xlabel("Thời gian (samples)")
plt.tight_layout()
plt.show()
