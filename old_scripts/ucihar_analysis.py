import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Đường dẫn tới thư mục dataset
DATA_PATH = r"C:\Users\Binh\Desktop\Code\human-gait-recognition-with-smartphone-sensors\data\raw\UCI HAR Dataset"

# Đọc danh sách đặc trưng và nhãn hoạt động
features = pd.read_csv(os.path.join(DATA_PATH, "features.txt"), sep="\s+", header=None, names=["index", "feature"])
activity_labels = pd.read_csv(os.path.join(DATA_PATH, "activity_labels.txt"), sep="\s+", header=None, names=["id", "activity"])

# Hàm đọc dữ liệu train/test
def load_split(split):
    X = pd.read_csv(os.path.join(DATA_PATH, split, f"X_{split}.txt"), sep="\s+", header=None)
    y = pd.read_csv(os.path.join(DATA_PATH, split, f"y_{split}.txt"), sep="\s+", header=None, names=["Activity"])
    subject = pd.read_csv(os.path.join(DATA_PATH, split, f"subject_{split}.txt"), sep="\s+", header=None, names=["Subject"])
    df = pd.concat([subject, y, X], axis=1)
    return df

# Đọc train/test và nối lại
train = load_split("train")
test = load_split("test")
data = pd.concat([train, test], axis=0).reset_index(drop=True)

# Gán tên cột cho các đặc trưng
data.columns = ["Subject", "Activity"] + features["feature"].tolist()

# Gán tên hoạt động (ví dụ 1 -> WALKING)
activity_map = dict(zip(activity_labels["id"], activity_labels["activity"]))
data["Activity"] = data["Activity"].map(activity_map)

# print("✅ Dữ liệu đọc thành công!")
# print("Kích thước dữ liệu:", data.shape)
# print(data.head())

# plt.figure(figsize=(8,5))
# sns.countplot(data=data, x="Activity", order=data["Activity"].value_counts().index)
# plt.title("Phân bố số mẫu theo hoạt động")
# plt.ylabel("Số mẫu")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,4))
# sns.countplot(data=data, x="Subject", palette="crest")
# plt.title("Phân bố số mẫu theo người tham gia (Subject ID)")
# plt.ylabel("Số mẫu")
# plt.tight_layout()
# plt.show()

# Lấy dữ liệu của một người và một hoạt động cụ thể
sample = data[(data["Subject"] == 1) & (data["Activity"] == "WALKING")].iloc[:, 2:8]  # chọn 6 đặc trưng đầu tiên

plt.figure(figsize=(10,5))
for col in sample.columns:
    plt.plot(sample[col].values[:200], label=col)
plt.title("Biểu đồ tín hiệu cảm biến (Subject 1 - Walking)")
plt.xlabel("Thời gian (mẫu)")
plt.ylabel("Giá trị đo cảm biến")
plt.legend()
plt.tight_layout()
plt.show()
