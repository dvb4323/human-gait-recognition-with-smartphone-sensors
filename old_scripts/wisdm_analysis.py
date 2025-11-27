# wisdm_analysis.py
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Đường dẫn tới file WISDM (phiên bản raw accelerometer)
DATA_PATH = r"/data/raw/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"

# Nơi lưu hình
OUT_DIR = r"/plots_wisdm"
os.makedirs(OUT_DIR, exist_ok=True)

pattern = re.compile(r'^\s*(\d+)\s*,\s*([^,]+)\s*,\s*([0-9]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*[,;\s]*$')

rows = []
bad_lines = 0
total_lines = 0

with open(DATA_PATH, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f, start=1):
        total_lines += 1
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            uid = int(m.group(1))
            activity = m.group(2).strip()
            timestamp = int(m.group(3))
            x = float(m.group(4))
            y = float(m.group(5))
            z = float(m.group(6))
            rows.append((uid, activity, timestamp, x, y, z))
        else:
            # Nếu dòng không match, tăng counter; có thể log ở file nếu cần debug thêm
            bad_lines += 1

print(f"Total lines read: {total_lines}")
print(f"Successfully parsed lines: {len(rows)}")
print(f"Skipped / malformed lines: {bad_lines}")

# Tạo DataFrame
df = pd.DataFrame(rows, columns=["User", "Activity", "Timestamp", "X", "Y", "Z"])

# Chuyển Timestamp sang datetime (nếu timestamp là ms)
try:
    df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], unit='ms')
except Exception:
    # nếu không phải ms, giữ nguyên timestamp
    df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Lọc chỉ các activity mong muốn (nếu cần)
valid_acts = {"Walking", "Jogging", "Sitting", "Standing", "Upstairs", "Downstairs"}
df = df[df["Activity"].isin(valid_acts)].copy()

print("After filtering valid activities:")
print(df['Activity'].value_counts())

# --- Vẽ và lưu hình ---

sns.set(style="whitegrid")

# 1) Phân bố số mẫu theo hoạt động
plt.figure(figsize=(8,5))
order = df['Activity'].value_counts().index
sns.countplot(data=df, x='Activity', order=order)
plt.title("WISDM - Phân bố số mẫu theo hoạt động")
plt.ylabel("Số mẫu")
plt.xticks(rotation=30)
plt.tight_layout()
fn1 = os.path.join(OUT_DIR, "wisdm_activity_distribution.png")
plt.savefig(fn1, dpi=200)
plt.show()

# 2) Phân bố số mẫu theo người tham gia (top 30 để biểu diễn)
plt.figure(figsize=(10,4))
top_users = df['User'].value_counts().index[:30]
sns.countplot(data=df[df['User'].isin(top_users)], x='User', order=top_users)
plt.title("WISDM - Phân bố số mẫu theo người tham gia (Top 30)")
plt.ylabel("Số mẫu")
plt.tight_layout()
fn2 = os.path.join(OUT_DIR, "wisdm_subject_distribution.png")
plt.savefig(fn2, dpi=200)
plt.show()

# 3) Minh họa tín hiệu gia tốc 3 trục cho 1 user + 1 activity
# Chọn user có nhiều mẫu, activity cụ thể
example_user = df['User'].value_counts().idxmax()
example_activity = 'Walking' if 'Walking' in df['Activity'].unique() else df['Activity'].unique()[0]
sample = df[(df['User'] == example_user) & (df['Activity'] == example_activity)].iloc[:300]

if sample.shape[0] >= 10:
    plt.figure(figsize=(10,5))
    plt.plot(sample['X'].values, label='X')
    plt.plot(sample['Y'].values, label='Y')
    plt.plot(sample['Z'].values, label='Z')
    plt.title(f"WISDM - Tín hiệu gia tốc (User {example_user} - {example_activity})")
    plt.xlabel("Mẫu thứ")
    plt.ylabel("Gia tốc")
    plt.legend()
    plt.tight_layout()
    fn3 = os.path.join(OUT_DIR, "wisdm_sensor_signal_example.png")
    plt.savefig(fn3, dpi=200)
    plt.show()
else:
    print("Không có đủ mẫu để vẽ tín hiệu ví dụ cho user/activity đã chọn.")

print("Plots saved to:", OUT_DIR)
