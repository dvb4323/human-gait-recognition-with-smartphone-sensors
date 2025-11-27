"""
File: src/utils.py
Mô tả: Hàm tiện ích dùng chung (ví dụ: tạo cửa sổ trượt, logging đơn giản)
"""
import numpy as np

def sliding_window(data, labels, window_size, overlap):
    """
    Tạo các cửa sổ trượt (windowing) cho dữ liệu chuỗi thời gian.
    data: ndarray [n_samples, n_features]
    labels: ndarray [n_samples]
    window_size: số điểm trong 1 cửa sổ (VD: 128)
    overlap: tỉ lệ chồng lấp (VD: 0.5 => stride = 64)
    """
    data = np.asarray(data)
    labels = np.asarray(labels).flatten()
    stride = int(window_size * (1 - overlap))
    if stride <= 0: raise ValueError("Overlap quá lớn, stride <= 0")
    X, y = [], []
    for start in range(0, len(data) - window_size + 1, stride):
        end = start + window_size
        X.append(data[start:end])
        # Nhãn gán theo giá trị mode (phần lớn trong cửa sổ)
        window_label = np.bincount(labels[start:end]).argmax()
        y.append(window_label)
    return np.array(X), np.array(y)
