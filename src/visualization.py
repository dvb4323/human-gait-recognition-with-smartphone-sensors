# ============================================
# 📁 File: src/visualization.py
# 📦 Mục đích: Trực quan hóa dữ liệu đặc trưng
# ============================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def visualize_feature_statistics(feature_df, save_dir="reports/figures"):
    """
    Tạo biểu đồ trung bình và độ lệch chuẩn của từng đặc trưng theo nhãn
    """
    os.makedirs(save_dir, exist_ok=True)

    numeric_cols = feature_df.select_dtypes(include=["float64", "int64"]).columns
    if "label" in numeric_cols:
        numeric_cols = numeric_cols.drop("label")

    grouped = feature_df.groupby("label")[numeric_cols].mean()
    std_grouped = feature_df.groupby("label")[numeric_cols].std()

    # 🔹 Heatmap Mean
    plt.figure(figsize=(12, 6))
    sns.heatmap(grouped, cmap="YlGnBu")
    plt.title("Feature Means per Class")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_means_heatmap.png"))
    plt.close()

    # 🔹 Heatmap Std
    plt.figure(figsize=(12, 6))
    sns.heatmap(std_grouped, cmap="coolwarm")
    plt.title("Feature Std per Class")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_stds_heatmap.png"))
    plt.close()


def visualize_pca(feature_df, save_dir="reports/figures", n_components=2):
    """
    Thực hiện PCA và vẽ biểu đồ scatter giữa các lớp
    """
    os.makedirs(save_dir, exist_ok=True)
    X = feature_df.drop(columns=["label"])
    y = feature_df["label"]

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df["label"] = y

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="label", palette="tab10", alpha=0.7)
    plt.title("PCA Scatter Plot of Features")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_scatter.png"))
    plt.close()


def generate_all_visualizations(feature_file, save_dir="reports/figures"):
    """
    Hàm tổng hợp: đọc file đặc trưng, sinh tất cả biểu đồ
    """
    df = pd.read_csv(feature_file)
    visualize_feature_statistics(df, save_dir)
    visualize_pca(df, save_dir)
    print(f"✅ Đã tạo biểu đồ trực quan tại: {save_dir}")
