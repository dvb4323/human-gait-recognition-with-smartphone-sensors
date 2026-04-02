import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Cấu hình hiển thị Tiếng Việt hoặc phông chữ sạch cho báo cáo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

class OUSimilarGaitStandaloneStats:
    def __init__(self, data_root: str, output_dir: str = "reports/statistics"):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Bảng nhãn theo Documentation của OU-ISIR
        self.activity_names = {
            -1: "Invalid/Noise",
             0: "Walking", 
             1: "Upstairs", 
             2: "Downstairs", 
             3: "Up-slope", 
             4: "Down-slope",
             5: "Transition"
        }
        self.columns = ['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az', 'ClassLabel', 'StepLabel']
        self.all_summaries = [] # Lưu thông tin tóm tắt từng file để tiết kiệm RAM

    def load_and_summarize(self, limit_files=None):
        """Đọc file và trích xuất thông tin thống kê thay vì lưu toàn bộ dữ liệu thô vào RAM"""
        center_dir = self.data_root / "Center"
        if not center_dir.exists():
            print(f"Lỗi: Không tìm thấy thư mục {center_dir}")
            return

        files = sorted(list(center_dir.glob("*.txt")))
        if limit_files:
            files = files[:limit_files]

        print(f"Đang xử lý {len(files)} tệp dữ liệu...")
        
        for f in tqdm(files):
            try:
                # Đọc dữ liệu, bỏ qua 2 dòng đầu
                df = pd.read_csv(f, sep=r'\s+', skiprows=2, names=self.columns)
                subject_id = f.name.split('_')[1]
                
                # Lưu tóm tắt thay vì lưu cả dataframe
                # Tính số lượng mẫu của mỗi nhãn trong file này
                label_counts = df['ClassLabel'].value_counts().to_dict()
                
                self.all_summaries.append({
                    'subject_id': subject_id,
                    'total_samples': len(df),
                    'duration_sec': len(df) / 100.0,
                    'label_counts': label_counts
                })
            except Exception as e:
                print(f"Lỗi khi đọc file {f.name}: {e}")

    def plot_full_statistics(self):
        if not self.all_summaries:
            print("Không có dữ liệu để thống kê.")
            return

        # 1. Chuyển đổi dữ liệu tóm tắt sang DataFrame để vẽ
        summary_df = pd.DataFrame(self.all_summaries)
        
        # 2. Thống kê tổng quan nhãn (Class Distribution)
        total_label_counts = {}
        for item in self.all_summaries:
            for lbl, count in item['label_counts'].items():
                total_label_counts[lbl] = total_label_counts.get(lbl, 0) + count
        
        lbl_df = pd.DataFrame([
            {'Label': self.activity_names.get(k, f"Unknown {k}"), 'Count': v} 
            for k, v in total_label_counts.items()
        ]).sort_values(by='Count', ascending=False)

        # Vẽ biểu đồ cột Class Distribution
        
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x='Label', y='Count', data=lbl_df, palette='magma')
        
        total_samples = lbl_df['Count'].sum()
        for i, p in enumerate(ax.patches):
            percentage = '{:.1f}%'.format(100 * p.get_height() / total_samples)
            ax.annotate(f'{int(p.get_height()):,}\n({percentage})', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 20), 
                        textcoords='offset points', fontweight='bold')

        plt.title('Phân bổ dữ liệu theo nhãn (Bao gồm nhãn -1)', fontsize=16)
        plt.ylabel('Số lượng mẫu (Samples)')
        plt.savefig(self.output_dir / "full_class_dist.png", dpi=300)
        plt.close()

        # 3. Biểu đồ tròn: Tỷ lệ Dữ liệu Sạch vs Nhiễu
        invalid_count = total_label_counts.get(-1, 0)
        valid_count = total_samples - invalid_count
        
        plt.figure(figsize=(8, 8))
        plt.pie([valid_count, invalid_count], 
                labels=['Dữ liệu hữu ích (0-5)', 'Dữ liệu Invalid (-1)'],
                autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], 
                explode=(0.1, 0), startangle=140)
        plt.title('Tỷ lệ chất lượng dữ liệu tổng thể', fontsize=15)
        plt.savefig(self.output_dir / "data_quality_pie.png", dpi=300)
        plt.close()

        # 4. Biểu đồ Histogram: Thời lượng bản ghi
        
        plt.figure(figsize=(10, 6))
        sns.histplot(summary_df['duration_sec'], bins=20, kde=True, color='teal')
        plt.axvline(summary_df['duration_sec'].mean(), color='red', linestyle='--', 
                    label=f"Trung bình: {summary_df['duration_sec'].mean():.1f}s")
        plt.title('Phân bố thời lượng ghi hình của các đối tượng')
        plt.xlabel('Thời gian (giây)')
        plt.legend()
        plt.savefig(self.output_dir / "recordings_duration_dist.png", dpi=300)
        plt.close()

        print(f"Xong! Các biểu đồ đã được lưu tại: {self.output_dir.absolute()}")
    
    def plot_signature_features(eda_instance, output_path="reports/activity_signature.png"):
    # 1. Gom dữ liệu sạch
        all_data = pd.concat(eda_instance.sample_data.values(), ignore_index=True)
        clean_data = all_data[all_data['ClassLabel'].isin([0, 1, 2, 3, 4])].copy()
    
    # Định nghĩa tên tiếng Việt cho dễ đọc trong báo cáo
        names = {0: "Đi bộ", 1: "Lên cầu thang", 2: "Xuống cầu thang", 3: "Lên dốc", 4: "Xuống dốc"}
        clean_data['Hành vi'] = clean_data['ClassLabel'].map(names)
    
    # 2. Tính Độ lớn gia tốc tổng hợp (đặc trưng quan trọng nhất)
        clean_data['Gia tốc tổng'] = np.sqrt(clean_data['Ax']**2 + clean_data['Ay']**2 + clean_data['Az']**2)

    # 3. Vẽ biểu đồ Ridge Plot (Biểu đồ dải núi)
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # Tạo bảng màu chuyển từ xanh sang đỏ (tượng trưng cho cường độ)
        pal = sns.cubehelix_palette(5, rot=-.25, light=.7)
    
        g = sns.FacetGrid(clean_data, row="Hành vi", hue="Hành vi", aspect=15, height=1.2, palette=pal)

    # Vẽ các đường mật độ
        g.map(sns.kdeplot, "Gia tốc tổng", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, "Gia tốc tổng", bw_adjust=.5, clip_on=False, color="w", lw=2)
    
    # Vẽ đường thẳng trục cơ sở
        g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Thêm tên hành vi vào từng dòng
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

        g.map(label, "Gia tốc tổng")

    # Tinh chỉnh thẩm mỹ
        g.fig.subplots_adjust(hspace=-.25)
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
    
        plt.suptitle("Dấu vân tay đặc trưng của các hành vi (Cường độ gia tốc)", fontsize=16, fontweight='bold')
        plt.xlabel("Độ lớn gia tốc (g) - Càng về bên phải vận động càng mạnh")
    
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ đặc trưng tại: {output_path}")
        plt.show()

if __name__ == "__main__":
    # --- THAY ĐỔI ĐƯỜNG DẪN TẠI ĐÂY ---
    DATASET_PATH = "data/raw/OU-SimilarGaitActivities" 
    
    stats = OUSimilarGaitStandaloneStats(DATASET_PATH)
    
    # Bạn có thể bỏ limit_files=100 để chạy TOÀN BỘ file
    stats.load_and_summarize(limit_files=None) 
    stats.plot_full_statistics()
    stats.plot_signature_features()