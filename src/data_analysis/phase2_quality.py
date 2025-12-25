import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class OUSimilarGaitQualityAssessment:
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.center_dir = self.data_root / "Center"
        
        # Quality metrics storage
        self.quality_metrics = {
            'missing_values': {},
            'outliers': {},
            'static_periods': {},
            'corrupted_files': [],
            'sensor_issues': {}
        }
        
        # Column names
        self.columns = ['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az', 'ClassLabel', 'StepLabel']
        self.sensor_cols = ['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az']
        
    def load_single_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(filepath, sep=r'\s+', skiprows=2, names=self.columns)
            
            # Convert to numeric
            for col in self.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(how='all')
            return df
        except Exception as e:
            return None
    
    def phase2_1_missing_values(self, max_files: int = None):
        print("=" * 80)
        print("PHASE 2.1: MISSING VALUE DETECTION")
        print("=" * 80)
        
        center_files = sorted(self.center_dir.glob("*.txt"))
        if max_files:
            center_files = center_files[:max_files]
        
        print(f"\nAnalyzing {len(center_files)} files...")
        
        files_with_missing = []
        missing_summary = {col: 0 for col in self.columns}
        total_samples = 0
        
        for filepath in tqdm(center_files, desc="Checking missing values"):
            df = self.load_single_file(filepath)
            
            if df is None:
                self.quality_metrics['corrupted_files'].append(str(filepath.name))
                continue
            
            # Check for missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                files_with_missing.append({
                    'file': filepath.name,
                    'missing_counts': missing.to_dict(),
                    'total_samples': len(df)
                })
                
                for col in self.columns:
                    missing_summary[col] += missing[col]
            
            total_samples += len(df)
        
        # Report results
        print(f"\nFILES ANALYZED: {len(center_files)}")
        print(f"TOTAL SAMPLES: {total_samples:,}")
        
        if files_with_missing:
            print(f"\nFILES WITH MISSING VALUES: {len(files_with_missing)}")
            print(f"\nMissing value summary:")
            for col, count in missing_summary.items():
                if count > 0:
                    pct = (count / total_samples) * 100
                    print(f"  {col}: {count} ({pct:.4f}%)")
        else:
            print(f"\nNO MISSING VALUES DETECTED")
        
        self.quality_metrics['missing_values'] = {
            'files_with_missing': len(files_with_missing),
            'missing_summary': missing_summary,
            'total_samples': total_samples
        }
        
        return files_with_missing
    
    def phase2_2_outlier_detection(self, max_files: int = None):
        print("\n" + "=" * 80)
        print("PHASE 2.2: OUTLIER DETECTION")
        print("=" * 80)
        
        center_files = sorted(self.center_dir.glob("*.txt"))
        if max_files:
            center_files = center_files[:max_files]
        
        # Collect all data for statistical analysis
        all_data = []
        
        print(f"\nLoading {len(center_files)} files for outlier analysis...")
        for filepath in tqdm(center_files, desc="Loading data"):
            df = self.load_single_file(filepath)
            if df is not None:
                all_data.append(df)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        print(f"\nAnalyzing {len(combined_data):,} total samples...")
        
        # Calculate outliers using IQR method
        outlier_results = {}
        
        for col in self.sensor_cols:
            Q1 = combined_data[col].quantile(0.25)
            Q3 = combined_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR  # 3*IQR for extreme outliers
            upper_bound = Q3 + 3 * IQR
            
            outliers = combined_data[(combined_data[col] < lower_bound) | 
                                    (combined_data[col] > upper_bound)]
            
            outlier_results[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(combined_data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_value': combined_data[col].min(),
                'max_value': combined_data[col].max()
            }
        
        # Report results
        print(f"\nOUTLIER DETECTION RESULTS (3*IQR method):")
        print(f"\n{'Sensor':<6} {'Outliers':<10} {'Percentage':<12} {'Bounds':<30} {'Actual Range':<20}")
        print("-" * 80)
        
        for col, results in outlier_results.items():
            bounds_str = f"[{results['lower_bound']:.3f}, {results['upper_bound']:.3f}]"
            range_str = f"[{results['min_value']:.3f}, {results['max_value']:.3f}]"
            print(f"{col:<6} {results['count']:<10} {results['percentage']:<12.4f}% "
                  f"{bounds_str:<30} {range_str:<20}")
        
        self.quality_metrics['outliers'] = outlier_results
        
        return outlier_results
    
    def phase2_3_static_period_detection(self, max_files: int = 50, threshold: float = 0.01):
        print("\n" + "=" * 80)
        print("PHASE 2.3: STATIC PERIOD DETECTION")
        print("=" * 80)
        
        center_files = sorted(self.center_dir.glob("*.txt"))[:max_files]
        
        print(f"\nAnalyzing {len(center_files)} files...")
        print(f"Threshold: {threshold} (std of sensor magnitude)")
        
        static_periods = []
        
        for filepath in tqdm(center_files, desc="Detecting static periods"):
            df = self.load_single_file(filepath)
            if df is None:
                continue
            
            # Calculate sensor magnitudes
            accel_mag = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2)
            gyro_mag = np.sqrt(df['Gx']**2 + df['Gy']**2 + df['Gz']**2)
            
            # Detect static periods using rolling window
            window_size = 100  # 1 second at 100 Hz
            accel_std = accel_mag.rolling(window=window_size).std()
            gyro_std = gyro_mag.rolling(window=window_size).std()
            
            # Static if both accelerometer and gyroscope have low std
            is_static = (accel_std < threshold) & (gyro_std < threshold)
            static_ratio = is_static.sum() / len(df)
            
            if static_ratio > 0.1:  # More than 10% static
                static_periods.append({
                    'file': filepath.name,
                    'static_ratio': static_ratio,
                    'static_samples': is_static.sum(),
                    'total_samples': len(df)
                })
        
        # Report results
        if static_periods:
            print(f"\nFILES WITH SIGNIFICANT STATIC PERIODS: {len(static_periods)}")
            print(f"\nTop 10 files with most static periods:")
            sorted_static = sorted(static_periods, key=lambda x: x['static_ratio'], reverse=True)[:10]
            for item in sorted_static:
                print(f"  {item['file']}: {item['static_ratio']*100:.2f}% static "
                      f"({item['static_samples']}/{item['total_samples']} samples)")
        else:
            print(f"\nNO SIGNIFICANT STATIC PERIODS DETECTED")
        
        self.quality_metrics['static_periods'] = {
            'files_with_static': len(static_periods),
            'avg_static_ratio': np.mean([s['static_ratio'] for s in static_periods]) if static_periods else 0
        }
        
        return static_periods
    
    def phase2_4_sensor_drift_analysis(self, max_files: int = 50):
        print("\n" + "=" * 80)
        print("PHASE 2.4: SENSOR DRIFT ANALYSIS")
        print("=" * 80)
        
        center_files = sorted(self.center_dir.glob("*.txt"))[:max_files]
        
        print(f"\nAnalyzing {len(center_files)} files...")
        
        drift_results = []
        
        for filepath in tqdm(center_files, desc="Analyzing drift"):
            df = self.load_single_file(filepath)
            if df is None or len(df) < 1000:  # Skip short recordings
                continue
            
            # Calculate drift as difference between first and last 100 samples
            first_100 = df[self.sensor_cols].iloc[:100].mean()
            last_100 = df[self.sensor_cols].iloc[-100:].mean()
            
            drift = (last_100 - first_100).abs()
            max_drift = drift.max()
            
            if max_drift > 0.1:  # Significant drift threshold
                drift_results.append({
                    'file': filepath.name,
                    'max_drift': max_drift,
                    'drift_per_sensor': drift.to_dict()
                })
        
        # Report results
        if drift_results:
            print(f"\nFILES WITH POTENTIAL DRIFT: {len(drift_results)}")
            print(f"\nTop 10 files with most drift:")
            sorted_drift = sorted(drift_results, key=lambda x: x['max_drift'], reverse=True)[:10]
            for item in sorted_drift:
                print(f"  {item['file']}: max drift = {item['max_drift']:.4f}")
        else:
            print(f"\nNO SIGNIFICANT SENSOR DRIFT DETECTED")
        
        self.quality_metrics['sensor_issues']['drift'] = {
            'files_with_drift': len(drift_results),
            'avg_max_drift': np.mean([d['max_drift'] for d in drift_results]) if drift_results else 0
        }
        
        return drift_results
    
    def phase2_5_timestamp_continuity(self, max_files: int = 50):
        print("\n" + "=" * 80)
        print("PHASE 2.5: TIMESTAMP CONTINUITY VERIFICATION")
        print("=" * 80)
        
        center_files = sorted(self.center_dir.glob("*.txt"))[:max_files]
        
        print(f"\nAnalyzing {len(center_files)} files...")
        print(f"Expected sampling rate: 100 Hz")
        
        # Check if all files have consistent sample counts
        sample_counts = []
        
        for filepath in tqdm(center_files, desc="Checking continuity"):
            df = self.load_single_file(filepath)
            if df is not None:
                sample_counts.append(len(df))
        
        # Report statistics
        print(f"\nSAMPLE COUNT STATISTICS:")
        print(f"  Mean: {np.mean(sample_counts):.0f} samples")
        print(f"  Std: {np.std(sample_counts):.0f} samples")
        print(f"  Min: {np.min(sample_counts)} samples ({np.min(sample_counts)/100:.2f}s)")
        print(f"  Max: {np.max(sample_counts)} samples ({np.max(sample_counts)/100:.2f}s)")
        
        # Check for very short or very long recordings
        short_files = [c for c in sample_counts if c < 1000]  # < 10 seconds
        long_files = [c for c in sample_counts if c > 6000]   # > 60 seconds
        
        if short_files:
            print(f"\n{len(short_files)} files with < 10 seconds of data")
        if long_files:
            print(f"{len(long_files)} files with > 60 seconds of data")
        
        self.quality_metrics['timestamp_continuity'] = {
            'mean_samples': np.mean(sample_counts),
            'std_samples': np.std(sample_counts),
            'short_files': len(short_files),
            'long_files': len(long_files),
        }
        
        return sample_counts
    
    def run_all_phase2(self, max_files: int = None):
        print("\n" + "-" * 40)
        print("RUNNING COMPLETE PHASE 2 DATA QUALITY ASSESSMENT")
        print("-" * 40)
        
        self.phase2_1_missing_values(max_files)
        self.phase2_2_outlier_detection(max_files)
        self.phase2_3_static_period_detection(max_files or 50)
        self.phase2_4_sensor_drift_analysis(max_files or 50)
        self.phase2_5_timestamp_continuity(max_files or 50)
        
        # Summary
        print("\n" + "=" * 80)
        print("PHASE 2 QUALITY ASSESSMENT SUMMARY")
        print("=" * 80)
        
        print(f"\nCorrupted files: {len(self.quality_metrics['corrupted_files'])}")
        print(f"Files with missing values: {self.quality_metrics['missing_values'].get('files_with_missing', 0)}")
        print(f"Files with static periods: {self.quality_metrics['static_periods'].get('files_with_static', 0)}")
        print(f"Files with drift: {self.quality_metrics['sensor_issues'].get('drift', {}).get('files_with_drift', 0)}")
        
        print("\n" + "-" * 40)
        print("PHASE 2 QUALITY ASSESSMENT COMPLETE!")
        print("-" * 40)
        
        return self.quality_metrics
    
    def save_quality_report(self, output_path: str):
        import json
        
        # Convert numpy types
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        metrics_converted = convert_types(self.quality_metrics)
        
        with open(output_path, 'w') as f:
            json.dump(metrics_converted, f, indent=2)
        
        print(f"\n Quality report saved to: {output_path}")


def main():
    # Set data path
    data_root = "data/raw/OU-SimilarGaitActivities"
    
    # Initialize quality assessment
    print("Initializing OU-SimilarGaitActivities Quality Assessment...")
    qa = OUSimilarGaitQualityAssessment(data_root)
    
    # Run all Phase 2 assessments
    # Use max_files=None to analyze ALL files, or set a number for testing
    quality_metrics = qa.run_all_phase2(max_files=None)  # Analyze first 100 files
    
    # Save quality report
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    qa.save_quality_report(output_dir / "phase2_quality_report.json")
    
    print("\n" + "=" * 80)
    print("Phase 2 Quality Assessment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
