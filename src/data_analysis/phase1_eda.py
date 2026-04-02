import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class OUSimilarGaitEDA:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.center_dir = self.data_root / "Center"
        self.left_dir = self.data_root / "Left"
        self.right_dir = self.data_root / "Right"
        self.protocol_dir = self.data_root / "PaperProtocol"
        
        # Data containers
        self.file_info = {}
        self.sample_data = {}
        self.statistics = {}
        
        # Column names
        self.columns = ['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az', 'ClassLabel', 'StepLabel']
        
    def load_single_file(self, filepath: Path) -> pd.DataFrame:
        try:
            # Skip first 2 lines (LineWidth: 8 and column headers)
            # Then read data with our predefined column names
            df = pd.read_csv(filepath, sep=r'\s+', skiprows=2, names=self.columns)
            
            # Convert to numeric, coerce errors to NaN
            for col in self.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows that are all NaN (in case of extra blank lines)
            df = df.dropna(how='all')
            
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def get_file_list(self, position: str = 'Center') -> List[Path]:
        position_dir = self.data_root / position
        return sorted(position_dir.glob("*.txt"))
    
    def extract_subject_id(self, filename: str) -> str:
        # Format: T0_Id######_ActLabelAndStepInfor.txt
        return filename.split('_')[1]
    
    def phase1_1_basic_statistics(self):
        print("=" * 80)
        print("PHASE 1.1: DATA LOADING & BASIC STATISTICS")
        print("=" * 80)
        
        # Count files per position
        center_files = self.get_file_list('Center')
        left_files = self.get_file_list('Left')
        right_files = self.get_file_list('Right')
        
        print(f"\nFile Counts:")
        print(f"  Center: {len(center_files)} files")
        print(f"  Left:   {len(left_files)} files")
        print(f"  Right:  {len(right_files)} files")
        
        # Get unique subjects
        center_subjects = {self.extract_subject_id(f.name) for f in center_files}
        left_subjects = {self.extract_subject_id(f.name) for f in left_files}
        right_subjects = {self.extract_subject_id(f.name) for f in right_files}
        
        all_subjects = center_subjects | left_subjects | right_subjects
        
        print(f"\nSubject Counts:")
        print(f"  Total unique subjects: {len(all_subjects)}")
        print(f"  Subjects with Center data: {len(center_subjects)}")
        print(f"  Subjects with Left data: {len(left_subjects)}")
        print(f"  Subjects with Right data: {len(right_subjects)}")
        
        # Find subjects with all three positions
        complete_subjects = center_subjects & left_subjects & right_subjects
        print(f"  Subjects with all 3 positions: {len(complete_subjects)}")
        
        # Find missing data
        missing_left = center_subjects - left_subjects
        missing_right = center_subjects - right_subjects
        
        if missing_left:
            print(f"\nSubjects missing Left sensor: {len(missing_left)}")
        if missing_right:
            print(f"Subjects missing Right sensor: {len(missing_right)}")
        
        # Load sample files to analyze
        print(f"\nLoading sample files for analysis...")
        sample_files = center_files[:10]  # Load first 10 files
        
        samples_per_file = []
        durations = []
        
        for filepath in tqdm(sample_files, desc="Loading samples"):
            df = self.load_single_file(filepath)
            if df is not None:
                subject_id = self.extract_subject_id(filepath.name)
                self.sample_data[subject_id] = df
                
                num_samples = len(df)
                duration = num_samples / 100.0  # 100 Hz sampling
                
                samples_per_file.append(num_samples)
                durations.append(duration)
        
        print(f"\nSample Statistics (from {len(sample_files)} files):")
        print(f"  Samples per file - Mean: {np.mean(samples_per_file):.0f}, "
              f"Std: {np.std(samples_per_file):.0f}")
        print(f"  Samples per file - Min: {np.min(samples_per_file)}, "
              f"Max: {np.max(samples_per_file)}")
        print(f"  Duration (seconds) - Mean: {np.mean(durations):.2f}s, "
              f"Std: {np.std(durations):.2f}s")
        print(f"  Duration (seconds) - Min: {np.min(durations):.2f}s, "
              f"Max: {np.max(durations):.2f}s")
        
        # Store statistics
        self.statistics['file_counts'] = {
            'center': len(center_files),
            'left': len(left_files),
            'right': len(right_files)
        }
        self.statistics['subject_counts'] = {
            'total': len(all_subjects),
            'complete': len(complete_subjects),
            'missing_left': len(missing_left),
            'missing_right': len(missing_right)
        }
        self.statistics['sample_stats'] = {
            'mean_samples': np.mean(samples_per_file),
            'std_samples': np.std(samples_per_file),
            'mean_duration': np.mean(durations),
            'std_duration': np.std(durations)
        }
        
        return self.statistics
    
    def phase1_2_sensor_distribution(self):
        print("\n" + "=" * 80)
        print("PHASE 1.2: SENSOR DATA DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        if not self.sample_data:
            print("No sample data loaded. Run phase1_1_basic_statistics() first.")
            return
        
        # Combine all sample data
        all_data = pd.concat(self.sample_data.values(), ignore_index=True)
        
        print(f"\nCombined sample data: {len(all_data)} total samples")
        
        # Accelerometer statistics
        print("\nACCELEROMETER STATISTICS:")
        accel_cols = ['Ax', 'Ay', 'Az']
        accel_stats = all_data[accel_cols].describe()
        print(accel_stats)
        
        # Gyroscope statistics
        print("\nGYROSCOPE STATISTICS:")
        gyro_cols = ['Gx', 'Gy', 'Gz']
        gyro_stats = all_data[gyro_cols].describe()
        print(gyro_stats)
        
        # Check for missing values
        missing = all_data.isnull().sum()
        if missing.sum() > 0:
            print("\nMISSING VALUES DETECTED:")
            print(missing[missing > 0])
        else:
            print("\nNo missing values detected")
        
        # Correlation analysis
        print("\nCORRELATION ANALYSIS:")
        print("\nAccelerometer correlation:")
        print(all_data[accel_cols].corr())
        print("\nGyroscope correlation:")
        print(all_data[gyro_cols].corr())
        
        # Store statistics
        self.statistics['sensor_stats'] = {
            'accelerometer': accel_stats.to_dict(),
            'gyroscope': gyro_stats.to_dict()
        }
        
        return accel_stats, gyro_stats
    
    def phase1_3_activity_analysis(self):
        print("\n" + "=" * 80)
        print("PHASE 1.3: ACTIVITY CLASS ANALYSIS")
        print("=" * 80)
        
        if not self.sample_data:
            print("No sample data loaded. Run phase1_1_basic_statistics() first.")
            return
        
        # Combine all sample data
        all_data = pd.concat(self.sample_data.values(), ignore_index=True)
        
        # Class distribution
        class_counts = all_data['ClassLabel'].value_counts().sort_index()
        
        print("\nACTIVITY CLASS DISTRIBUTION:")
        print(class_counts)
        print(f"\nTotal samples: {len(all_data)}")
        
        print("\nClass Percentages:")
        class_pct = (class_counts / len(all_data) * 100).round(2)
        for cls, pct in class_pct.items():
            print(f"  Class {cls}: {pct}%")
        
        # Activity names (based on dataset description)
        activity_names = {
            0: "Walking (flat ground)",
            1: "Walking up stairs",
            2: "Walking down stairs",
            3: "Walking up slope",
            4: "Walking down slope"
        }
        
        print("\nACTIVITY LABELS:")
        for cls, name in activity_names.items():
            if cls in class_counts.index:
                print(f"  Class {cls}: {name} ({class_counts[cls]} samples)")
        
        # Analyze transitions
        print("\nACTIVITY TRANSITIONS:")
        transitions = []
        for subject_id, df in self.sample_data.items():
            class_changes = df['ClassLabel'].diff().fillna(0) != 0
            num_transitions = class_changes.sum()
            transitions.append(num_transitions)
        
        print(f"  Average transitions per recording: {np.mean(transitions):.2f}")
        print(f"  Min transitions: {np.min(transitions)}")
        print(f"  Max transitions: {np.max(transitions)}")
        
        # Store statistics
        self.statistics['activity_stats'] = {
            'class_counts': class_counts.to_dict(),
            'class_percentages': class_pct.to_dict(),
            'avg_transitions': np.mean(transitions)
        }
        
        return class_counts
    
    def phase1_4_step_analysis(self):
        print("\n" + "=" * 80)
        print("PHASE 1.4: STEP LABEL ANALYSIS")
        print("=" * 80)
        
        if not self.sample_data:
            print("No sample data loaded. Run phase1_1_basic_statistics() first.")
            return
        
        # Combine all sample data
        all_data = pd.concat(self.sample_data.values(), ignore_index=True)
        
        # Step vs non-step data
        non_step_count = (all_data['StepLabel'] == -1).sum()
        step_count = (all_data['StepLabel'] != -1).sum()
        
        print(f"\nSTEP LABEL DISTRIBUTION:")
        print(f"  Non-step data (label = -1): {non_step_count} samples ({non_step_count/len(all_data)*100:.2f}%)")
        print(f"  Step data (label >= 0): {step_count} samples ({step_count/len(all_data)*100:.2f}%)")
        
        # Unique step labels
        step_labels = all_data[all_data['StepLabel'] != -1]['StepLabel'].unique()
        print(f"\n  Unique step labels: {len(step_labels)}")
        
        # Steps per subject
        steps_per_subject = []
        for subject_id, df in self.sample_data.items():
            unique_steps = df[df['StepLabel'] != -1]['StepLabel'].nunique()
            steps_per_subject.append(unique_steps)
        
        print(f"\nSTEPS PER RECORDING:")
        print(f"  Mean: {np.mean(steps_per_subject):.2f}")
        print(f"  Std: {np.std(steps_per_subject):.2f}")
        print(f"  Min: {np.min(steps_per_subject)}")
        print(f"  Max: {np.max(steps_per_subject)}")
        
        # Store statistics
        self.statistics['step_stats'] = {
            'non_step_count': int(non_step_count),
            'step_count': int(step_count),
            'avg_steps_per_recording': np.mean(steps_per_subject)
        }
        
        return self.statistics['step_stats']
    
    def phase1_5_subject_analysis(self):
        print("\n" + "=" * 80)
        print("PHASE 1.5: SUBJECT-LEVEL ANALYSIS")
        print("=" * 80)
        
        # Load gallery and probe lists
        gallery_file = self.protocol_dir / "gallery_list.txt"
        probe_file = self.protocol_dir / "probe_list.txt"
        
        if gallery_file.exists() and probe_file.exists():
            with open(gallery_file, 'r') as f:
                gallery_subjects = [line.strip().split('_')[1] for line in f if line.strip()]
            
            with open(probe_file, 'r') as f:
                probe_subjects = [line.strip().split('_')[1] for line in f if line.strip()]
            
            print(f"\nGALLERY/PROBE PROTOCOL:")
            print(f"  Gallery subjects: {len(gallery_subjects)}")
            print(f"  Probe subjects: {len(probe_subjects)}")
            
            # Check for overlap
            overlap = set(gallery_subjects) & set(probe_subjects)
            if overlap:
                print(f" Overlap: {len(overlap)} subjects in both sets")
            else:
                print(f" No overlap between gallery and probe")
            
            self.statistics['protocol'] = {
                'gallery_count': len(gallery_subjects),
                'probe_count': len(probe_subjects),
                'overlap': len(overlap)
            }
        
        # Subject variability
        print(f"\nSUBJECT VARIABILITY (from {len(self.sample_data)} samples):")
        
        subject_durations = []
        subject_activities = []
        
        for subject_id, df in self.sample_data.items():
            duration = len(df) / 100.0
            num_activities = df['ClassLabel'].nunique()
            
            subject_durations.append(duration)
            subject_activities.append(num_activities)
        
        print(f"  Recording duration - Mean: {np.mean(subject_durations):.2f}s, "
              f"Std: {np.std(subject_durations):.2f}s")
        print(f"  Activities per subject - Mean: {np.mean(subject_activities):.2f}, "
              f"Std: {np.std(subject_activities):.2f}")
        
        return self.statistics
    
    def run_all_phase1(self):
        print("\n" + "-" * 40)
        print("RUNNING COMPLETE PHASE 1 EDA")
        print("-" * 40)
        
        self.phase1_1_basic_statistics()
        self.phase1_2_sensor_distribution()
        self.phase1_3_activity_analysis()
        self.phase1_4_step_analysis()
        self.phase1_5_subject_analysis()
        
        print("\n" + "-" * 40)
        print("PHASE 1 EDA COMPLETE!")
        print("-" * 40)
        
        return self.statistics
    
    def save_statistics(self, output_path: str):
        import json
        
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        stats_converted = convert_types(self.statistics)
        
        with open(output_path, 'w') as f:
            json.dump(stats_converted, f, indent=2)
        
        print(f"\nStatistics saved to: {output_path}")


def main():
    # Set data path
    data_root = "data/raw/OU-SimilarGaitActivities"
    
    # Initialize EDA
    print("Initializing OU-SimilarGaitActivities EDA...")
    eda = OUSimilarGaitEDA(data_root)
    
    # Run all Phase 1 analyses
    statistics = eda.run_all_phase1()
    
    # Save statistics
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    eda.save_statistics(output_dir / "phase1_statistics.json")
    
    print("\n" + "=" * 80)
    print("Phase 1 EDA completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
