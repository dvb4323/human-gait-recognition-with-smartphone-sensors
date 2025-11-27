"""
Data loader for OU-SimilarGaitActivities dataset.
Loads Center sensor data and filters unlabeled samples.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json


class OUGaitDataLoader:
    """Load and prepare OU-SimilarGaitActivities data for preprocessing."""
    
    def __init__(self, data_root: str, sensor_position: str = 'Center'):
        """
        Initialize data loader.
        
        Args:
            data_root: Path to OU-SimilarGaitActivities directory
            sensor_position: Sensor position ('Center', 'Left', or 'Right')
        """
        self.data_root = Path(data_root)
        self.sensor_dir = self.data_root / sensor_position
        self.protocol_dir = self.data_root / "PaperProtocol"
        
        # Column names
        self.columns = ['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az', 'ClassLabel', 'StepLabel']
        self.sensor_cols = ['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az']
        
        # Data storage
        self.data = {}
        self.metadata = {}
        
        # Load gallery/probe lists
        self.gallery_subjects = self._load_subject_list('gallery_list.txt')
        self.probe_subjects = self._load_subject_list('probe_list.txt')
        
    def _load_subject_list(self, filename: str) -> List[str]:
        """Load gallery or probe subject list."""
        filepath = self.protocol_dir / filename
        with open(filepath, 'r') as f:
            # Extract subject IDs from format: T0_Id######
            subjects = [line.strip().split('_')[1] for line in f if line.strip()]
        return subjects
    
    def _load_single_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load a single data file."""
        try:
            df = pd.read_csv(filepath, sep=r'\s+', skiprows=2, names=self.columns)
            
            # Convert to numeric
            for col in self.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(how='all')
            return df
        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")
            return None
    
    def _extract_subject_id(self, filename: str) -> str:
        """Extract subject ID from filename."""
        return filename.split('_')[1]
    
    def load_all_data(self, remove_unlabeled: bool = True, verbose: bool = True) -> Dict:
        """
        Load all data files.
        
        Args:
            remove_unlabeled: If True, remove samples with ClassLabel = -1
            verbose: Print progress information
            
        Returns:
            Dictionary with subject data and metadata
        """
        files = sorted(self.sensor_dir.glob("*.txt"))
        
        if verbose:
            print(f"Loading {len(files)} files from {self.sensor_dir.name}...")
        
        total_samples_before = 0
        total_samples_after = 0
        class_counts = {i: 0 for i in range(5)}  # Classes 0-4
        
        for filepath in tqdm(files, desc="Loading data", disable=not verbose):
            df = self._load_single_file(filepath)
            if df is None:
                continue
            
            subject_id = self._extract_subject_id(filepath.name)
            total_samples_before += len(df)
            
            # Remove unlabeled data if requested
            if remove_unlabeled:
                df = df[df['ClassLabel'] != -1].copy()
            
            # Skip if no labeled data remains
            if len(df) == 0:
                continue
            
            total_samples_after += len(df)
            
            # Count classes
            for class_label in df['ClassLabel'].unique():
                if class_label in class_counts:
                    class_counts[int(class_label)] += (df['ClassLabel'] == class_label).sum()
            
            # Store data
            self.data[subject_id] = {
                'sensor_data': df[self.sensor_cols].values,  # Shape: (n_samples, 6)
                'class_labels': df['ClassLabel'].values,      # Shape: (n_samples,)
                'step_labels': df['StepLabel'].values,        # Shape: (n_samples,)
                'subject_id': subject_id,
                'is_gallery': subject_id in self.gallery_subjects,
                'is_probe': subject_id in self.probe_subjects
            }
        
        # Store metadata
        self.metadata = {
            'num_subjects': len(self.data),
            'total_samples_before': total_samples_before,
            'total_samples_after': total_samples_after,
            'samples_removed': total_samples_before - total_samples_after,
            'removal_percentage': ((total_samples_before - total_samples_after) / total_samples_before * 100) if total_samples_before > 0 else 0,
            'class_counts': class_counts,
            'gallery_subjects': len([s for s in self.data.values() if s['is_gallery']]),
            'probe_subjects': len([s for s in self.data.values() if s['is_probe']])
        }
        
        if verbose:
            self._print_summary()
        
        return self.data
    
    def _print_summary(self):
        """Print data loading summary."""
        print("\n" + "=" * 80)
        print("DATA LOADING SUMMARY")
        print("=" * 80)
        print(f"\n📁 Subjects loaded: {self.metadata['num_subjects']}")
        print(f"📊 Total samples (before filtering): {self.metadata['total_samples_before']:,}")
        print(f"📊 Total samples (after filtering): {self.metadata['total_samples_after']:,}")
        print(f"🗑️  Samples removed: {self.metadata['samples_removed']:,} ({self.metadata['removal_percentage']:.2f}%)")
        
        print(f"\n👥 Gallery subjects: {self.metadata['gallery_subjects']}")
        print(f"👥 Probe subjects: {self.metadata['probe_subjects']}")
        
        print(f"\n📈 Class distribution (after filtering):")
        total = sum(self.metadata['class_counts'].values())
        for class_id, count in sorted(self.metadata['class_counts'].items()):
            pct = (count / total * 100) if total > 0 else 0
            print(f"  Class {class_id}: {count:,} samples ({pct:.2f}%)")
    
    def get_subject_data(self, subject_id: str) -> Optional[Dict]:
        """Get data for a specific subject."""
        return self.data.get(subject_id)
    
    def get_gallery_subjects(self) -> List[str]:
        """Get list of gallery subject IDs."""
        return [sid for sid, data in self.data.items() if data['is_gallery']]
    
    def get_probe_subjects(self) -> List[str]:
        """Get list of probe subject IDs."""
        return [sid for sid, data in self.data.items() if data['is_probe']]
    
    def save_metadata(self, output_path: str):
        """Save metadata to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"\n💾 Metadata saved to: {output_path}")


def main():
    """Test data loader."""
    loader = OUGaitDataLoader("data/raw/OU-SimilarGaitActivities")
    data = loader.load_all_data(remove_unlabeled=True)
    
    # Save metadata
    from pathlib import Path
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    loader.save_metadata(output_dir / "data_loading_summary.json")


if __name__ == "__main__":
    main()
