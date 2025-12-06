"""
Fixed preprocessing pipeline with NO OVERLAP to prevent data leakage.
Saves to data/processed_no_overlap directory.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('src/preprocessing')

from data_loader import OUGaitDataLoader
from preprocessor import SensorPreprocessor, create_windows
from augmentation import balance_classes


class FixedPreprocessingPipeline:
    """Preprocessing pipeline with fixed overlap to prevent data leakage."""
    
    def __init__(self, config: dict):
        self.config = config
        self.loader = None
        self.preprocessor = None
        self.data = {}
        
    def run(self):
        """Run complete preprocessing pipeline."""
        print("\n" + "🔧" * 40)
        print("FIXED PREPROCESSING PIPELINE - NO OVERLAP")
        print("🔧" * 40)
        print("\n⚠️  This pipeline fixes data leakage by using 0% overlap")
        print("⚠️  Expected accuracy will be LOWER but more HONEST (85-90%)")
        
        # Step 1: Load data
        print("\n" + "=" * 80)
        print("STEP 1: LOADING DATA")
        print("=" * 80)
        self._load_data()
        
        # Step 2: Create windows (NO OVERLAP!)
        print("\n" + "=" * 80)
        print("STEP 2: CREATING WINDOWS (0% OVERLAP)")
        print("=" * 80)
        self._create_windows()
        
        # Step 3: Split data
        print("\n" + "=" * 80)
        print("STEP 3: TRAIN/VAL/TEST SPLIT")
        print("=" * 80)
        self._split_data()
        
        # Step 4: Preprocess
        print("\n" + "=" * 80)
        print("STEP 4: PREPROCESSING")
        print("=" * 80)
        self._preprocess_data()
        
        # Step 5: Augment training data
        print("\n" + "=" * 80)
        print("STEP 5: DATA AUGMENTATION")
        print("=" * 80)
        self._augment_data()
        
        # Step 6: Save processed data
        print("\n" + "=" * 80)
        print("STEP 6: SAVING PROCESSED DATA")
        print("=" * 80)
        self._save_data()
        
        print("\n" + "✅" * 40)
        print("FIXED PREPROCESSING COMPLETE!")
        print("✅" * 40)
        print("\n📊 Data saved to:", self.config['output_dir'])
        print("📊 Overlap:", self.config['overlap'])
        print("📊 Expected accuracy: 85-90% (honest, no leakage)")
    
    def _load_data(self):
        """Load raw data."""
        self.loader = OUGaitDataLoader(
            self.config['data_root'],
            sensor_position='Center'
        )
        raw_data = self.loader.load_all_data(
            remove_unlabeled=self.config['remove_unlabeled'],
            verbose=True
        )
        self.data['raw'] = raw_data
    
    def _create_windows(self):
        """Create fixed-length windows with NO OVERLAP."""
        window_size = self.config['window_size']
        overlap = self.config['overlap']
        
        all_windows = []
        all_labels = []
        all_subject_ids = []
        
        print(f"\n📊 Window size: {window_size} samples ({window_size/100:.1f}s)")
        print(f"📊 Overlap: {overlap*100:.0f}% ← FIXED (was 50%)")
        
        for subject_id, subject_data in tqdm(self.data['raw'].items(), desc="Creating windows"):
            sensor_data = subject_data['sensor_data']
            class_labels = subject_data['class_labels']
            
            # Create windows with specified overlap
            windows, window_labels = create_windows(
                sensor_data, class_labels, window_size, overlap
            )
            
            all_windows.append(windows)
            all_labels.append(window_labels)
            all_subject_ids.extend([subject_id] * len(windows))
        
        # Combine all windows
        self.data['windows'] = np.concatenate(all_windows, axis=0)
        self.data['labels'] = np.concatenate(all_labels, axis=0)
        self.data['subject_ids'] = np.array(all_subject_ids)
        
        print(f"\n✅ Total windows created: {len(self.data['windows']):,}")
        print(f"✅ Window shape: {self.data['windows'].shape}")
        
        # Print class distribution
        print(f"\n📈 Class distribution in windows:")
        unique, counts = np.unique(self.data['labels'], return_counts=True)
        for cls, cnt in zip(unique, counts):
            pct = cnt / len(self.data['labels']) * 100
            print(f"  Class {int(cls)}: {cnt:,} windows ({pct:.2f}%)")
    
    def _split_data(self):
        """Split data into train/val/test sets (subject-independent)."""
        gallery_subjects = self.loader.get_gallery_subjects()
        probe_subjects = self.loader.get_probe_subjects()
        
        # Get indices for gallery and probe
        gallery_mask = np.isin(self.data['subject_ids'], gallery_subjects)
        probe_mask = np.isin(self.data['subject_ids'], probe_subjects)
        
        # Gallery → Train + Val
        gallery_windows = self.data['windows'][gallery_mask]
        gallery_labels = self.data['labels'][gallery_mask]
        gallery_ids = self.data['subject_ids'][gallery_mask]
        
        # Split gallery by subjects (not by windows) - CORRECT!
        unique_gallery = np.unique(gallery_ids)
        train_subjects, val_subjects = train_test_split(
            unique_gallery,
            test_size=self.config['val_split'],
            random_state=self.config['random_seed'],
            stratify=None
        )
        
        train_mask = np.isin(gallery_ids, train_subjects)
        val_mask = np.isin(gallery_ids, val_subjects)
        
        # Create splits
        self.data['X_train'] = gallery_windows[train_mask]
        self.data['y_train'] = gallery_labels[train_mask]
        self.data['train_subjects'] = gallery_ids[train_mask]
        
        self.data['X_val'] = gallery_windows[val_mask]
        self.data['y_val'] = gallery_labels[val_mask]
        self.data['val_subjects'] = gallery_ids[val_mask]
        
        # Probe → Test
        self.data['X_test'] = self.data['windows'][probe_mask]
        self.data['y_test'] = self.data['labels'][probe_mask]
        self.data['test_subjects'] = self.data['subject_ids'][probe_mask]
        
        print(f"\n✅ Train set: {len(self.data['X_train']):,} windows from {len(train_subjects)} subjects")
        print(f"✅ Val set: {len(self.data['X_val']):,} windows from {len(val_subjects)} subjects")
        print(f"✅ Test set: {len(self.data['X_test']):,} windows from {len(probe_subjects)} subjects")
        
        # Print class distribution per split
        for split_name in ['train', 'val', 'test']:
            labels = self.data[f'y_{split_name}']
            print(f"\n{split_name.upper()} class distribution:")
            unique, counts = np.unique(labels, return_counts=True)
            for cls, cnt in zip(unique, counts):
                pct = cnt / len(labels) * 100
                print(f"  Class {int(cls)}: {cnt:,} ({pct:.2f}%)")
    
    def _preprocess_data(self):
        """Apply preprocessing transformations."""
        self.preprocessor = SensorPreprocessor(
            normalize=self.config['normalize'],
            filter_data=self.config['filter_data'],
            filter_cutoff=self.config['filter_cutoff'],
            sampling_rate=self.config['sampling_rate']
        )
        
        # Fit on training data
        print("\n📊 Fitting preprocessor on training data...")
        train_flat = self.data['X_train'].reshape(-1, self.data['X_train'].shape[-1])
        self.preprocessor.fit(train_flat)
        
        # Transform all splits
        print("📊 Transforming data...")
        for split in ['train', 'val', 'test']:
            X = self.data[f'X_{split}']
            original_shape = X.shape
            
            X_flat = X.reshape(-1, X.shape[-1])
            X_processed = self.preprocessor.transform(X_flat)
            
            self.data[f'X_{split}'] = X_processed.reshape(original_shape)
        
        print("✅ Preprocessing applied to all splits")
    
    def _augment_data(self):
        """Augment training data to balance classes."""
        if not self.config['augment']:
            print("⏭️  Skipping augmentation (disabled in config)")
            return
        
        minority_classes = self.config['minority_classes']
        augmentation_factor = self.config['augmentation_factor']
        
        print(f"\n📊 Augmenting minority classes: {minority_classes}")
        print(f"📊 Augmentation factor: {augmentation_factor}x")
        
        X_aug, y_aug = balance_classes(
            self.data['X_train'],
            self.data['y_train'],
            minority_classes=minority_classes,
            augmentation_factor=augmentation_factor,
            seed=self.config['random_seed']
        )
        
        print(f"\n✅ Training set before augmentation: {len(self.data['X_train']):,} windows")
        print(f"✅ Training set after augmentation: {len(X_aug):,} windows")
        
        self.data['X_train'] = X_aug
        self.data['y_train'] = y_aug
        
        print(f"\n📈 Class distribution after augmentation:")
        unique, counts = np.unique(y_aug, return_counts=True)
        for cls, cnt in zip(unique, counts):
            pct = cnt / len(y_aug) * 100
            print(f"  Class {int(cls)}: {cnt:,} ({pct:.2f}%)")
    
    def _save_data(self):
        """Save processed data to disk."""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train/val/test splits
        for split in ['train', 'val', 'test']:
            split_dir = output_dir / split
            split_dir.mkdir(exist_ok=True)
            
            np.save(split_dir / f'X_{split}.npy', self.data[f'X_{split}'])
            np.save(split_dir / f'y_{split}.npy', self.data[f'y_{split}'])
            
            metadata = {
                'num_samples': len(self.data[f'X_{split}']),
                'shape': self.data[f'X_{split}'].shape,
                'num_subjects': len(np.unique(self.data[f'{split}_subjects'])),
                'class_distribution': {
                    int(cls): int(cnt) 
                    for cls, cnt in zip(*np.unique(self.data[f'y_{split}'], return_counts=True))
                }
            }
            
            with open(split_dir / f'metadata_{split}.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Saved {split} split to {split_dir}")
        
        # Save preprocessing config
        config_path = output_dir / 'preprocessing_config.json'
        config_to_save = self.config.copy()
        config_to_save['preprocessor_params'] = self.preprocessor.get_params()
        config_to_save['fixed_for_leakage'] = True
        config_to_save['overlap_changed_from'] = 0.5
        
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        print(f"✅ Saved preprocessing config to {config_path}")
        
        # Save summary
        summary = {
            'total_subjects': self.loader.metadata['num_subjects'],
            'total_windows': len(self.data['windows']),
            'train_windows': len(self.data['X_train']),
            'val_windows': len(self.data['X_val']),
            'test_windows': len(self.data['X_test']),
            'window_size': self.config['window_size'],
            'overlap': self.config['overlap'],
            'overlap_note': 'Fixed from 50% to 0% to prevent data leakage',
            'augmentation_applied': self.config['augment']
        }
        
        summary_path = output_dir / 'preprocessing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Saved summary to {summary_path}")


def main():
    """Run fixed preprocessing pipeline."""
    config = {
        # Data paths
        'data_root': 'data/raw/OU-SimilarGaitActivities',
        'output_dir': 'data/processed_no_overlap',  # NEW DIRECTORY
        
        # Data loading
        'remove_unlabeled': True,
        
        # Windowing - FIXED!
        'window_size': 200,  # 2 seconds at 100 Hz
        'overlap': 0.0,  # 0% overlap (FIXED from 50%)
        
        # Preprocessing
        'normalize': True,
        'filter_data': False,
        'filter_cutoff': 20.0,
        'sampling_rate': 100.0,
        
        # Data split
        'val_split': 0.2,
        'random_seed': 42,
        
        # Augmentation
        'augment': True,
        'minority_classes': [1, 2],
        'augmentation_factor': 10
    }
    
    # Run pipeline
    pipeline = FixedPreprocessingPipeline(config)
    pipeline.run()
    
    print("\n" + "🎯" * 40)
    print("NEXT STEPS:")
    print("🎯" * 40)
    print("\n1. Retrain models using data from: data/processed_no_overlap/")
    print("2. Expected accuracy: 85-90% (lower but honest)")
    print("3. Training will be slower (gradual convergence)")
    print("4. This is CORRECT behavior - no data leakage!")


if __name__ == "__main__":
    main()
