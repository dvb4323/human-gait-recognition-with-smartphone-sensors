"""
Data loader for preprocessed gait recognition data.
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict
import tensorflow as tf


class GaitDataLoader:
    """Load preprocessed data for model training."""
    
    def __init__(self, data_dir: str = 'data/processed'):       
        self.data_dir = Path(data_dir)
        self.data = {}
        self.metadata = {}
        
    def load_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        split_dir = self.data_dir / split
        
        X = np.load(split_dir / f'X_{split}.npy')
        y = np.load(split_dir / f'y_{split}.npy')
        if (split_dir / f'{split}_subjects.npy').exists():
            subjects = np.load(split_dir / f'{split}_subjects.npy')
            self.data[f'{split}_subjects'] = subjects
        # Load metadata
        with open(split_dir / f'metadata_{split}.json', 'r') as f:
            self.metadata[split] = json.load(f)
        
        print(f"Loaded {split} split:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Classes: {np.unique(y)}")
        
        return X, y
    
    def load_all(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        self.data['train'] = self.load_split('train')
        self.data['val'] = self.load_split('val')
        self.data['test'] = self.load_split('test')
        
        return self.data
    
    def create_tf_dataset(self, split: str, batch_size: int = 32, 
                         shuffle: bool = True) -> tf.data.Dataset:
        X, y = self.data[split]
        
        dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int64)))
        
        if shuffle:
            buffer_size = len(X)
            dataset = dataset.shuffle(buffer_size=buffer_size, seed=42)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_class_weights(self, split: str = 'train') -> Dict[int, float]:
        _, y = self.data[split]
        
        # Count samples per class
        unique, counts = np.unique(y, return_counts=True)
        
        # Calculate weights (inverse frequency)
        total = len(y)
        weights = {int(cls): total / (len(unique) * count) 
                  for cls, count in zip(unique, counts)}
        
        return weights
    
    def print_summary(self):
        """Print data summary."""
        print("\n" + "=" * 80)
        print("DATA SUMMARY")
        print("=" * 80)
        
        for split in ['train', 'val', 'test']:
            if split in self.data:
                X, y = self.data[split]
                print(f"\n{split.upper()}:")
                print(f"  Samples: {len(X):,}")
                print(f"  Shape: {X.shape}")
                
                # Class distribution
                unique, counts = np.unique(y, return_counts=True)
                print(f"  Class distribution:")
                for cls, cnt in zip(unique, counts):
                    pct = cnt / len(y) * 100
                    print(f"    Class {int(cls)}: {cnt:,} ({pct:.2f}%)")


def main():
    """Test data loader."""
    loader = GaitDataLoader('data/processed')
    data = loader.load_all()
    loader.print_summary()
    
    # Test TensorFlow dataset creation
    print("\nCreating TensorFlow datasets...")
    train_ds = loader.create_tf_dataset('train', batch_size=64, shuffle=True)
    val_ds = loader.create_tf_dataset('val', batch_size=64, shuffle=False)
    test_ds = loader.create_tf_dataset('test', batch_size=64, shuffle=False)
    
    print(f"Train dataset: {train_ds}")
    print(f"Val dataset: {val_ds}")
    print(f"Test dataset: {test_ds}")
    
    # Calculate class weights
    class_weights = loader.get_class_weights('train')
    print(f"\nClass weights: {class_weights}")


if __name__ == "__main__":
    main()
