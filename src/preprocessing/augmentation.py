"""
Data augmentation techniques for time-series sensor data.
Focuses on augmenting minority classes to address class imbalance.
"""

import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAugmenter:
    """Data augmentation for time-series IMU data."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize augmenter.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
    
    def jittering(self, data: np.ndarray, sigma: float = 0.05) -> np.ndarray:
        """
        Add random Gaussian noise to data.
        
        Args:
            data: Input data, shape (window_size, n_features)
            sigma: Standard deviation of noise
            
        Returns:
            Augmented data, same shape as input
        """
        noise = self.rng.normal(0, sigma, data.shape)
        return data + noise
    
    def scaling(self, data: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        Scale data by random factor.
        
        Args:
            data: Input data, shape (window_size, n_features)
            sigma: Standard deviation of scaling factor
            
        Returns:
            Augmented data, same shape as input
        """
        scale_factor = self.rng.normal(1.0, sigma, (1, data.shape[1]))
        return data * scale_factor
    
    def time_warping(self, data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """
        Apply time warping by resampling with random speed.
        
        Args:
            data: Input data, shape (window_size, n_features)
            sigma: Standard deviation of warping factor
            
        Returns:
            Augmented data, same shape as input
        """
        window_size = data.shape[0]
        
        # Generate random warping curve
        warp_steps = self.rng.normal(1.0, sigma, window_size)
        warp_steps = np.cumsum(warp_steps)
        warp_steps = warp_steps / warp_steps[-1] * (window_size - 1)
        
        # Interpolate
        warped = np.zeros_like(data)
        for i in range(data.shape[1]):
            warped[:, i] = np.interp(np.arange(window_size), warp_steps, data[:, i])
        
        return warped
    
    def rotation(self, data: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
        """
        Rotate sensor data (simulate sensor orientation change).
        
        Args:
            data: Input data with 6 channels [Gx, Gy, Gz, Ax, Ay, Az]
            max_angle: Maximum rotation angle in degrees
            
        Returns:
            Augmented data, same shape as input
        """
        # Random rotation angles for each axis
        angles = self.rng.uniform(-max_angle, max_angle, 3) * np.pi / 180
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ])
        
        Ry = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])
        
        Rz = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        
        # Apply rotation to gyroscope (first 3 channels)
        rotated = data.copy()
        rotated[:, :3] = (R @ data[:, :3].T).T
        
        # Apply rotation to accelerometer (last 3 channels)
        rotated[:, 3:] = (R @ data[:, 3:].T).T
        
        return rotated
    
    def augment(self, data: np.ndarray, methods: list = None) -> np.ndarray:
        """
        Apply multiple augmentation methods.
        
        Args:
            data: Input data, shape (window_size, n_features)
            methods: List of methods to apply. If None, randomly choose.
            
        Returns:
            Augmented data
        """
        if methods is None:
            # Randomly choose 1-2 methods
            all_methods = ['jittering', 'scaling', 'time_warping', 'rotation']
            n_methods = self.rng.randint(1, 3)
            methods = self.rng.choice(all_methods, n_methods, replace=False)
        
        augmented = data.copy()
        
        for method in methods:
            if method == 'jittering':
                augmented = self.jittering(augmented)
            elif method == 'scaling':
                augmented = self.scaling(augmented)
            elif method == 'time_warping':
                augmented = self.time_warping(augmented)
            elif method == 'rotation':
                augmented = self.rotation(augmented)
        
        return augmented
    
    def augment_batch(self, data: np.ndarray, labels: np.ndarray,
                      target_class: int, augmentation_factor: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment samples of a specific class.
        
        Args:
            data: Input data, shape (n_samples, window_size, n_features)
            labels: Labels, shape (n_samples,)
            target_class: Class to augment
            augmentation_factor: How many augmented samples per original
            
        Returns:
            Augmented data and labels
        """
        # Find samples of target class
        class_mask = labels == target_class
        class_samples = data[class_mask]
        
        if len(class_samples) == 0:
            return data, labels
        
        # Generate augmented samples
        augmented_samples = []
        for sample in class_samples:
            for _ in range(augmentation_factor):
                aug_sample = self.augment(sample)
                augmented_samples.append(aug_sample)
        
        # Combine with original data
        augmented_data = np.concatenate([data, np.array(augmented_samples)], axis=0)
        augmented_labels = np.concatenate([labels, np.full(len(augmented_samples), target_class)], axis=0)
        
        return augmented_data, augmented_labels


def balance_classes(data: np.ndarray, labels: np.ndarray,
                    minority_classes: list = [1, 2],
                    augmentation_factor: int = 10,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance classes by augmenting minority classes.
    
    Args:
        data: Input data, shape (n_samples, window_size, n_features)
        labels: Labels, shape (n_samples,)
        minority_classes: List of classes to augment
        augmentation_factor: Augmentation factor for minority classes
        seed: Random seed
        
    Returns:
        Balanced data and labels
    """
    augmenter = TimeSeriesAugmenter(seed=seed)
    
    balanced_data = data.copy()
    balanced_labels = labels.copy()
    
    for class_id in minority_classes:
        balanced_data, balanced_labels = augmenter.augment_batch(
            balanced_data, balanced_labels, class_id, augmentation_factor
        )
    
    # Shuffle
    rng = np.random.RandomState(seed)
    shuffle_idx = rng.permutation(len(balanced_data))
    balanced_data = balanced_data[shuffle_idx]
    balanced_labels = balanced_labels[shuffle_idx]
    
    return balanced_data, balanced_labels


def main():
    """Test augmentation functions."""
    # Create dummy data
    np.random.seed(42)
    window_size = 200
    n_features = 6
    n_samples = 100
    
    data = np.random.randn(n_samples, window_size, n_features)
    labels = np.random.randint(0, 5, n_samples)
    
    # Make class 1 and 2 minority
    labels[labels == 1] = np.random.choice([0, 3, 4], (labels == 1).sum())
    labels[labels == 2] = np.random.choice([0, 3, 4], (labels == 2).sum())
    labels[:5] = 1  # Add a few samples of class 1
    labels[5:10] = 2  # Add a few samples of class 2
    
    print("Original class distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples")
    
    # Test augmentation
    print("\nApplying augmentation...")
    augmenter = TimeSeriesAugmenter(seed=42)
    
    # Test individual methods
    sample = data[0]
    print(f"\nOriginal sample shape: {sample.shape}")
    
    jittered = augmenter.jittering(sample)
    print(f"Jittered sample shape: {jittered.shape}")
    
    scaled = augmenter.scaling(sample)
    print(f"Scaled sample shape: {scaled.shape}")
    
    warped = augmenter.time_warping(sample)
    print(f"Time-warped sample shape: {warped.shape}")
    
    rotated = augmenter.rotation(sample)
    print(f"Rotated sample shape: {rotated.shape}")
    
    # Test batch augmentation
    balanced_data, balanced_labels = balance_classes(
        data, labels, minority_classes=[1, 2], augmentation_factor=10
    )
    
    print(f"\nBalanced class distribution:")
    unique, counts = np.unique(balanced_labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples")
    
    print(f"\nOriginal data shape: {data.shape}")
    print(f"Balanced data shape: {balanced_data.shape}")
    
    print("\n✅ Augmentation tests passed!")


if __name__ == "__main__":
    main()
