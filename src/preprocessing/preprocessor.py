"""
Preprocessing functions for sensor data.
Includes normalization, filtering, and data transformations.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SensorPreprocessor:
    """Preprocessing utilities for IMU sensor data."""
    
    def __init__(self, normalize: bool = True, filter_data: bool = False,
                 filter_cutoff: float = 20.0, sampling_rate: float = 100.0):
        """
        Initialize preprocessor.
        
        Args:
            normalize: Apply Z-score normalization
            filter_data: Apply low-pass Butterworth filter
            filter_cutoff: Cutoff frequency for filter (Hz)
            sampling_rate: Sampling rate (Hz)
        """
        self.normalize = normalize
        self.filter_data = filter_data
        self.filter_cutoff = filter_cutoff
        self.sampling_rate = sampling_rate
        
        # Statistics for normalization (computed from training data)
        self.mean = None
        self.std = None
        self.is_fitted = False
        
    def fit(self, data: np.ndarray):
        """
        Fit normalization parameters on training data.
        
        Args:
            data: Training data, shape (n_samples, n_features)
        """
        if self.normalize:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            # Avoid division by zero
            self.std = np.where(self.std == 0, 1.0, self.std)
            self.is_fitted = True
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing transformations.
        
        Args:
            data: Input data, shape (n_samples, n_features)
            
        Returns:
            Preprocessed data, same shape as input
        """
        processed = data.copy()
        
        # Apply filtering first (before normalization)
        if self.filter_data:
            processed = self._apply_filter(processed)
        
        # Apply normalization
        if self.normalize:
            if not self.is_fitted:
                raise ValueError("Preprocessor not fitted. Call fit() first.")
            processed = (processed - self.mean) / self.std
        
        return processed
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
    
    def _apply_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply low-pass Butterworth filter.
        
        Args:
            data: Input data, shape (n_samples, n_features)
            
        Returns:
            Filtered data
        """
        # Design Butterworth filter
        nyquist = self.sampling_rate / 2.0
        normal_cutoff = self.filter_cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        
        # Apply filter to each channel
        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered
    
    def get_params(self) -> dict:
        """Get preprocessing parameters."""
        return {
            'normalize': self.normalize,
            'filter_data': self.filter_data,
            'filter_cutoff': self.filter_cutoff,
            'sampling_rate': self.sampling_rate,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'is_fitted': self.is_fitted
        }


def create_windows(data: np.ndarray, labels: np.ndarray, 
                   window_size: int, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create fixed-length windows from time-series data.
    
    Args:
        data: Input data, shape (n_samples, n_features)
        labels: Labels for each sample, shape (n_samples,)
        window_size: Number of samples per window
        overlap: Overlap ratio (0.0 to 1.0)
        
    Returns:
        windows: Shape (n_windows, window_size, n_features)
        window_labels: Shape (n_windows,) - majority vote per window
    """
    n_samples, n_features = data.shape
    step_size = int(window_size * (1 - overlap))
    
    windows = []
    window_labels = []
    
    for start_idx in range(0, n_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # Extract window
        window = data[start_idx:end_idx, :]
        window_label_seq = labels[start_idx:end_idx]
        
        # Majority vote for window label
        unique, counts = np.unique(window_label_seq, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        
        windows.append(window)
        window_labels.append(majority_label)
    
    return np.array(windows), np.array(window_labels)


def pad_sequence(data: np.ndarray, target_length: int, mode: str = 'edge') -> np.ndarray:
    """
    Pad sequence to target length.
    
    Args:
        data: Input data, shape (n_samples, n_features)
        target_length: Desired length
        mode: Padding mode ('edge', 'constant', 'reflect')
        
    Returns:
        Padded data, shape (target_length, n_features)
    """
    if len(data) >= target_length:
        return data[:target_length]
    
    pad_width = target_length - len(data)
    
    if mode == 'edge':
        # Repeat edge values
        return np.pad(data, ((0, pad_width), (0, 0)), mode='edge')
    elif mode == 'constant':
        # Pad with zeros
        return np.pad(data, ((0, pad_width), (0, 0)), mode='constant')
    elif mode == 'reflect':
        # Mirror reflection
        return np.pad(data, ((0, pad_width), (0, 0)), mode='reflect')
    else:
        raise ValueError(f"Unknown padding mode: {mode}")


def calculate_sensor_magnitude(data: np.ndarray, sensor_type: str = 'accel') -> np.ndarray:
    """
    Calculate magnitude of 3-axis sensor data.
    
    Args:
        data: Input data with 6 channels [Gx, Gy, Gz, Ax, Ay, Az]
        sensor_type: 'accel' or 'gyro'
        
    Returns:
        Magnitude values, shape (n_samples,)
    """
    if sensor_type == 'accel':
        # Accelerometer: columns 3, 4, 5
        mag = np.sqrt(data[:, 3]**2 + data[:, 4]**2 + data[:, 5]**2)
    elif sensor_type == 'gyro':
        # Gyroscope: columns 0, 1, 2
        mag = np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2)
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")
    
    return mag


def main():
    """Test preprocessing functions."""
    # Create dummy data
    np.random.seed(42)
    data = np.random.randn(1000, 6)  # 1000 samples, 6 features
    labels = np.random.randint(0, 5, 1000)  # 5 classes
    
    # Test preprocessor
    print("Testing SensorPreprocessor...")
    preprocessor = SensorPreprocessor(normalize=True, filter_data=True)
    preprocessor.fit(data)
    processed = preprocessor.transform(data)
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {processed.shape}")
    print(f"Output mean: {processed.mean(axis=0)}")
    print(f"Output std: {processed.std(axis=0)}")
    
    # Test windowing
    print("\nTesting create_windows...")
    window_size = 200  # 2 seconds at 100 Hz
    windows, window_labels = create_windows(processed, labels, window_size, overlap=0.5)
    print(f"Number of windows: {len(windows)}")
    print(f"Window shape: {windows.shape}")
    print(f"Labels shape: {window_labels.shape}")
    
    print("\n✅ Preprocessing tests passed!")


if __name__ == "__main__":
    main()
