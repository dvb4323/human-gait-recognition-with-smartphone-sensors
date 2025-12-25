import numpy as np
from scipy import signal
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SensorPreprocessor:   
    def __init__(self, normalize: bool = True, filter_data: bool = False,
                 filter_cutoff: float = 20.0, sampling_rate: float = 100.0):
        self.normalize = normalize
        self.filter_data = filter_data
        self.filter_cutoff = filter_cutoff
        self.sampling_rate = sampling_rate
        
        # Statistics for normalization (computed from training data)
        self.mean = None
        self.std = None
        self.is_fitted = False
        
    def fit(self, data: np.ndarray):
        if self.normalize:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            # Avoid division by zero
            self.std = np.where(self.std == 0, 1.0, self.std)
            self.is_fitted = True
    
    def transform(self, data: np.ndarray) -> np.ndarray:
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
        self.fit(data)
        return self.transform(data)
    
    def _apply_filter(self, data: np.ndarray) -> np.ndarray:
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
    n_samples, n_features = data.shape
    step_size = int(window_size * (1 - overlap))
    
    windows = []
    window_labels = []
    
    for start_idx in range(0, n_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window_label_seq = labels[start_idx:end_idx]
        
        unique, counts = np.unique(window_label_seq, return_counts=True)
        
        # Extract window
        window = data[start_idx:end_idx, :]
        
        # Majority vote for window label
        majority_label = unique[np.argmax(counts)] 
        if majority_label != -1:      
            windows.append(window)
            window_labels.append(majority_label)
    
    return np.array(windows), np.array(window_labels)


def pad_sequence(data: np.ndarray, target_length: int, mode: str = 'edge') -> np.ndarray:
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
    
    print("\nPreprocessing tests passed!")


if __name__ == "__main__":
    main()
