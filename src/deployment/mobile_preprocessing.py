"""
Mobile-friendly preprocessing functions for gait recognition.

This module contains preprocessing logic that can be easily ported to JavaScript/TypeScript
for use in React Native apps.
"""

import numpy as np
from typing import Tuple, List
import json


def normalize_sensor_data(data: np.ndarray, method: str = 'z-score') -> np.ndarray:
    """
    Normalize sensor data.
    
    Args:
        data: Sensor data (n_samples, n_features) or (window_size, n_features)
        method: Normalization method ('z-score', 'min-max', 'none')
        
    Returns:
        Normalized data
    """
    if method == 'z-score':
        # Z-score normalization (mean=0, std=1)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        normalized = (data - mean) / std
        
    elif method == 'min-max':
        # Min-max normalization (0 to 1)
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        normalized = (data - min_val) / range_val
        
    else:
        # No normalization
        normalized = data
    
    return normalized


def create_window(buffer: np.ndarray, window_size: int = 200) -> np.ndarray:
    """
    Extract window from buffer.
    
    Args:
        buffer: Sensor data buffer (>= window_size, n_features)
        window_size: Number of samples in window (default: 200 = 2 seconds at 100 Hz)
        
    Returns:
        Window data (window_size, n_features)
    """
    if len(buffer) < window_size:
        raise ValueError(f"Buffer too small: {len(buffer)} < {window_size}")
    
    # Take most recent window_size samples
    window = buffer[-window_size:]
    return window


def resample_data(data: np.ndarray, from_hz: int, to_hz: int = 100) -> np.ndarray:
    """
    Resample sensor data to target frequency.
    
    Args:
        data: Sensor data (n_samples, n_features)
        from_hz: Original sampling rate
        to_hz: Target sampling rate (default: 100 Hz)
        
    Returns:
        Resampled data
    """
    if from_hz == to_hz:
        return data
    
    n_samples = len(data)
    n_features = data.shape[1]
    
    # Calculate new number of samples
    new_n_samples = int(n_samples * to_hz / from_hz)
    
    # Create new time indices
    old_indices = np.linspace(0, n_samples - 1, n_samples)
    new_indices = np.linspace(0, n_samples - 1, new_n_samples)
    
    # Interpolate each feature
    resampled = np.zeros((new_n_samples, n_features))
    for i in range(n_features):
        resampled[:, i] = np.interp(new_indices, old_indices, data[:, i])
    
    return resampled


def preprocess_for_inference(
    raw_data: np.ndarray,
    window_size: int = 200,
    normalization: str = 'z-score',
    sampling_rate: int = 100,
    target_rate: int = 100
) -> np.ndarray:
    """
    Complete preprocessing pipeline for inference.
    
    Args:
        raw_data: Raw sensor data (n_samples, 6) - [Gx, Gy, Gz, Ax, Ay, Az]
        window_size: Window size in samples
        normalization: Normalization method
        sampling_rate: Original sampling rate
        target_rate: Target sampling rate
        
    Returns:
        Preprocessed window ready for model inference (1, window_size, 6)
    """
    # Resample if needed
    if sampling_rate != target_rate:
        data = resample_data(raw_data, sampling_rate, target_rate)
    else:
        data = raw_data
    
    # Extract window
    if len(data) >= window_size:
        window = create_window(data, window_size)
    else:
        # Pad if too short
        padding = np.zeros((window_size - len(data), data.shape[1]))
        window = np.vstack([padding, data])
    
    # Normalize
    normalized = normalize_sensor_data(window, method=normalization)
    
    # Add batch dimension
    batch = np.expand_dims(normalized, axis=0)
    
    return batch.astype(np.float32)


class SlidingWindowBuffer:
    """
    Sliding window buffer for real-time inference.
    
    This class maintains a buffer of sensor samples and provides windows
    for inference with configurable overlap.
    """
    
    def __init__(
        self,
        window_size: int = 200,
        overlap: float = 0.5,
        n_features: int = 6
    ):
        """
        Initialize buffer.
        
        Args:
            window_size: Number of samples per window
            overlap: Overlap ratio (0.0 to 1.0)
            n_features: Number of sensor features
        """
        self.window_size = window_size
        self.overlap = overlap
        self.n_features = n_features
        self.step_size = int(window_size * (1 - overlap))
        
        # Initialize buffer
        self.buffer = np.zeros((0, n_features))
        self.samples_since_last_window = 0
    
    def add_sample(self, sample: np.ndarray) -> bool:
        """
        Add new sample to buffer.
        
        Args:
            sample: Sensor sample (n_features,)
            
        Returns:
            True if ready for inference, False otherwise
        """
        # Add sample to buffer
        self.buffer = np.vstack([self.buffer, sample.reshape(1, -1)])
        self.samples_since_last_window += 1
        
        # Check if ready for inference
        if len(self.buffer) >= self.window_size and \
           self.samples_since_last_window >= self.step_size:
            return True
        
        return False
    
    def get_window(self) -> np.ndarray:
        """
        Get current window for inference.
        
        Returns:
            Window data (window_size, n_features)
        """
        if len(self.buffer) < self.window_size:
            raise ValueError("Buffer not ready for window extraction")
        
        # Extract window
        window = self.buffer[-self.window_size:]
        
        # Reset counter
        self.samples_since_last_window = 0
        
        # Trim buffer to prevent unlimited growth
        max_buffer_size = self.window_size + self.step_size
        if len(self.buffer) > max_buffer_size:
            self.buffer = self.buffer[-max_buffer_size:]
        
        return window
    
    def reset(self):
        """Reset buffer."""
        self.buffer = np.zeros((0, self.n_features))
        self.samples_since_last_window = 0


def export_preprocessing_params(output_path: str):
    """
    Export preprocessing parameters to JSON for mobile app.
    
    Args:
        output_path: Path to save JSON file
    """
    params = {
        'window_size': 200,
        'sampling_rate': 100,
        'overlap': 0.5,
        'normalization': 'z-score',
        'features': ['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az'],
        'feature_order': 'gyroscope_xyz_then_accelerometer_xyz',
        'units': {
            'gyroscope': 'rad/s or deg/s',
            'accelerometer': 'g (gravity)'
        },
        'notes': {
            'window_duration': '2 seconds at 100 Hz',
            'step_size': 100,
            'step_duration': '1 second (50% overlap)'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"✅ Preprocessing parameters saved to: {output_path}")


# JavaScript/TypeScript equivalent pseudocode for reference
"""
// TypeScript equivalent for React Native

interface SensorSample {
  Gx: number;  // Gyroscope X
  Gy: number;  // Gyroscope Y
  Gz: number;  // Gyroscope Z
  Ax: number;  // Accelerometer X
  Ay: number;  // Accelerometer Y
  Az: number;  // Accelerometer Z
}

function normalizeSensorData(data: number[][]): number[][] {
  const mean = calculateMean(data);
  const std = calculateStd(data);
  
  return data.map((sample, i) => 
    sample.map((value, j) => (value - mean[j]) / (std[j] || 1))
  );
}

class SlidingWindowBuffer {
  private buffer: number[][] = [];
  private windowSize: number = 200;
  private stepSize: number = 100;
  private samplesSinceLastWindow: number = 0;
  
  addSample(sample: SensorSample): boolean {
    const array = [sample.Gx, sample.Gy, sample.Gz, sample.Ax, sample.Ay, sample.Az];
    this.buffer.push(array);
    this.samplesSinceLastWindow++;
    
    return this.buffer.length >= this.windowSize && 
           this.samplesSinceLastWindow >= this.stepSize;
  }
  
  getWindow(): number[][] {
    const window = this.buffer.slice(-this.windowSize);
    this.samplesSinceLastWindow = 0;
    return window;
  }
}
"""


if __name__ == '__main__':
    # Test preprocessing functions
    print("Testing preprocessing functions...\n")
    
    # Create dummy sensor data
    n_samples = 250
    n_features = 6
    dummy_data = np.random.randn(n_samples, n_features)
    
    print(f"Input data shape: {dummy_data.shape}")
    
    # Test normalization
    normalized = normalize_sensor_data(dummy_data)
    print(f"✅ Normalized data: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    
    # Test window extraction
    window = create_window(dummy_data, window_size=200)
    print(f"✅ Window extracted: shape={window.shape}")
    
    # Test resampling
    resampled = resample_data(dummy_data, from_hz=100, to_hz=50)
    print(f"✅ Resampled: {len(dummy_data)} samples @ 100Hz → {len(resampled)} samples @ 50Hz")
    
    # Test full pipeline
    preprocessed = preprocess_for_inference(dummy_data)
    print(f"✅ Preprocessed for inference: shape={preprocessed.shape}")
    
    # Test sliding window buffer
    print("\nTesting SlidingWindowBuffer...")
    buffer = SlidingWindowBuffer(window_size=200, overlap=0.5)
    
    ready_count = 0
    for i in range(300):
        sample = np.random.randn(6)
        if buffer.add_sample(sample):
            window = buffer.get_window()
            ready_count += 1
    
    print(f"✅ Buffer processed 300 samples, {ready_count} windows ready")
    
    # Export parameters
    export_preprocessing_params('preprocessing_params.json')
    
    print("\n✅ All tests passed!")
