"""
1D Convolutional Neural Network for gait-based activity classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple


def create_1d_cnn(input_shape: Tuple[int, int] = (200, 6),
                  num_classes: int = 5,
                  filters: list = [64, 128, 256],
                  kernel_sizes: list = [5, 5, 3],
                  dense_units: int = 128,
                  dropout_rate: float = 0.5) -> keras.Model:
    """
    Create 1D CNN model for time-series classification.
    
    Args:
        input_shape: Input shape (window_size, n_features)
        num_classes: Number of output classes
        filters: Number of filters for each conv layer
        kernel_sizes: Kernel sizes for each conv layer
        dense_units: Number of units in dense layer
        dropout_rate: Dropout rate
        
    Returns:
        Keras model
    """
    model = models.Sequential(name='1D_CNN')
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # Convolutional blocks
    for i, (n_filters, kernel_size) in enumerate(zip(filters, kernel_sizes)):
        # Conv1D + BatchNorm + ReLU
        model.add(layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding='same',
            name=f'conv1d_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
        model.add(layers.Activation('relu', name=f'relu_{i+1}'))
        
        # MaxPooling
        model.add(layers.MaxPooling1D(
            pool_size=2,
            name=f'maxpool_{i+1}'
        ))
    
    # Global pooling
    model.add(layers.GlobalAveragePooling1D(name='global_avg_pool'))
    
    # Dense layers
    model.add(layers.Dense(dense_units, name='dense_1'))
    model.add(layers.Activation('relu', name='relu_dense'))
    model.add(layers.Dropout(dropout_rate, name='dropout'))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model


def create_simple_1d_cnn(input_shape: Tuple[int, int] = (200, 6),
                         num_classes: int = 5) -> keras.Model:
    """
    Create a simpler 1D CNN model (faster training, good baseline).
    
    Args:
        input_shape: Input shape (window_size, n_features)
        num_classes: Number of output classes
        
    Returns:
        Keras model
    """
    model = models.Sequential(name='Simple_1D_CNN')
    
    model.add(layers.Input(shape=input_shape))
    
    # Block 1
    model.add(layers.Conv1D(64, 5, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(2))
    
    # Block 2
    model.add(layers.Conv1D(128, 5, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(2))
    
    # Global pooling and output
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def create_deep_1d_cnn(input_shape: Tuple[int, int] = (200, 6),
                       num_classes: int = 5) -> keras.Model:
    """
    Create a deeper 1D CNN model (more capacity, may need more data).
    
    Args:
        input_shape: Input shape (window_size, n_features)
        num_classes: Number of output classes
        
    Returns:
        Keras model
    """
    model = models.Sequential(name='Deep_1D_CNN')
    
    model.add(layers.Input(shape=input_shape))
    
    # Block 1
    model.add(layers.Conv1D(64, 7, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv1D(64, 7, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(2))
    
    # Block 2
    model.add(layers.Conv1D(128, 5, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv1D(128, 5, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(2))
    
    # Block 3
    model.add(layers.Conv1D(256, 3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv1D(256, 3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(2))
    
    # Global pooling and output
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def compile_model(model: keras.Model,
                  learning_rate: float = 0.001,
                  metrics: list = None) -> keras.Model:
    """
    Compile model with optimizer and loss.
    
    Args:
        model: Keras model
        learning_rate: Learning rate for Adam optimizer
        metrics: List of metrics to track
        
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = ['accuracy']
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )
    
    return model


def main():
    """Test model creation."""
    print("Creating 1D CNN models...\n")
    
    # Standard model
    model = create_1d_cnn()
    print("Standard 1D CNN:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}\n")
    
    # Simple model
    simple_model = create_simple_1d_cnn()
    print("\n" + "=" * 80)
    print("Simple 1D CNN:")
    simple_model.summary()
    print(f"\nTotal parameters: {simple_model.count_params():,}\n")
    
    # Deep model
    deep_model = create_deep_1d_cnn()
    print("\n" + "=" * 80)
    print("Deep 1D CNN:")
    deep_model.summary()
    print(f"\nTotal parameters: {deep_model.count_params():,}\n")
    
    # Compile and test
    print("\n" + "=" * 80)
    print("Testing model compilation...")
    model = compile_model(model)
    print("✅ Model compiled successfully!")
    
    # Test with dummy data
    import numpy as np
    X_dummy = np.random.randn(32, 200, 6)  # Batch of 32 samples
    y_pred = model.predict(X_dummy, verbose=0)
    print(f"\n✅ Prediction test passed!")
    print(f"   Input shape: {X_dummy.shape}")
    print(f"   Output shape: {y_pred.shape}")
    print(f"   Output probabilities sum: {y_pred[0].sum():.4f} (should be ~1.0)")


if __name__ == "__main__":
    main()
