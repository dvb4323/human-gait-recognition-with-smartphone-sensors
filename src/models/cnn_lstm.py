"""
CNN-LSTM Hybrid model for gait-based activity classification.
Combines CNN's feature extraction with LSTM's temporal modeling.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple


def create_cnn_lstm(input_shape: Tuple[int, int] = (200, 6),
                    num_classes: int = 5,
                    cnn_filters: list = [64, 128],
                    cnn_kernels: list = [5, 5],
                    lstm_units: int = 128,
                    dense_units: int = 64,
                    dropout_rate: float = 0.5) -> keras.Model:
    """
    Create CNN-LSTM hybrid model.
    
    CNN layers extract spatial features, then LSTM models temporal dependencies.
    
    Args:
        input_shape: Input shape (window_size, n_features)
        num_classes: Number of output classes
        cnn_filters: Number of filters for each CNN layer
        cnn_kernels: Kernel sizes for each CNN layer
        lstm_units: Number of LSTM units
        dense_units: Number of units in dense layer
        dropout_rate: Dropout rate
        
    Returns:
        Keras model
    """
    model = models.Sequential(name='CNN_LSTM')
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # CNN feature extraction blocks
    for i, (n_filters, kernel_size) in enumerate(zip(cnn_filters, cnn_kernels)):
        model.add(layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding='same',
            name=f'conv1d_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
        model.add(layers.Activation('relu', name=f'relu_{i+1}'))
        model.add(layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}'))
    
    # LSTM temporal modeling
    model.add(layers.LSTM(
        units=lstm_units,
        return_sequences=False,
        recurrent_dropout=0.2,
        name='lstm'
    ))
    model.add(layers.Dropout(dropout_rate, name='dropout_lstm'))
    
    # Dense layers
    model.add(layers.Dense(dense_units, activation='relu', name='dense_1'))
    model.add(layers.Dropout(dropout_rate, name='dropout_dense'))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model


def create_simple_cnn_lstm(input_shape: Tuple[int, int] = (200, 6),
                           num_classes: int = 5) -> keras.Model:
    """
    Create a simpler CNN-LSTM model (faster training).
    
    Args:
        input_shape: Input shape (window_size, n_features)
        num_classes: Number of output classes
        
    Returns:
        Keras model
    """
    model = models.Sequential(name='Simple_CNN_LSTM')
    
    model.add(layers.Input(shape=input_shape))
    
    # Single CNN block
    model.add(layers.Conv1D(64, 5, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(2))
    
    # LSTM
    model.add(layers.LSTM(128, return_sequences=False, recurrent_dropout=0.2))
    model.add(layers.Dropout(0.5))
    
    # Output
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def create_deep_cnn_lstm(input_shape: Tuple[int, int] = (200, 6),
                         num_classes: int = 5) -> keras.Model:
    """
    Create a deeper CNN-LSTM model (more capacity).
    
    Args:
        input_shape: Input shape (window_size, n_features)
        num_classes: Number of output classes
        
    Returns:
        Keras model
    """
    model = models.Sequential(name='Deep_CNN_LSTM')
    
    model.add(layers.Input(shape=input_shape))
    
    # CNN blocks
    model.add(layers.Conv1D(64, 7, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.Conv1D(128, 5, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.Conv1D(256, 3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(2))
    
    # Stacked LSTM
    model.add(layers.LSTM(128, return_sequences=True, recurrent_dropout=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(64, return_sequences=False, recurrent_dropout=0.2))
    model.add(layers.Dropout(0.5))
    
    # Dense layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
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
    print("Creating CNN-LSTM models...\n")
    
    # Standard CNN-LSTM
    model = create_cnn_lstm()
    print("Standard CNN-LSTM:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}\n")
    
    # Simple CNN-LSTM
    simple_model = create_simple_cnn_lstm()
    print("\n" + "=" * 80)
    print("Simple CNN-LSTM:")
    simple_model.summary()
    print(f"\nTotal parameters: {simple_model.count_params():,}\n")
    
    # Deep CNN-LSTM
    deep_model = create_deep_cnn_lstm()
    print("\n" + "=" * 80)
    print("Deep CNN-LSTM:")
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
