"""
LSTM (Long Short-Term Memory) model for gait-based activity classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple


def create_lstm(input_shape: Tuple[int, int] = (200, 6),
                num_classes: int = 5,
                lstm_units: list = [128, 64],
                dense_units: int = 64,
                dropout_rate: float = 0.3,
                recurrent_dropout: float = 0.2) -> keras.Model:
    model = models.Sequential(name='LSTM')
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # LSTM layers
    for i, units in enumerate(lstm_units[:-1]):
        model.add(layers.LSTM(
            units=units,
            return_sequences=True,  # Return sequences for stacked LSTM
            recurrent_dropout=recurrent_dropout,
            name=f'lstm_{i+1}'
        ))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_lstm_{i+1}'))
    
    # Last LSTM layer (no return_sequences)
    model.add(layers.LSTM(
        units=lstm_units[-1],
        return_sequences=False,
        recurrent_dropout=recurrent_dropout,
        name=f'lstm_{len(lstm_units)}'
    ))
    model.add(layers.Dropout(dropout_rate, name=f'dropout_lstm_{len(lstm_units)}'))
    
    # Dense layers
    model.add(layers.Dense(dense_units, activation='relu', name='dense_1'))
    model.add(layers.Dropout(dropout_rate, name='dropout_dense'))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model

def create_bidirectional_lstm(input_shape: Tuple[int, int] = (200, 6),
                               num_classes: int = 5) -> keras.Model:
    model = models.Sequential(name='BiLSTM')
    
    model.add(layers.Input(shape=input_shape))
    
    # Bidirectional LSTM layers
    model.add(layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, recurrent_dropout=0.2),
        name='bilstm_1'
    ))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, recurrent_dropout=0.2),
        name='bilstm_2'
    ))
    model.add(layers.Dropout(0.3))
    
    # Dense layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    # Output
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def create_gru(input_shape: Tuple[int, int] = (200, 6),
               num_classes: int = 5) -> keras.Model:
    model = models.Sequential(name='GRU')
    
    model.add(layers.Input(shape=input_shape))
    
    # GRU layers
    model.add(layers.GRU(128, return_sequences=True, recurrent_dropout=0.2))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.GRU(64, return_sequences=False, recurrent_dropout=0.2))
    model.add(layers.Dropout(0.3))
    
    # Dense layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    # Output
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def compile_model(model: keras.Model,
                  learning_rate: float = 0.001,
                  metrics: list = None) -> keras.Model:
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
    print("Creating LSTM models...\n")
    
    # Standard LSTM
    model = create_lstm()
    print("Standard LSTM:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}\n")
    
    # Bidirectional LSTM
    bilstm_model = create_bidirectional_lstm()
    print("\n" + "=" * 80)
    print("Bidirectional LSTM:")
    bilstm_model.summary()
    print(f"\nTotal parameters: {bilstm_model.count_params():,}\n")
    
    # GRU
    gru_model = create_gru()
    print("\n" + "=" * 80)
    print("GRU:")
    gru_model.summary()
    print(f"\nTotal parameters: {gru_model.count_params():,}\n")
    
    # Compile and test
    print("\n" + "=" * 80)
    print("Testing model compilation...")
    model = compile_model(model)
    print("Model compiled successfully!")
    
    # Test with dummy data
    import numpy as np
    X_dummy = np.random.randn(32, 200, 6)  # Batch of 32 samples
    y_pred = model.predict(X_dummy, verbose=0)
    print(f"\nPrediction test passed!")
    print(f"   Input shape: {X_dummy.shape}")
    print(f"   Output shape: {y_pred.shape}")
    print(f"   Output probabilities sum: {y_pred[0].sum():.4f} (should be ~1.0)")


if __name__ == "__main__":
    main()
