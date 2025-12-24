"""
Training script for CNN-LSTM hybrid model.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import GaitDataLoader
from cnn_lstm import create_cnn_lstm, create_simple_cnn_lstm, create_deep_cnn_lstm, compile_model


class CNNLSTMTrainer:
    """Train and evaluate CNN-LSTM model."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.history = None
        self.results_dir = None
        
        np.random.seed(config['random_seed'])
        tf.random.set_seed(config['random_seed'])
        
    def setup_results_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config['model_name']
        self.results_dir = Path(f"results/{model_name}_{timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Results will be saved to: {self.results_dir}")
        
        with open(self.results_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def create_callbacks(self):
        callbacks = []
        
        checkpoint_path = self.results_dir / 'best_model.h5'
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ))
        
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ))
        
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        ))
        
        log_dir = self.results_dir / 'logs'
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1
        ))
        
        csv_path = self.results_dir / 'training_history.csv'
        callbacks.append(keras.callbacks.CSVLogger(str(csv_path)))
        
        return callbacks
    
    def train(self, train_ds, val_ds, class_weights=None, callbacks=None):
        print("\n" + "=" * 80)
        print("TRAINING MODEL")
        print("=" * 80)
        
        variant = self.config['model_variant']
        if variant == 'simple':
            self.model = create_simple_cnn_lstm(
                input_shape=self.config['input_shape'],
                num_classes=self.config['num_classes']
            )
        elif variant == 'deep':
            self.model = create_deep_cnn_lstm(
                input_shape=self.config['input_shape'],
                num_classes=self.config['num_classes']
            )
        else:
            self.model = create_cnn_lstm(
                input_shape=self.config['input_shape'],
                num_classes=self.config['num_classes']
            )
        
        self.model = compile_model(
            self.model,
            learning_rate=self.config['learning_rate']
        )
        
        print("\n📊 Model Architecture:")
        self.model.summary()
        print(f"\n📊 Total parameters: {self.model.count_params():,}")
        
        callbacks = self.create_callbacks()
        
        print(f"\n🚀 Starting training for {self.config['epochs']} epochs...")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Learning rate: {self.config['learning_rate']}")
        
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config['epochs'],
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        print("\n✅ Training complete!")
        
        final_model_path = self.results_dir / 'final_model.h5'
        self.model.save(final_model_path)
        print(f"💾 Final model saved to: {final_model_path}")
        
        history_path = self.results_dir / 'history.json'
        with open(history_path, 'w') as f:
            history_dict = {k: [float(v) for v in vals] 
                          for k, vals in self.history.history.items()}
            json.dump(history_dict, f, indent=2)
    
    def evaluate(self, test_ds, X_test, y_test):
        print("\n" + "=" * 80)
        print("EVALUATING MODEL")
        print("=" * 80)
        
        test_loss, test_acc = self.model.evaluate(test_ds, verbose=1)
        print(f"\n📊 Test Loss: {test_loss:.4f}")
        print(f"📊 Test Accuracy: {test_acc:.4f}")
        
        y_pred_probs = self.model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        print("\n" + "=" * 80)
        print("CLASSIFICATION REPORT")
        print("=" * 80)
        
        class_names = [f'Class {i}' for i in range(self.config['num_classes'])]
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(report)
        
        report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        with open(self.results_dir / 'classification_report.json', 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, class_names)
        
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'classification_report': report_dict
        }
        
        with open(self.results_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def plot_confusion_matrix(self, cm, class_names):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = self.results_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n💾 Confusion matrix saved to: {cm_path}")
    
    def plot_training_history(self, test_accuracy=None):
        """Plot and save training history with optional test accuracy."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        
        # Add test accuracy as horizontal line
        if test_accuracy is not None:
            ax1.axhline(y=test_accuracy, color='red', linestyle='--', linewidth=2, 
                       label=f'Test ({test_accuracy:.4f})')
        
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Train', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        history_path = self.results_dir / 'training_history.png'
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Training history plot saved to: {history_path}")


def main():
    config = {
        'model_name': 'cnn_lstm',
        'data_dir': 'data/processed',
        'model_variant': 'standard',  # 'simple', 'standard', or 'deep'
        'input_shape': (200, 6),
        'num_classes': 5,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.001,
        'early_stopping_patience': 10,
        'reduce_lr_patience': 5,
        'random_seed': 42
    }
    
    print("\n" + "🚀" * 40)
    print("CNN-LSTM TRAINING PIPELINE")
    print("🚀" * 40)
    
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    loader = GaitDataLoader(config['data_dir'])
    data = loader.load_all()
    loader.print_summary()
    
    print("\n📊 Creating TensorFlow datasets...")
    train_ds = loader.create_tf_dataset('train', batch_size=config['batch_size'], shuffle=True)
    val_ds = loader.create_tf_dataset('val', batch_size=config['batch_size'], shuffle=False)
    test_ds = loader.create_tf_dataset('test', batch_size=config['batch_size'], shuffle=False)
    
    # class_weights = loader.get_class_weights('train')
    # print(f"\n📊 Class weights: {class_weights}")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6)
    ]
    
    trainer = CNNLSTMTrainer(config)
    trainer.setup_results_dir()
    
    trainer.train(train_ds, val_ds, callbacks=callbacks)
    
    X_test, y_test = data['test']
    results = trainer.evaluate(test_ds, X_test, y_test)
    
    # Plot training history with test accuracy
    trainer.plot_training_history(test_accuracy=results['test_accuracy'])
    
    print("\n" + "✅" * 40)
    print("TRAINING COMPLETE!")
    print("✅" * 40)
    print(f"\n📊 Final Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"📁 Results saved to: {trainer.results_dir}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
