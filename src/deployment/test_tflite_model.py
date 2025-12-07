"""
Test TensorFlow Lite model accuracy against original Keras model.

This script:
1. Loads original Keras model and TFLite model
2. Runs predictions on test dataset
3. Compares accuracy and predictions
4. Generates validation report
"""

import tensorflow as tf
import numpy as np
import json
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse


class TFLiteModel:
    """Wrapper for TensorFlow Lite model inference."""
    
    def __init__(self, model_path: str):
        """Load TFLite model."""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"✅ TFLite model loaded: {model_path}")
        print(f"   Input shape: {self.input_details[0]['shape']}")
        print(f"   Output shape: {self.output_details[0]['shape']}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference on batch of samples.
        
        Args:
            X: Input data (batch_size, window_size, n_features)
            
        Returns:
            Predictions (batch_size, num_classes)
        """
        predictions = []
        
        for sample in X:
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                np.expand_dims(sample, axis=0).astype(np.float32)
            )
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions.append(output[0])
        
        return np.array(predictions)


def compare_models(
    keras_model_path: str,
    tflite_model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list = None
) -> dict:
    """
    Compare Keras and TFLite model predictions.
    
    Args:
        keras_model_path: Path to .h5 model
        tflite_model_path: Path to .tflite model
        X_test: Test data
        y_test: Test labels
        class_names: Activity class names
        
    Returns:
        Comparison results dictionary
    """
    if class_names is None:
        class_names = ["Flat Walk", "Up Stairs", "Down Stairs", "Up Slope", "Down Slope"]
    
    print(f"\n{'='*60}")
    print(f"Model Comparison: Keras vs TFLite")
    print(f"{'='*60}\n")
    
    # Load Keras model
    print("Loading Keras model...")
    keras_model = tf.keras.models.load_model(keras_model_path)
    
    # Load TFLite model
    print("Loading TFLite model...")
    tflite_model = TFLiteModel(tflite_model_path)
    
    # Predictions
    print(f"\nRunning predictions on {len(X_test)} samples...")
    print("Keras model...")
    keras_preds = keras_model.predict(X_test, verbose=0)
    keras_pred_classes = np.argmax(keras_preds, axis=1)
    
    print("TFLite model...")
    tflite_preds = tflite_model.predict(X_test)
    tflite_pred_classes = np.argmax(tflite_preds, axis=1)
    
    # Calculate accuracies
    keras_accuracy = accuracy_score(y_test, keras_pred_classes)
    tflite_accuracy = accuracy_score(y_test, tflite_pred_classes)
    
    # Prediction agreement
    agreement = np.mean(keras_pred_classes == tflite_pred_classes)
    
    # Results
    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}\n")
    print(f"Keras Model Accuracy:  {keras_accuracy*100:.2f}%")
    print(f"TFLite Model Accuracy: {tflite_accuracy*100:.2f}%")
    print(f"Accuracy Difference:   {abs(keras_accuracy - tflite_accuracy)*100:.2f}%")
    print(f"Prediction Agreement:  {agreement*100:.2f}%")
    
    # Per-class comparison
    print(f"\n{'='*60}")
    print(f"Per-Class Accuracy")
    print(f"{'='*60}\n")
    
    keras_report = classification_report(y_test, keras_pred_classes, 
                                         target_names=class_names, 
                                         output_dict=True, zero_division=0)
    tflite_report = classification_report(y_test, tflite_pred_classes, 
                                          target_names=class_names, 
                                          output_dict=True, zero_division=0)
    
    print(f"{'Class':<15} {'Keras':<10} {'TFLite':<10} {'Diff':<10}")
    print("-" * 50)
    for class_name in class_names:
        keras_acc = keras_report[class_name]['recall'] * 100
        tflite_acc = tflite_report[class_name]['recall'] * 100
        diff = abs(keras_acc - tflite_acc)
        print(f"{class_name:<15} {keras_acc:>6.2f}%   {tflite_acc:>6.2f}%   {diff:>6.2f}%")
    
    # Check for significant differences
    max_diff = max([abs(keras_report[cn]['recall'] - tflite_report[cn]['recall']) 
                    for cn in class_names])
    
    print(f"\n{'='*60}")
    if max_diff < 0.02:  # Less than 2% difference
        print("✅ VALIDATION PASSED: Models are equivalent!")
    elif max_diff < 0.05:  # Less than 5% difference
        print("⚠️  VALIDATION WARNING: Small differences detected")
    else:
        print("❌ VALIDATION FAILED: Significant differences detected")
    print(f"{'='*60}\n")
    
    # Return results
    results = {
        'keras_accuracy': float(keras_accuracy),
        'tflite_accuracy': float(tflite_accuracy),
        'accuracy_difference': float(abs(keras_accuracy - tflite_accuracy)),
        'prediction_agreement': float(agreement),
        'max_class_difference': float(max_diff),
        'validation_passed': max_diff < 0.02,
        'per_class_keras': {cn: float(keras_report[cn]['recall']) for cn in class_names},
        'per_class_tflite': {cn: float(tflite_report[cn]['recall']) for cn in class_names}
    }
    
    return results


def main():
    """Main testing script."""
    parser = argparse.ArgumentParser(description='Test TFLite model accuracy')
    parser.add_argument('--keras-model', type=str, help='Path to .h5 model')
    parser.add_argument('--tflite-model', type=str, help='Path to .tflite model')
    parser.add_argument('--output', type=str, default='models/mobile/validation_report.json',
                        help='Output path for validation report')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Load test data
    print("Loading test data...")
    data_dir = project_root / 'data' / 'processed'
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    print(f"✅ Loaded {len(X_test)} test samples")
    
    # Find models if not specified
    if not args.keras_model or not args.tflite_model:
        print("\nAuto-detecting models...")
        
        # Find latest LSTM model
        results_dir = project_root / 'results'
        lstm_dirs = sorted(results_dir.glob('lstm_*'), reverse=True)
        
        if lstm_dirs:
            keras_model = lstm_dirs[0] / 'best_model.h5'
            tflite_model = project_root / 'models' / 'mobile' / 'gait_lstm_model.tflite'
            
            if keras_model.exists() and tflite_model.exists():
                args.keras_model = str(keras_model)
                args.tflite_model = str(tflite_model)
                print(f"✅ Found LSTM models")
    
    if not args.keras_model or not args.tflite_model:
        print("Error: Could not find models. Specify --keras-model and --tflite-model")
        return
    
    # Compare models
    results = compare_models(
        keras_model_path=args.keras_model,
        tflite_model_path=args.tflite_model,
        X_test=X_test,
        y_test=y_test
    )
    
    # Save validation report
    output_path = project_root / args.output
    os.makedirs(output_path.parent, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Validation report saved to: {output_path}")


if __name__ == '__main__':
    main()
