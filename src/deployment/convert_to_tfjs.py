"""
Convert Keras models to TensorFlow.js format.
This script works around NumPy compatibility issues with tensorflowjs_converter.
"""

import tensorflow as tf
import tensorflowjs as tfjs
import os
from pathlib import Path
import json


def convert_keras_to_tfjs(keras_model_path: str, output_dir: str, model_name: str):
    """
    Convert Keras .h5 model to TensorFlow.js format.
    
    Args:
        keras_model_path: Path to .h5 model file
        output_dir: Output directory for TensorFlow.js model
        model_name: Name of the model (for display)
    """
    print(f"\n{'='*60}")
    print(f"Converting {model_name} model")
    print(f"{'='*60}\n")
    
    try:
        # Load Keras model
        print(f"Loading Keras model from: {keras_model_path}")
        model = tf.keras.models.load_model(keras_model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Parameters: {model.count_params():,}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to TensorFlow.js
        print(f"\nConverting to TensorFlow.js format...")
        tfjs.converters.save_keras_model(model, output_dir)
        
        print(f"✅ Conversion successful!")
        print(f"   Output directory: {output_dir}")
        
        # List generated files
        files = os.listdir(output_dir)
        print(f"\n   Generated files:")
        for file in files:
            file_path = os.path.join(output_dir, file)
            size_kb = os.path.getsize(file_path) / 1024
            print(f"   - {file} ({size_kb:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting {model_name}: {e}")
        return False


def main():
    """Convert all models."""
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Define models to convert
    models = [
        {
            'name': 'GRU',
            'keras_path': project_root / 'results' / 'lstm_20251206_170855' / 'best_model.h5',
            'output_dir': project_root / 'mobile-app' / 'src' / 'assets' / 'models' / 'gru'
        },
        {
            'name': '1D CNN',
            'keras_path': project_root / 'results' / '1d_cnn_20251206_154352' / 'best_model.h5',
            'output_dir': project_root / 'mobile-app' / 'src' / 'assets' / 'models' / 'cnn'
        },
        {
            'name': 'CNN-LSTM',
            'keras_path': project_root / 'results' / 'cnn_lstm_20251206_154643' / 'best_model.h5',
            'output_dir': project_root / 'mobile-app' / 'src' / 'assets' / 'models' / 'cnn_lstm'
        }
    ]
    
    print(f"\n{'='*60}")
    print(f"Keras to TensorFlow.js Model Conversion")
    print(f"{'='*60}")
    print(f"Converting {len(models)} models...\n")
    
    # Convert each model
    results = []
    for model_info in models:
        if not model_info['keras_path'].exists():
            print(f"⚠️  Skipping {model_info['name']}: Model file not found")
            print(f"   Expected: {model_info['keras_path']}")
            results.append(False)
            continue
        
        success = convert_keras_to_tfjs(
            keras_model_path=str(model_info['keras_path']),
            output_dir=str(model_info['output_dir']),
            model_name=model_info['name']
        )
        results.append(success)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Conversion Summary")
    print(f"{'='*60}\n")
    
    successful = sum(results)
    total = len(results)
    
    for i, model_info in enumerate(models):
        status = "✅ Success" if results[i] else "❌ Failed"
        print(f"{model_info['name']}: {status}")
    
    print(f"\nTotal: {successful}/{total} models converted successfully")
    
    if successful == total:
        print("\n🎉 All models converted successfully!")
        print("\nNext steps:")
        print("1. Run: cd mobile-app")
        print("2. Run: npm install")
        print("3. Run: npx react-native run-android")
    else:
        print("\n⚠️  Some models failed to convert. Check errors above.")


if __name__ == '__main__':
    main()
