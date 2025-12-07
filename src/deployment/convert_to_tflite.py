"""
Convert trained Keras models (.h5) to TensorFlow Lite format for mobile deployment.

This script:
1. Loads trained models (LSTM, CNN)
2. Converts to TensorFlow Lite format
3. Applies optimization (quantization)
4. Saves .tflite files and metadata
"""

import tensorflow as tf
import numpy as np
import json
import os
from pathlib import Path
from typing import Tuple, Dict
import argparse


def convert_model_to_tflite(
    model_path: str,
    output_path: str,
    quantize: str = 'float16',
    representative_dataset: np.ndarray = None
) -> Tuple[str, Dict]:
    """
    Convert Keras model to TensorFlow Lite format.
    
    Args:
        model_path: Path to .h5 model file
        output_path: Path to save .tflite file
        quantize: Quantization type ('none', 'float16', 'int8')
        representative_dataset: Sample data for int8 quantization
        
    Returns:
        Tuple of (tflite_path, conversion_info)
    """
    print(f"\n{'='*60}")
    print(f"Converting: {model_path}")
    print(f"{'='*60}\n")
    
    # Load Keras model
    print("Loading Keras model...")
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded: {model.name}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Parameters: {model.count_params():,}")
    
    # Convert to TensorFlow Lite
    print("\nConverting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable support for LSTM/GRU layers (required for recurrent models)
    # This allows TensorFlow ops in addition to TFLite built-in ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # TFLite built-in ops
        tf.lite.OpsSet.SELECT_TF_OPS     # TensorFlow ops (needed for LSTM/GRU)
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    # Apply optimization
    if quantize == 'float16':
        print("Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == 'int8':
        print("Applying int8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if representative_dataset is not None:
            def representative_data_gen():
                for sample in representative_dataset[:100]:  # Use 100 samples
                    yield [sample.astype(np.float32)]
            
            converter.representative_dataset = representative_data_gen
            # Keep SELECT_TF_OPS for LSTM/GRU support
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
    else:
        print("No quantization applied (float32)...")
    
    # Convert
    tflite_model = converter.convert()
    
    # Save TFLite model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file sizes
    original_size = os.path.getsize(model_path)
    tflite_size = os.path.getsize(output_path)
    compression_ratio = (1 - tflite_size / original_size) * 100
    
    print(f"\n✅ Conversion successful!")
    print(f"   Original size: {original_size / 1024:.1f} KB")
    print(f"   TFLite size: {tflite_size / 1024:.1f} KB")
    print(f"   Compression: {compression_ratio:.1f}%")
    print(f"   Saved to: {output_path}")
    
    conversion_info = {
        'model_name': model.name,
        'input_shape': list(model.input_shape),
        'output_shape': list(model.output_shape),
        'parameters': int(model.count_params()),
        'original_size_kb': round(original_size / 1024, 2),
        'tflite_size_kb': round(tflite_size / 1024, 2),
        'compression_ratio': round(compression_ratio, 2),
        'quantization': quantize
    }
    
    return output_path, conversion_info


def create_model_metadata(
    model_info: Dict,
    output_path: str,
    class_names: list = None,
    preprocessing_params: Dict = None
) -> str:
    """
    Create metadata JSON file for mobile app.
    
    Args:
        model_info: Model conversion information
        output_path: Path to save metadata JSON
        class_names: List of activity class names
        preprocessing_params: Preprocessing parameters
        
    Returns:
        Path to metadata file
    """
    if class_names is None:
        class_names = [
            "Flat Walk",
            "Up Stairs",
            "Down Stairs",
            "Up Slope",
            "Down Slope"
        ]
    
    if preprocessing_params is None:
        preprocessing_params = {
            'window_size': 200,
            'sampling_rate': 100,
            'overlap': 0.5,
            'normalization': 'z-score',
            'features': ['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az']
        }
    
    metadata = {
        'model_info': model_info,
        'class_names': class_names,
        'preprocessing': preprocessing_params,
        'version': '1.0.0'
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Metadata saved to: {output_path}")
    return output_path


def main():
    """Main conversion script."""
    parser = argparse.ArgumentParser(description='Convert Keras models to TensorFlow Lite')
    parser.add_argument('--model', type=str, help='Path to .h5 model file')
    parser.add_argument('--output-dir', type=str, default='models/mobile',
                        help='Output directory for .tflite files')
    parser.add_argument('--quantize', type=str, default='float16',
                        choices=['none', 'float16', 'int8'],
                        help='Quantization type')
    parser.add_argument('--all', action='store_true',
                        help='Convert all models in results directory')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Models to convert
    models_to_convert = []
    
    if args.all:
        # Find latest LSTM and CNN models
        results_dir = project_root / 'results'
        
        # Find latest LSTM
        lstm_dirs = sorted(results_dir.glob('lstm_*'), reverse=True)
        if lstm_dirs:
            lstm_model = lstm_dirs[0] / 'best_model.h5'
            if lstm_model.exists():
                models_to_convert.append(('lstm', str(lstm_model)))
        
        # Find latest CNN
        cnn_dirs = sorted(results_dir.glob('1d_cnn_*'), reverse=True)
        if cnn_dirs:
            cnn_model = cnn_dirs[0] / 'best_model.h5'
            if cnn_model.exists():
                models_to_convert.append(('cnn', str(cnn_model)))
        
        # Find latest CNN-LSTM
        cnn_lstm_dirs = sorted(results_dir.glob('cnn_lstm_*'), reverse=True)
        if cnn_lstm_dirs:
            cnn_lstm_model = cnn_lstm_dirs[0] / 'best_model.h5'
            if cnn_lstm_model.exists():
                models_to_convert.append(('cnn_lstm', str(cnn_lstm_model)))
    
    elif args.model:
        model_name = Path(args.model).parent.parent.name.split('_')[0]
        models_to_convert.append((model_name, args.model))
    
    else:
        print("Error: Specify --model or --all")
        return
    
    if not models_to_convert:
        print("No models found to convert!")
        return
    
    print(f"\n{'='*60}")
    print(f"TensorFlow Lite Model Conversion")
    print(f"{'='*60}")
    print(f"Found {len(models_to_convert)} model(s) to convert")
    
    # Load representative dataset for int8 quantization
    representative_data = None
    if args.quantize == 'int8':
        print("\nLoading representative dataset for int8 quantization...")
        try:
            data_dir = project_root / 'data' / 'processed'
            X_test = np.load(data_dir / 'X_test.npy')
            representative_data = X_test[:100]  # Use 100 samples
            print(f"✅ Loaded {len(representative_data)} samples")
        except Exception as e:
            print(f"⚠️  Could not load representative dataset: {e}")
            print("   Falling back to float16 quantization")
            args.quantize = 'float16'
    
    # Convert each model
    output_dir = project_root / args.output_dir
    all_metadata = {}
    
    for model_name, model_path in models_to_convert:
        # Output path
        output_path = output_dir / f'gait_{model_name}_model.tflite'
        
        # Convert
        tflite_path, conversion_info = convert_model_to_tflite(
            model_path=model_path,
            output_path=str(output_path),
            quantize=args.quantize,
            representative_dataset=representative_data
        )
        
        # Save individual metadata
        metadata_path = output_dir / f'gait_{model_name}_metadata.json'
        create_model_metadata(
            model_info=conversion_info,
            output_path=str(metadata_path)
        )
        
        all_metadata[model_name] = conversion_info
    
    # Save combined metadata
    combined_metadata_path = output_dir / 'models_metadata.json'
    with open(combined_metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ All conversions complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"\nConverted models:")
    for model_name in all_metadata:
        print(f"  - gait_{model_name}_model.tflite")
    print(f"\nNext steps:")
    print(f"  1. Test converted models: python src/deployment/test_tflite_model.py")
    print(f"  2. Copy .tflite files to React Native app")
    print(f"  3. Use metadata.json for preprocessing parameters")


if __name__ == '__main__':
    main()
