# 1D CNN Model Training

## Overview
Complete implementation of 1D Convolutional Neural Network for gait-based activity classification.

## Quick Start

### Train Model
```bash
cd src/models
python train.py
```

This will:
1. Load preprocessed data from `data/processed/`
2. Create and compile 1D CNN model
3. Train for 50 epochs with early stopping
4. Evaluate on test set
5. Save results to `results/1d_cnn_TIMESTAMP/`

### Expected Training Time
- **GPU**: ~10-15 minutes
- **CPU**: ~30-45 minutes

## Modules

### 1. data_loader.py
Loads preprocessed `.npy` data and creates TensorFlow datasets.

**Usage**:
```python
from data_loader import GaitDataLoader

loader = GaitDataLoader('data/processed')
data = loader.load_all()
train_ds = loader.create_tf_dataset('train', batch_size=64)
```

### 2. cnn_1d.py
Three 1D CNN model variants:

- **Simple** (~50K params): Fast, good baseline
- **Standard** (~200K params): Balanced performance/speed
- **Deep** (~500K params): More capacity, needs more data

**Usage**:
```python
from cnn_1d import create_1d_cnn, create_simple_1d_cnn, compile_model

model = create_1d_cnn(input_shape=(200, 6), num_classes=5)
model = compile_model(model, learning_rate=0.001)
```

### 3. train.py
Complete training pipeline with:
- ✅ Model checkpointing (saves best model)
- ✅ Early stopping (patience=10 epochs)
- ✅ Learning rate reduction (on plateau)
- ✅ TensorBoard logging
- ✅ Automatic evaluation and visualization

## Configuration

Edit `train.py` to customize:
```python
config = {
    'model_variant': 'standard',  # 'simple', 'standard', or 'deep'
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 10
}
```

## Output Structure

```
results/1d_cnn_TIMESTAMP/
├── best_model.h5              # Best model (highest val_accuracy)
├── final_model.h5             # Final model after all epochs
├── config.json                # Training configuration
├── history.json               # Training history
├── training_history.csv       # Epoch-by-epoch metrics
├── training_history.png       # Accuracy/loss plots
├── confusion_matrix.png       # Confusion matrix visualization
├── classification_report.json # Per-class metrics
├── evaluation_results.json    # Test set results
└── logs/                      # TensorBoard logs
```

## Expected Results

Based on similar HAR tasks with IMU data:
- **Target accuracy**: 80-85% on test set
- **Training time**: ~10-15 min (GPU)
- **Best epoch**: Usually around 20-30 epochs

### Per-Class Performance
- Class 0 (Flat walk): 85-90% (most common)
- Class 1 (Up stairs): 70-75% (augmented)
- Class 2 (Down stairs): 70-75% (augmented)
- Class 3 (Up slope): 75-80%
- Class 4 (Down slope): 75-80%

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir results/1d_cnn_TIMESTAMP/logs
```
Then open http://localhost:6006

### Training Progress
Watch for:
- ✅ Validation accuracy increasing
- ✅ Training/validation gap not too large (overfitting check)
- ✅ Learning rate reductions (indicates plateau)
- ✅ Early stopping trigger (if val_loss doesn't improve)

## Troubleshooting

### Low Accuracy (<70%)
1. Check data quality (visualize samples)
2. Increase augmentation factor
3. Try different model variant
4. Adjust learning rate

### Overfitting (train >> val)
1. Increase dropout (0.5 → 0.6)
2. Add more augmentation
3. Use simpler model variant
4. Reduce training epochs

### Underfitting (both low)
1. Use deeper model variant
2. Train longer (more epochs)
3. Increase learning rate
4. Check data preprocessing

## Next Steps

After training:
1. Review `classification_report.json` for per-class performance
2. Analyze `confusion_matrix.png` for common errors
3. If performance is good, try LSTM or CNN-LSTM
4. If performance is low, iterate on preprocessing/augmentation
