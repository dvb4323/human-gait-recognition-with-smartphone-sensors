# Preprocessing Pipeline - Phase 5

## Overview
Complete preprocessing pipeline for OU-SimilarGaitActivities dataset (Center sensor only).

## Features
✅ Loads Center sensor data (503 subjects, 100% coverage)
✅ Removes unlabeled data (ClassLabel = -1)
✅ Creates fixed-length windows (2s with 50% overlap)
✅ Applies normalization and optional filtering
✅ Augments minority classes (stairs: 10x augmentation)
✅ Respects gallery/probe protocol for train/val/test splits
✅ Saves processed data to `data/processed/`

## Quick Start

### Run Complete Pipeline
```bash
cd src/preprocessing
python pipeline.py
```

This will:
1. Load all 503 Center sensor files
2. Remove unlabeled data (ClassLabel = -1)
3. Create 2-second windows with 50% overlap
4. Split: Gallery → Train (80%) + Val (20%), Probe → Test (100%)
5. Apply Z-score normalization
6. Augment stairs classes (1, 2) by 10x
7. Save to `data/processed/train/`, `data/processed/val/`, `data/processed/test/`

### Configuration
Edit `pipeline.py` to customize:
```python
config = {
    'window_size': 200,  # 2 seconds at 100 Hz
    'overlap': 0.5,  # 50% overlap
    'normalize': True,
    'filter_data': False,  # Enable Butterworth filter
    'augment': True,
    'minority_classes': [1, 2],  # Stairs classes
    'augmentation_factor': 10
}
```

## Modules

### 1. data_loader.py
- Loads Center sensor data
- Filters unlabeled samples (ClassLabel = -1)
- Manages gallery/probe subject lists
- **Usage**:
```python
from data_loader import OUGaitDataLoader

loader = OUGaitDataLoader("data/raw/OU-SimilarGaitActivities")
data = loader.load_all_data(remove_unlabeled=True)
```

### 2. preprocessor.py
- Z-score normalization (fit on training data)
- Butterworth low-pass filter (optional, 20 Hz cutoff)
- Fixed-length windowing with overlap
- **Usage**:
```python
from preprocessor import SensorPreprocessor, create_windows

preprocessor = SensorPreprocessor(normalize=True, filter_data=False)
preprocessor.fit(train_data)
processed = preprocessor.transform(test_data)

windows, labels = create_windows(data, labels, window_size=200, overlap=0.5)
```

### 3. augmentation.py
- Jittering (Gaussian noise)
- Scaling (amplitude variation)
- Time warping (speed variation)
- Rotation (sensor orientation)
- **Usage**:
```python
from augmentation import balance_classes

balanced_X, balanced_y = balance_classes(
    X_train, y_train,
    minority_classes=[1, 2],
    augmentation_factor=10
)
```

### 4. pipeline.py
- Orchestrates complete workflow
- Handles train/val/test splits
- Saves processed data and metadata

## Output Structure
```
data/processed/
├── train/
│   ├── X_train.npy          # Shape: (N_train, 200, 6)
│   ├── y_train.npy          # Shape: (N_train,)
│   └── metadata_train.json
├── val/
│   ├── X_val.npy
│   ├── y_val.npy
│   └── metadata_val.json
├── test/
│   ├── X_test.npy
│   ├── y_test.npy
│   └── metadata_test.json
├── preprocessing_config.json
└── preprocessing_summary.json
```

## Expected Results
Based on Phase 1-2 analysis:
- **Total subjects**: 503
- **Labeled samples**: ~850K (after removing 52% unlabeled)
- **Windows** (2s, 50% overlap): ~85K windows
- **Train**: ~54K windows (after 10x augmentation of stairs)
- **Val**: ~7K windows
- **Test**: ~24K windows

## Class Distribution
Before augmentation:
- Class 0 (Flat): ~60%
- Class 1 (Up stairs): ~5%
- Class 2 (Down stairs): ~5%
- Class 3 (Up slope): ~15%
- Class 4 (Down slope): ~15%

After augmentation (training only):
- All classes more balanced (~15-25% each)

## Next Steps
After preprocessing:
1. Load processed data for model training
2. Build baseline models (1D CNN, LSTM)
3. Evaluate on test set (probe subjects)
4. Iterate on model architecture

## Testing Individual Modules
```bash
# Test data loader
python data_loader.py

# Test preprocessor
python preprocessor.py

# Test augmentation
python augmentation.py
```
