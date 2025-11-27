# Next Steps: Baseline Model Development

## 🎉 Preprocessing Complete!

**Data Successfully Processed:**
- ✅ 503 subjects loaded
- ✅ 7,713 training windows (after 10x augmentation)
- ✅ 751 validation windows
- ✅ 3,957 test windows
- ✅ Window size: 200 samples (2 seconds at 100 Hz)
- ✅ 50% overlap applied
- ✅ Saved to `data/processed/`

---

## 🚀 Phase 6: Baseline Model Development

### Objective
Build and evaluate baseline deep learning models for gait-based activity classification (5 classes).

### Models to Implement

#### 1. **1D Convolutional Neural Network (1D CNN)** ⭐ START HERE
**Why**: Simple, fast, effective for time-series classification

**Architecture**:
```
Input: (200, 6)
├── Conv1D(64, kernel=5) + ReLU + BatchNorm
├── MaxPool1D(2)
├── Conv1D(128, kernel=5) + ReLU + BatchNorm
├── MaxPool1D(2)
├── Conv1D(256, kernel=3) + ReLU + BatchNorm
├── GlobalAveragePooling1D()
├── Dense(128) + ReLU + Dropout(0.5)
└── Dense(5, softmax)
```

**Expected Performance**: 75-85% accuracy

---

#### 2. **LSTM (Long Short-Term Memory)**
**Why**: Captures temporal dependencies in gait patterns

**Architecture**:
```
Input: (200, 6)
├── LSTM(128, return_sequences=True)
├── Dropout(0.3)
├── LSTM(64)
├── Dropout(0.3)
├── Dense(64) + ReLU
└── Dense(5, softmax)
```

**Expected Performance**: 70-80% accuracy

---

#### 3. **Hybrid CNN-LSTM**
**Why**: Combines spatial feature extraction (CNN) with temporal modeling (LSTM)

**Architecture**:
```
Input: (200, 6)
├── Conv1D(64, kernel=5) + ReLU + BatchNorm
├── MaxPool1D(2)
├── Conv1D(128, kernel=5) + ReLU + BatchNorm
├── MaxPool1D(2)
├── LSTM(128, return_sequences=False)
├── Dropout(0.5)
├── Dense(64) + ReLU
└── Dense(5, softmax)
```

**Expected Performance**: 80-90% accuracy (best expected)

---

## 📋 Implementation Tasks

### Task 1: Setup Model Training Infrastructure
- [ ] Create `src/models/` directory
- [ ] Implement data loader for processed data
- [ ] Create training utilities (metrics, callbacks)
- [ ] Setup TensorBoard logging
- [ ] Create model checkpointing

### Task 2: Implement 1D CNN Baseline
- [ ] Define model architecture
- [ ] Implement training loop
- [ ] Train on training set
- [ ] Evaluate on validation set
- [ ] Test on test set (probe subjects)
- [ ] Save model and results

### Task 3: Implement LSTM Baseline
- [ ] Define model architecture
- [ ] Train and evaluate
- [ ] Compare with CNN

### Task 4: Implement CNN-LSTM Hybrid
- [ ] Define model architecture
- [ ] Train and evaluate
- [ ] Compare with previous models

### Task 5: Evaluation & Analysis
- [ ] Generate confusion matrices
- [ ] Per-class accuracy analysis
- [ ] Identify misclassified samples
- [ ] Error analysis (which activities confused?)
- [ ] Create performance report

---

## 🎯 Training Strategy

### Hyperparameters
```python
config = {
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy'],
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5
}
```

### Class Weighting
Since we augmented training data, classes should be more balanced. But still use class weights:
```python
# Inverse frequency weighting
class_weights = {
    0: 1.0,  # Flat walk (most common)
    1: 2.0,  # Up stairs (augmented)
    2: 2.0,  # Down stairs (augmented)
    3: 1.5,  # Up slope
    4: 1.5   # Down slope
}
```

### Callbacks
- **EarlyStopping**: Stop if val_loss doesn't improve for 10 epochs
- **ReduceLROnPlateau**: Reduce learning rate if val_loss plateaus
- **ModelCheckpoint**: Save best model based on val_accuracy
- **TensorBoard**: Log training metrics

---

## 📊 Evaluation Metrics

### Primary Metrics
1. **Accuracy**: Overall classification accuracy
2. **Per-class Accuracy**: Accuracy for each activity
3. **F1-Score**: Harmonic mean of precision and recall
4. **Confusion Matrix**: Visualize misclassifications

### Secondary Metrics
5. **Precision**: True positives / (True positives + False positives)
6. **Recall**: True positives / (True positives + False negatives)
7. **Subject-level Accuracy**: Aggregate predictions per subject

### Success Criteria
- ✅ Overall accuracy > 80% on test set
- ✅ Per-class accuracy > 70% for all classes
- ✅ Stairs classes (1, 2) accuracy > 65% (hardest classes)

---

## 🔄 Iterative Improvement Plan

### If Performance is Low (<75%)
1. **Check data quality**:
   - Visualize sample windows
   - Verify labels are correct
   - Check for data leakage

2. **Adjust preprocessing**:
   - Try different window sizes (1.5s, 2.5s, 3s)
   - Experiment with overlap (25%, 75%)
   - Add/remove filtering

3. **Increase augmentation**:
   - 15x or 20x for stairs classes
   - Add more augmentation techniques

### If Stairs Classes Perform Poorly
1. **Collect more data** (if possible)
2. **Use focal loss** (focuses on hard examples)
3. **Separate stairs classifier** (binary: stairs vs non-stairs, then up vs down)

### If Model Overfits (train >> val)
1. **Increase dropout** (0.5 → 0.6)
2. **Add L2 regularization**
3. **Reduce model capacity**
4. **More data augmentation**

### If Model Underfits (train and val both low)
1. **Increase model capacity** (more layers/units)
2. **Train longer** (more epochs)
3. **Reduce regularization**
4. **Check learning rate**

---

## 📁 Expected File Structure

```
src/models/
├── __init__.py
├── cnn_1d.py              # 1D CNN model
├── lstm.py                # LSTM model
├── cnn_lstm.py            # Hybrid model
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── data_loader.py         # Load processed data
└── utils.py               # Utilities (metrics, callbacks)

results/
├── cnn_1d/
│   ├── model.h5
│   ├── history.json
│   ├── confusion_matrix.png
│   └── evaluation_report.json
├── lstm/
└── cnn_lstm/
```

---

## ⏱️ Estimated Timeline

- **Day 1**: Setup infrastructure + implement 1D CNN
- **Day 2**: Train CNN, implement LSTM
- **Day 3**: Train LSTM, implement CNN-LSTM
- **Day 4**: Train CNN-LSTM, comprehensive evaluation
- **Day 5**: Error analysis, iteration, final report

**Total**: ~5 days for baseline models

---

## 🎯 Immediate Next Action

**I recommend**: Start with **1D CNN baseline**

**Why**:
1. ✅ Simplest architecture
2. ✅ Fastest to train (~10-15 min)
3. ✅ Good baseline performance expected
4. ✅ Easy to debug and iterate

**Shall I create the model training infrastructure and 1D CNN implementation?**

This will include:
- Data loader for processed data
- 1D CNN model definition
- Training script with callbacks
- Evaluation script with metrics
- Visualization utilities (confusion matrix, training curves)
