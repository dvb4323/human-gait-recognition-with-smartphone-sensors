# Phase 1: Model Training Guide

## Overview
Train 3 additional models to complete Phase 1: GRU, CNN-LSTM, and BiLSTM.

## Quick Training Commands

### 1. GRU (Fastest - ~20-30 min)
```bash
cd src/models
# Edit train_lstm.py: set model_variant='gru'
python train_lstm.py
```

### 2. Bidirectional LSTM (~45-60 min)
```bash
cd src/models
# Edit train_lstm.py: set model_variant='bidirectional'
python train_lstm.py
```

### 3. CNN-LSTM Hybrid (~30-45 min)
```bash
cd src/models
python train_cnn_lstm.py
```

## Expected Results

| Model | Expected Accuracy | Training Time | Parameters |
|-------|------------------|---------------|------------|
| **GRU** | 90-91% | 20-30 min | ~150K |
| **BiLSTM** | 91-92% | 45-60 min | ~400K |
| **CNN-LSTM** | 91-93% | 30-45 min | ~250K |

## After Training

You'll have **5 models total**:
1. ✅ 1D CNN: 90.3%
2. ✅ LSTM: 91.6%
3. ⏳ GRU: ?
4. ⏳ BiLSTM: ?
5. ⏳ CNN-LSTM: ?

## Phase 1 Completion Checklist

- [ ] Train GRU model
- [ ] Train BiLSTM model
- [ ] Train CNN-LSTM model
- [ ] Compare all 5 models
- [ ] Select top 2-3 for Phase 2

## Model Selection Criteria

Select models based on:
1. **Accuracy** > 91%
2. **Stairs performance** (Class 1, 2)
3. **Balanced F1-scores** across classes
4. **Inference speed** (if deploying to edge devices)

## Phase 2 Preview

After selecting top 2-3 models, you'll:
- Fine-tune hyperparameters
- Try focal loss for stairs
- Experiment with window sizes
- Create ensemble
- Target: 92-94% accuracy

## Tips

**For GRU**:
- Faster than LSTM
- Similar performance expected
- Good if you need speed

**For BiLSTM**:
- Sees full context (forward + backward)
- Best for complex patterns
- May improve stairs/slope distinction

**For CNN-LSTM**:
- Best of both worlds
- CNN extracts features, LSTM models time
- Likely highest accuracy
