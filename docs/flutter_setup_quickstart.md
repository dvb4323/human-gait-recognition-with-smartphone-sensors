# Flutter App Setup - Quick Reference

## \ud83d\ude80 Files to Copy from Training Project

### Required Files

**1. TFLite Model** (209 KB)
```
FROM: models/mobile/gait_lstm_model.tflite
TO:   <flutter-app>/assets/models/gait_model.tflite
```

**2. Preprocessing Parameters** (CRITICAL!)
```
EXTRACT FROM: data/processed_no_overlap/preprocessing_config.json
CREATE:       <flutter-app>/assets/models/preprocessing_params.json

Structure:
{
  "mean": [6 values from preprocessor_params.mean],
  "std": [6 values from preprocessor_params.std],
  "window_size": 200,
  "sampling_rate": 100
}
```

### Optional Files

**3. Model Metadata** (for reference)
```
FROM: models/mobile/gait_lstm_metadata.json
TO:   <flutter-app>/assets/models/model_metadata.json
```

---

## ✅ Setup Checklist

- [ ] Copy gait_lstm_model.tflite (GRU model)
- [ ] Extract mean & std from preprocessing_config.json
- [ ] Create preprocessing_params.json with extracted values
- [ ] Update pubspec.yaml to include asset files
- [ ] Verify model file size (should be ~209 KB)
- [ ] Test with dummy data before deploying

---

## \u26a0\ufe0f  Critical Information

### Model Input
- **Shape**: [1, 200, 6]
- **Type**: float32
- **Format**: [Gx, Gy, Gz, Ax, Ay, Az]
- **Window**: 200 samples = 2 seconds at 100 Hz

### Model Output
- **Shape**: [1, 5]
- **Type**: float32
- **Format**: [prob_class0, prob_class1, ..., prob_class4]
- **Classes**: 0=Flat Walk, 1=Up Stairs, 2=Down Stairs, 3=Up Slope, 4=Down Slope

### Normalization Formula
```dart
normalized_value = (raw_value - mean[i]) / std[i]
```

**Without correct normalization, predictions will be wrong!**

---

## \ud83d\udcf1 Expected Performance

- **Accuracy**: 89.9% (matches GRU test accuracy)
- **Latency**: 50-100ms per prediction
- **Battery**: 5-10% per hour
- **Confidence**: 70-95% for clear activities

---

## \ud83d\udd17 Reference Documentation

- **Full Guide**: See `docs/mobile_app_guide.md`
- **Model Conversion**: `src/deployment/convert_to_tflite.py`
- **Training Results**: `docs/leakage_fix_verification.md`
