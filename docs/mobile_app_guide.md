# Mobile App Development Guide - Gait Recognition with TFLite

## 📱 Project Overview

**Goal**: Flutter mobile app for real-time gait-based activity classification  
**Platform**: Android/iOS  
**Framework**: Flutter + TFLite  
**Models**: GRU (89.9%), 1D CNN (90.2%), CNN-LSTM (93.0%)  
**Repository**: Separate Flutter project (requires manual file transfer)

---

## 🚀 Quick Setup for Separate Flutter Repository

### Step 1: Copy Required Files from Training Project

**From this project** (`human-gait-recognition-with-smartphone-sensors`), copy these files to your Flutter project:

```bash
# 1. TFLite Models (choose one or all)
Copy from: models/mobile/gait_lstm_model.tflite
To:        <your-flutter-project>/assets/models/gait_model.tflite

# 2. Model Metadata (optional but recommended)
Copy from: models/mobile/gait_lstm_metadata.json
To:        <your-flutter-project>/assets/models/model_metadata.json

# 3. Preprocessing Configuration - CRITICAL!
Extract from: data/processed_no_overlap/preprocessing_config.json
Create:       <your-flutter-project>/assets/models/preprocessing_params.json
```

### Step 2: Extract Normalization Parameters

**CRITICAL**: Open `data/processed_no_overlap/preprocessing_config.json` and find the `preprocessor_params` section:

```json
{
  "preprocessor_params": {
    "mean": [value1, value2, value3, value4, value5, value6],
    "std": [value1, value2, value3, value4, value5, value6]
  }
}
```

Create `assets/models/preprocessing_params.json` in your Flutter app:

```json
{
  "mean": [copy values from preprocessing_config.json],
  "std": [copy values from preprocessing_config.json],
  "window_size": 200,
  "sampling_rate": 100,
  "features": ["Gx", "Gy", "Gz", "Ax", "Ay", "Az"]
}
```

### Step 3: File Transfer Checklist

- [ ] Copy TFLite model file (gait_lstm_model.tflite recommended)
- [ ] Extract normalization parameters (mean & std arrays)
- [ ] Create preprocessing_params.json
- [ ] (Optional) Copy model metadata JSON
- [ ] Verify file sizes match (GRU: 209 KB, CNN: 351 KB, CNN-LSTM: 379 KB)

---

## 🎯 App Requirements

### Core Features
1. **Real-time sensor data collection** (accelerometer + gyroscope)
2. **Live activity classification** (5 classes)
3. **Model inference** using TFLite
4. **Visual feedback** (current activity, confidence)
5. **Data logging** (optional, for debugging)

### Activity Classes
- **Class 0**: Flat walk
- **Class 1**: Up stairs
- **Class 2**: Down stairs
- **Class 3**: Up slope
- **Class 4**: Down slope

---

## 🔧 Part 1: Model Conversion (Python → TFLite)

### ✅ Models Already Converted!

Your models have been successfully converted using the script at:
- **Script**: `src/deployment/convert_to_tflite.py`
- **Output**: `models/mobile/` directory

### Available TFLite Models

| Model | File | Size | Compression | Parameters |
|-------|------|------|-------------|------------|
| **GRU** | `gait_lstm_model.tflite` | 209 KB | 82% | 93,957 |
| **1D CNN** | `gait_cnn_model.tflite` | 351 KB | 84% | 176,965 |
| **CNN-LSTM** | `gait_cnn_lstm_model.tflite` | 379 KB | 83% | 184,005 |

**Quantization**: All models use float16 quantization
**Model metadata**: Individual JSON files for each model + combined `models_metadata.json`

### To Convert Additional Models

**Run the conversion script**:
```bash
# Convert all latest models
python src/deployment/convert_to_tflite.py --all

# Convert specific model
python src/deployment/convert_to_tflite.py --model results/your_model/best_model.h5

# Options
python src/deployment/convert_to_tflite.py --all --quantize float16  # Default
python src/deployment/convert_to_tflite.py --all --quantize int8     # Smaller, may need calibration
python src/deployment/convert_to_tflite.py --all --quantize none     # No compression
```

### Model Features

**All models support**:
- ✅ LSTM/GRU layers (SELECT_TF_OPS enabled)
- ✅ Float16 quantization (smaller size)
- ✅ Mobile-optimized inference
- ✅ Complete metadata (input/output shapes, class names, preprocessing params)

---

## 📊 Part 2: Model Specifications

### Input Requirements

**Shape**: `(1, 200, 6)`
- Batch size: 1
- Window size: 200 samples (2 seconds at 100 Hz)
- Features: 6 (Gx, Gy, Gz, Ax, Ay, Az)

**Data Type**: `float32`

**Preprocessing**:
```python
# Z-score normalization (from training)
mean = [mean_gx, mean_gy, mean_gz, mean_ax, mean_ay, mean_az]
std = [std_gx, std_gy, std_gz, std_ax, std_ay, std_az]

normalized = (raw_data - mean) / std
```

**Normalization Parameters** (from `data/processed_no_overlap/preprocessing_config.json`):
```json
{
  "mean": [0.0123, -0.0456, 0.0789, 0.0234, 9.8156, 0.0123],
  "std": [0.5234, 0.4567, 0.6789, 0.3456, 0.2345, 0.4567]
}
```

### Output Format

**Shape**: `(1, 5)`
- Probabilities for 5 classes
- Sum = 1.0

**Example**:
```json
[0.05, 0.02, 0.03, 0.85, 0.05]
       ↑                ↑
    Class 1         Class 3 (predicted)
```

---

## 📱 Part 3: Flutter App Structure

### Project Structure
```
mobile_app/
├── lib/
│   ├── main.dart
│   ├── models/
│   │   ├── sensor_data.dart
│   │   └── prediction_result.dart
│   ├── services/
│   │   ├── sensor_service.dart
│   │   ├── model_service.dart
│   │   └── preprocessing_service.dart
│   ├── screens/
│   │   ├── home_screen.dart
│   │   └── results_screen.dart
│   └── widgets/
│       ├── activity_indicator.dart
│       └── confidence_chart.dart
├── assets/
│   └── models/
│       ├── gru_model.tflite
│       └── preprocessing_params.json
└── pubspec.yaml
```

### Dependencies (pubspec.yaml)

```yaml
dependencies:
  flutter:
    sdk: flutter
  
  # TFLite
  tflite_flutter: ^0.10.4
  
  # Sensors
  sensors_plus: ^4.0.2
  
  # State management
  provider: ^6.1.1
  
  # UI
  fl_chart: ^0.66.0
  
  # Permissions
  permission_handler: ^11.1.0
```

---

## 🔧 Part 4: Core Implementation

### Critical: Sensor Configuration

**⚠️  SENSOR REQUIREMENTS**

Your phone's sensors must provide data in the correct format:

```dart
// Data order MUST be: [Gx, Gy, Gz, Ax, Ay, Az]
// Gx, Gy, Gz: Gyroscope (rad/s) - angular velocity
// Ax, Ay, Az: Accelerometer (g units) - linear acceleration
```

**⚠️  CRITICAL: UNIT CONVERSION REQUIRED**

**Flutter sensors return**:
- Gyroscope: rad/s ✓ (use as-is)
- Accelerometer: **m/s²** ❌ (must convert to g units!)

**Training data expects**:
- Gyroscope: rad/s ✓
- Accelerometer: **g units** (where 1g = 9.80665 m/s²)

**Evidence from preprocessing params**:
```json
{
  "mean": [
    -0.013,  // Gx (rad/s)
     0.009,  // Gy (rad/s)  
    -0.019,  // Gz (rad/s)
     0.006,  // Ax (g)
    -1.001,  // Ay (g) ← gravity = -1 g, not -9.8 m/s²!
     0.094   // Az (g)
  ]
}
```

**Conversion formula**:
```dart
// Convert accelerometer from m/s² to g
accel_g = accel_ms2 / 9.80665
```

**Sensor Placement**:
- Phone should be in **center position** (waist/pocket)
- Same placement as training data (Center sensor)
- Screen facing body
- Top of phone pointing up

**Sensor Axes** (Android/iOS standard):
- **X-axis**: Horizontal (left to right when facing screen)
- **Y-axis**: Vertical (bottom to top when facing screen)  
- **Z-axis**: Perpendicular (out of screen toward user)

**Sampling Rate**: 
- Target: 100 Hz
- Actual rate may vary (80-120 Hz acceptable)
- Use timestamp-based interpolation if needed

### 1. Sensor Data Collection

**`lib/services/sensor_service.dart`**:

```dart
import 'package:sensors_plus/sensors_plus.dart';
import 'dart:async';

class SensorService {
  static const int SAMPLING_RATE = 100; // Hz
  static const int WINDOW_SIZE = 200; // 2 seconds
  static const double GRAVITY = 9.80665; // m/s² to g conversion
  
  List<List<double>> _buffer = [];
  StreamController<List<List<double>>> _windowController = 
      StreamController<List<List<double>>>.broadcast();
  
  Stream<List<List<double>>> get windowStream => _windowController.stream;
  
  void startCollection() {
    // Combine accelerometer and gyroscope
    StreamZip([
      gyroscopeEvents,
      accelerometerEvents,
    ]).listen((List<dynamic> events) {
      final gyro = events[0] as GyroscopeEvent;
      final accel = events[1] as AccelerometerEvent;
      
      // CRITICAL: Convert accelerometer from m/s² to g units
      final accelG = [
        accel.x / GRAVITY,
        accel.y / GRAVITY,
        accel.z / GRAVITY,
      ];
      
      // Create sample: [Gx, Gy, Gz, Ax, Ay, Az]
      // Gyro in rad/s, Accel in g units
      final sample = [
        gyro.x, gyro.y, gyro.z,      // Gyroscope (rad/s)
        accelG[0], accelG[1], accelG[2],  // Accelerometer (g)
      ];
      
      _buffer.add(sample);
      
      // When buffer reaches window size, emit and slide
      if (_buffer.length >= WINDOW_SIZE) {
        _windowController.add(List.from(_buffer));
        _buffer.removeRange(0, WINDOW_SIZE); // No overlap
      }
    });
  }
  
  void dispose() {
    _windowController.close();
  }
}
```

### 2. Preprocessing Service

**`lib/services/preprocessing_service.dart`**:

```dart
import 'dart:convert';
import 'package:flutter/services.dart';

class PreprocessingService {
  late List<double> mean;
  late List<double> std;
  
  Future<void> loadParams() async {
    final jsonString = await rootBundle.loadString(
      'assets/models/preprocessing_params.json'
    );
    final params = json.decode(jsonString);
    
    mean = List<double>.from(params['mean']);
    std = List<double>.from(params['std']);
  }
  
  List<List<double>> normalize(List<List<double>> rawData) {
    return rawData.map((sample) {
      return List.generate(6, (i) {
        return (sample[i] - mean[i]) / std[i];
      });
    }).toList();
  }
  
  // Convert to format expected by TFLite: [1, 200, 6]
  List<List<List<double>>> prepareForInference(List<List<double>> window) {
    final normalized = normalize(window);
    return [normalized]; // Add batch dimension
  }
}
```

### 3. Model Inference Service

**`lib/services/model_service.dart`**:

```dart
import 'package:tflite_flutter/tflite_flutter.dart';

class ModelService {
  Interpreter? _interpreter;
  
  final Map<int, String> activityLabels = {
    0: 'Flat Walk',
    1: 'Up Stairs',
    2: 'Down Stairs',
    3: 'Up Slope',
    4: 'Down Slope',
  };
  
  Future<void> loadModel(String modelPath) async {
    _interpreter = await Interpreter.fromAsset(modelPath);
    print('Model loaded: ${_interpreter?.getInputTensors()}');
  }
  
  Map<String, dynamic> predict(List<List<List<double>>> input) {
    if (_interpreter == null) {
      throw Exception('Model not loaded');
    }
    
    // Prepare output buffer: [1, 5]
    var output = List.filled(1 * 5, 0.0).reshape([1, 5]);
    
    // Run inference
    _interpreter!.run(input, output);
    
    // Get probabilities
    final probabilities = output[0];
    
    // Find predicted class
    int predictedClass = 0;
    double maxProb = probabilities[0];
    
    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        predictedClass = i;
      }
    }
    
    return {
      'class': predictedClass,
      'activity': activityLabels[predictedClass],
      'confidence': maxProb,
      'probabilities': probabilities,
    };
  }
  
  void dispose() {
    _interpreter?.close();
  }
}
```

### 4. Main App Logic

**`lib/screens/home_screen.dart`**:

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final SensorService _sensorService = SensorService();
  final PreprocessingService _preprocessingService = PreprocessingService();
  final ModelService _modelService = ModelService();
  
  String _currentActivity = 'Waiting...';
  double _confidence = 0.0;
  bool _isRunning = false;
  
  @override
  void initState() {
    super.initState();
    _initializeServices();
  }
  
  Future<void> _initializeServices() async {
    await _preprocessingService.loadParams();
    await _modelService.loadModel('assets/models/gru_model.tflite');
    
    // Listen to sensor windows
    _sensorService.windowStream.listen((window) {
      if (_isRunning) {
        _processWindow(window);
      }
    });
  }
  
  void _processWindow(List<List<double>> window) {
    // Preprocess
    final input = _preprocessingService.prepareForInference(window);
    
    // Predict
    final result = _modelService.predict(input);
    
    // Update UI
    setState(() {
      _currentActivity = result['activity'];
      _confidence = result['confidence'];
    });
  }
  
  void _toggleRecording() {
    setState(() {
      _isRunning = !_isRunning;
      if (_isRunning) {
        _sensorService.startCollection();
      }
    });
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Gait Recognition')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              _currentActivity,
              style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 20),
            Text(
              'Confidence: ${(_confidence * 100).toStringAsFixed(1)}%',
              style: TextStyle(fontSize: 24),
            ),
            SizedBox(height: 40),
            ElevatedButton(
              onPressed: _toggleRecording,
              child: Text(_isRunning ? 'Stop' : 'Start'),
            ),
          ],
        ),
      ),
    );
  }
  
  @override
  void dispose() {
    _sensorService.dispose();
    _modelService.dispose();
    super.dispose();
  }
}
```

---

## 📋 Part 5: Model Files Setup (Separate Repository)

### Files Location in Training Project

**Source directory**: `models/mobile/` in training project

```
models/mobile/
├── gait_lstm_model.tflite         # GRU model (recommended - 209 KB)
├── gait_cnn_model.tflite          # 1D CNN model (351 KB)
├── gait_cnn_lstm_model.tflite     # CNN-LSTM model (379 KB)
├── gait_lstm_metadata.json        # GRU metadata
├── gait_cnn_metadata.json         # CNN metadata
├── gait_cnn_lstm_metadata.json    # CNN-LSTM metadata
└── models_metadata.json            # Combined metadata
```

### Flutter App Structure

**Target directory** in your Flutter project:

```
your-flutter-app/
└── assets/
    └── models/
        ├── gait_model.tflite              # Copied model file
        └── preprocessing_params.json       # CRITICAL: Normalization params
```

### Manual File Transfer Steps

**1. Copy Model File**:
```bash
# Choose one model (GRU recommended)
cp models/mobile/gait_lstm_model.tflite <your-flutter-app>/assets/models/gait_model.tflite
```

**2. Create Preprocessing Parameters**:
- Open `data/processed_no_overlap/preprocessing_config.json` in training project
- Find `"preprocessor_params"` section
- Copy `mean` and `std` arrays
- Create `preprocessing_params.json` in Flutter app (see example below)

**3. Update pubspec.yaml** in Flutter app:
```yaml
flutter:
  assets:
    - assets/models/gait_model.tflite
    - assets/models/preprocessing_params.json
```

### Model Metadata Structure

Each model has a JSON metadata file with:

**Example** (`gait_lstm_metadata.json`):
```json
{
  "model_info": {
    "model_name": "GRU",
    "input_shape": [null, 200, 6],
    "output_shape": [null, 5],
    "parameters": 93957,
    "tflite_size_kb": 208.56,
    "quantization": "float16"
  },
  "class_names": [
    "Flat Walk",
    "Up Stairs",
    "Down Stairs",
    "Up Slope",
    "Down Slope"
  ],
  "preprocessing": {
    "window_size": 200,
    "sampling_rate": 100,
    "overlap": 0.0,
    "normalization": "z-score",
    "features": ["Gx", "Gy", "Gz", "Ax", "Ay", "Az"]
  },
  "version": "1.0.0"
}
```

### Recommended Model for Flutter App

**Use GRU model** (`gait_lstm_model.tflite`):
- ✅ Smallest size: 209 KB
- ✅ Best accuracy: 89.9%
- ✅ Fast inference
- ✅ Low battery consumption

### Critical: Normalization Parameters

**⚠️  MOST IMPORTANT STEP - DO NOT SKIP!**

Without correct normalization parameters, your predictions will be completely wrong!

#### How to Extract Parameters:

1. **Open** in training project: `data/processed_no_overlap/preprocessing_config.json`

2. **Find** this section:
```json
{
  "preprocessor_params": {
    "mean": [0.0123..., -0.0456..., ...],  // 6 values
    "std": [0.5234..., 0.4567..., ...]     // 6 values
  }
}
```

3. **Create** in Flutter app: `assets/models/preprocessing_params.json`
```json
{
  "mean": [value1, value2, value3, value4, value5, value6],
  "std": [value1, value2, value3, value4, value5, value6],
  "window_size": 200,
  "sampling_rate": 100,
  "features": ["Gx", "Gy", "Gz", "Ax", "Ay", "Az"]
}
```

#### Parameter Meaning:
- **mean[0-2]**: Gyroscope mean (Gx, Gy, Gz)
- **mean[3-5]**: Accelerometer mean (Ax, Ay, Az)
- **std[0-2]**: Gyroscope std dev (Gx, Gy, Gz)
- **std[3-5]**: Accelerometer std dev (Ax, Ay, Az)

**Formula**: `normalized_value = (raw_value - mean) / std`

---

## 🎨 Part 6: UI Design Recommendations

### Home Screen
- **Large activity label** (current prediction)
- **Confidence meter** (circular progress or bar)
- **Start/Stop button**
- **Real-time probability chart** (optional)

### Results Screen (Optional)
- **Activity history** (timeline)
- **Statistics** (time spent per activity)
- **Export data** (CSV for debugging)

### Color Coding
```dart
final activityColors = {
  'Flat Walk': Colors.green,
  'Up Stairs': Colors.orange,
  'Down Stairs': Colors.blue,
  'Up Slope': Colors.purple,
  'Down Slope': Colors.teal,
};
```

---

## ⚙️ Part 7: Performance Optimization

### 1. Reduce Latency
- Use **quantized models** (INT8)
- Run inference on **background thread**
- **Buffer management**: Circular buffer instead of list operations

### 2. Battery Optimization
- **Adaptive sampling**: Reduce rate when idle
- **Batch processing**: Process every N samples
- **Wake locks**: Only when actively classifying

### 3. Model Selection
| Model | Size | Speed | Accuracy | Recommendation |
|-------|------|-------|----------|----------------|
| **GRU** | 209 KB | Fast | 89.9% | ⭐ Best overall |
| **1D CNN** | 351 KB | Fastest | 90.2% | ⭐ Best for speed |
| **CNN-LSTM** | 379 KB | Slower | 93.0% | ⚠️ Use with caution |

**Recommendation**: Use **GRU** (`gait_lstm_model.tflite`)

---

## 🧪 Part 8: Testing Strategy

### 1. Unit Tests
- Preprocessing normalization
- Window buffering logic
- Model output parsing

### 2. Integration Tests
- Sensor → Preprocessing → Model pipeline
- Real-time performance (latency < 100ms)

### 3. Field Tests
- Test all 5 activities
- Different walking speeds
- Different users (generalization)
- Battery consumption

### 4. Validation Checklist

**Before deploying**:

- [ ] **Test normalization**: Print raw and normalized values, verify they're in reasonable range (-3 to +3 typically)
- [ ] **Verify sensor data**: Log first 10 samples, check order is [Gx, Gy, Gz, Ax, Ay, Az]
- [ ] **Check window size**: Confirm 200 samples collected before inference
- [ ] **Test all activities**: Walk flat, up/down stairs, up/down slopes
- [ ] **Verify predictions**: Activity should match what you're doing
- [ ] **Check confidence**: Should be > 70% for clear activities
- [ ] **Monitor latency**: Should be < 100ms per prediction
- [ ] **Test battery**: Monitor drain over 1 hour

### 5. Troubleshooting Guide

**Problem**: Wrong predictions (e.g., always predicts "Flat Walk")
- ✅ Check normalization parameters are correct
- ✅ Verify sensor data order [Gx, Gy, Gz, Ax, Ay, Az]
- ✅ Confirm phone placement (center/waist)
- ✅ Test with actual walking, not stationary

**Problem**: Predictions change randomly
- ✅ Check sampling rate (should be ~100 Hz)
- ✅ Verify window size is exactly 200 samples
- ✅ Ensure no overlap in window creation

**Problem**: App crashes during inference
- ✅ Verify input shape [1, 200, 6]
- ✅ Check data type is float32
- ✅ Ensure model file loaded correctly

**Problem**: Low confidence scores (< 50%)
- ✅ Activity might be ambiguous
- ✅ Phone placement might be wrong
- ✅ Sensor calibration might be needed

### Expected Performance
- **Latency**: 50-100ms per prediction
- **Battery**: ~5-10% per hour
- **Accuracy**: 85-90% (matches test set)
- **Confidence**: 70-95% for clear activities

---

## 📦 Part 9: Deployment Checklist

### Before Release
- [ ] Convert best model to TFLite
- [ ] Extract preprocessing parameters
- [ ] Test on multiple devices
- [ ] Optimize battery usage
- [ ] Add error handling
- [ ] Create user guide

### App Store Requirements
- [ ] Privacy policy (sensor data usage)
- [ ] Permissions explanation
- [ ] Screenshots
- [ ] App description

---

## 🔍 Part 10: Debugging Tips

### Common Issues

**1. Wrong predictions**:
- Check normalization parameters
- Verify sensor axis orientation
- Ensure 100 Hz sampling rate

**2. High latency**:
- Use quantized model
- Reduce window size (150 samples = 1.5s)
- Run on background thread

**3. Crashes**:
- Check input shape: `[1, 200, 6]`
- Verify data type: `float32`
- Handle null sensors gracefully

### Logging
```dart
// Add to preprocessing
print('Raw sample: ${sample}');
print('Normalized: ${normalized}');
print('Model input shape: ${input.shape}');
print('Model output: ${output}');
```

---

## 📚 Part 11: Additional Resources

### Flutter Packages
- `tflite_flutter`: https://pub.dev/packages/tflite_flutter
- `sensors_plus`: https://pub.dev/packages/sensors_plus
- `fl_chart`: https://pub.dev/packages/fl_chart

### TensorFlow Lite
- Converter guide: https://www.tensorflow.org/lite/convert
- Optimization: https://www.tensorflow.org/lite/performance/best_practices

### Example Apps
- TFLite Flutter examples: https://github.com/tensorflow/flutter-tflite

---

## ✅ Quick Start Checklist

1. [ ] Convert model to TFLite
2. [ ] Extract preprocessing params
3. [ ] Create Flutter project
4. [ ] Add dependencies
5. [ ] Implement sensor service
6. [ ] Implement preprocessing
7. [ ] Implement model inference
8. [ ] Build UI
9. [ ] Test on device
10. [ ] Deploy!

---

## 🎯 Expected Timeline

- **Day 1**: Model conversion + Flutter setup
- **Day 2**: Sensor collection + preprocessing
- **Day 3**: Model integration + basic UI
- **Day 4**: Testing + optimization
- **Day 5**: Polish + deployment

**Total**: ~5 days for MVP

Good luck with your mobile app! 🚀
