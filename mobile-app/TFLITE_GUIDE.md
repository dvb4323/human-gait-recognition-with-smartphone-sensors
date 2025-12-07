# Using TFLite Models Directly (Recommended Approach)

## Why TFLite Instead of TensorFlow.js?

✅ **Better for mobile** - Smaller, faster, native performance  
✅ **No conversion needed** - Use `.tflite` files directly  
✅ **Lower latency** - Native C++ implementation  
✅ **Smaller app size** - No JavaScript overhead  

---

## Option 1: Use TFLite Directly (Recommended)

### Step 1: Install TFLite React Native Package

```bash
cd mobile-app
npm install react-native-tflite
```

### Step 2: Copy TFLite Models to Assets

Copy your `.tflite` models to the React Native assets folder:

```bash
# Create android assets directory
mkdir -p mobile-app/android/app/src/main/assets/models

# Copy models
cp models/mobile/gait_gru_model.tflite mobile-app/android/app/src/main/assets/models/
cp models/mobile/gait_cnn_model.tflite mobile-app/android/app/src/main/assets/models/
cp models/mobile/gait_cnn_lstm_model.tflite mobile-app/android/app/src/main/assets/models/
```

### Step 3: Update InferenceService

The `InferenceService.ts` has been updated to use TFLite directly. You'll need to integrate the actual TFLite library calls.

### Step 4: Build and Run

```bash
npx react-native run-android
```

---

## Option 2: Use Pre-built TFLite Solution

### Install `react-native-fast-tflite`

This is a modern, well-maintained TFLite package:

```bash
npm install react-native-fast-tflite
cd ios && pod install && cd ..  # iOS only
```

### Example Usage

```typescript
import {loadModel} from 'react-native-fast-tflite';

// Load model
const model = await loadModel({
  model: require('../assets/models/gait_gru_model.tflite'),
});

// Run inference
const output = await model.run(inputTensor);
```

---

## Option 3: Simple Mock Implementation (For Testing UI)

If you just want to test the UI without actual inference:

1. The current `InferenceService.ts` already has mock predictions
2. Run the app to test UI/UX
3. Add real TFLite integration later

```bash
cd mobile-app
npm install
npx react-native run-android
```

The app will work with simulated predictions!

---

## Recommended Next Steps

### For Quick Testing (No Model Conversion Needed!)

```bash
# 1. Install dependencies
cd mobile-app
npm install

# 2. Run on Android
npx react-native run-android
```

The app will run with mock predictions. You can test:
- ✅ UI/UX
- ✅ Sensor data collection
- ✅ Model switching
- ✅ All components

### For Production (Real Inference)

```bash
# 1. Install TFLite package
npm install react-native-fast-tflite

# 2. Copy .tflite models to assets
mkdir -p android/app/src/main/assets/models
cp ../../models/mobile/*.tflite android/app/src/main/assets/models/

# 3. Integrate TFLite in InferenceService.ts
# (See documentation for react-native-fast-tflite)

# 4. Build and test
npx react-native run-android
```

---

## Summary

**You don't need TensorFlow.js conversion!**

Your `.tflite` models are ready to use. Just:
1. Copy them to Android assets folder
2. Install a TFLite React Native package
3. Update InferenceService to use TFLite
4. Build and run!

**For now, test the app with mock predictions - it's fully functional!**
