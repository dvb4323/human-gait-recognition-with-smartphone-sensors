# Mobile App Setup Guide

## Overview

The React Native mobile app is now ready in the `mobile-app/` directory. This guide will help you complete the setup and build the APK.

---

## Step 1: Convert TFLite Models to TensorFlow.js

Your models are currently in `.tflite` format, but TensorFlow.js requires a different format.

### Option A: Use TensorFlow.js Converter (Recommended)

```bash
# Install converter
pip install tensorflowjs

# Convert each model
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    models/mobile/gait_gru_model.tflite \
    mobile-app/src/assets/models/gru/

tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    models/mobile/gait_cnn_model.tflite \
    mobile-app/src/assets/models/cnn/

tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    models/mobile/gait_cnn_lstm_model.tflite \
    mobile-app/src/assets/models/cnn_lstm/
```

### Option B: Use Original Keras Models

Alternatively, convert directly from the original `.h5` files:

```bash
# Convert from Keras models
tensorflowjs_converter \
    --input_format=keras \
    results/lstm_20251206_170855/best_model.h5 \
    mobile-app/src/assets/models/gru/

tensorflowjs_converter \
    --input_format=keras \
    results/1d_cnn_20251206_154352/best_model.h5 \
    mobile-app/src/assets/models/cnn/

tensorflowjs_converter \
    --input_format=keras \
    results/cnn_lstm_20251206_154643/best_model.h5 \
    mobile-app/src/assets/models/cnn_lstm/
```

---

## Step 2: Install Dependencies

```bash
cd mobile-app
npm install
```

This will install:
- React Native 0.73
- TensorFlow.js for React Native
- Sensor libraries
- UI components

---

## Step 3: Setup Android Environment

### Prerequisites

1. **Install Android Studio**
2. **Install Java JDK 11 or higher**
3. **Set environment variables**:

```bash
# Add to your PATH
ANDROID_HOME=C:\Users\YourName\AppData\Local\Android\Sdk
JAVA_HOME=C:\Program Files\Java\jdk-11
```

### Initialize Android Project

```bash
npx react-native init-android
```

---

## Step 4: Test on Device/Emulator

### Using Android Emulator

```bash
# Start Metro bundler
npx react-native start

# In another terminal, run Android
npx react-native run-android
```

### Using Physical Device

1. Enable **Developer Options** on your Android phone
2. Enable **USB Debugging**
3. Connect phone via USB
4. Run: `npx react-native run-android`

---

## Step 5: Build APK for Distribution

### Debug APK (for testing)

```bash
cd android
./gradlew assembleDebug
```

Output: `android/app/build/outputs/apk/debug/app-debug.apk`

### Release APK (for distribution)

1. **Generate signing key**:

```bash
keytool -genkeypair -v -storetype PKCS12 -keystore my-release-key.keystore -alias my-key-alias -keyalg RSA -keysize 2048 -validity 10000
```

2. **Configure signing** in `android/app/build.gradle`:

```gradle
android {
    ...
    signingConfigs {
        release {
            storeFile file('my-release-key.keystore')
            storePassword 'your-password'
            keyAlias 'my-key-alias'
            keyPassword 'your-password'
        }
    }
    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
            proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules.pro'
        }
    }
}
```

3. **Build release APK**:

```bash
cd android
./gradlew assembleRelease
```

Output: `android/app/build/outputs/apk/release/app-release.apk`

---

## Step 6: Install APK on Device

### Via USB

```bash
adb install android/app/build/outputs/apk/release/app-release.apk
```

### Via File Transfer

1. Copy APK to phone
2. Open file manager
3. Tap APK file
4. Allow installation from unknown sources
5. Install

---

## Troubleshooting

### Model Loading Fails

**Issue**: "Failed to load model"

**Solution**:
- Verify models are in `mobile-app/src/assets/models/`
- Check model format is TensorFlow.js (not .tflite)
- Ensure model files include both `.json` and `.bin` files

### Sensors Not Working

**Issue**: No sensor data collected

**Solution**:
- Use physical device (emulator sensors are unreliable)
- Grant sensor permissions in app settings
- Check phone has accelerometer and gyroscope

### Build Errors

**Issue**: Gradle build fails

**Solution**:
```bash
cd android
./gradlew clean
cd ..
npx react-native start --reset-cache
```

### Metro Bundler Issues

**Issue**: "Unable to resolve module"

**Solution**:
```bash
rm -rf node_modules
npm install
npx react-native start --reset-cache
```

---

## Next Steps

1. ✅ Convert models to TensorFlow.js format
2. ✅ Install dependencies
3. ✅ Test on device
4. ✅ Build APK
5. ⏭️ Test real-world gait recognition
6. ⏭️ Gather performance metrics
7. ⏭️ Iterate and improve

---

## File Structure

```
mobile-app/
├── src/
│   ├── assets/
│   │   └── models/
│   │       ├── gru/
│   │       │   ├── model.json
│   │       │   └── group1-shard1of1.bin
│   │       ├── cnn/
│   │       └── cnn_lstm/
│   ├── components/
│   ├── services/
│   ├── screens/
│   └── utils/
├── android/
├── package.json
└── README.md
```

---

## Performance Tips

1. **Use GRU model** for best balance of accuracy and speed
2. **Test in pocket** - place phone in pocket for realistic testing
3. **Calibrate** - walk on flat ground first to establish baseline
4. **Battery** - Expect ~5% battery per hour of continuous monitoring

---

## Support

For issues or questions:
1. Check console logs: `npx react-native log-android`
2. Review README.md in mobile-app directory
3. Check TensorFlow.js documentation
