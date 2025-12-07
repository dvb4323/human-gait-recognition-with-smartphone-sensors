# EAS Development Build - Step-by-Step Guide

## 🎯 Goal
Create a custom development build with `react-native-fast-tflite` support for real TFLite inference.

---

## 📋 Prerequisites

- ✅ Expo account (free)
- ✅ Android device for testing
- ✅ ~20 minutes for build time

---

## 🚀 Step 1: Install EAS CLI

```bash
npm install -g eas-cli
```

---

## 🔐 Step 2: Login to Expo

```bash
eas login
```

If you don't have an account:
```bash
eas register
```

---

## ⚙️ Step 3: Configure EAS Build

```bash
cd gait-recognition-expo
eas build:configure
```

This will create `eas.json` configuration file.

---

## 📝 Step 4: Update eas.json

The file should look like this:

```json
{
  "cli": {
    "version": ">= 5.0.0"
  },
  "build": {
    "development": {
      "developmentClient": true,
      "distribution": "internal",
      "android": {
        "buildType": "apk"
      }
    },
    "preview": {
      "distribution": "internal",
      "android": {
        "buildType": "apk"
      }
    },
    "production": {
      "android": {
        "buildType": "apk"
      }
    }
  },
  "submit": {
    "production": {}
  }
}
```

---

## 📦 Step 5: Install Development Client

```bash
npx expo install expo-dev-client
```

---

## 📱 Step 6: Copy TFLite Models to Assets

```bash
# Create assets directory if it doesn't exist
mkdir -p assets/models

# Copy your TFLite models
cp ../models/mobile/gait_gru_model.tflite assets/models/
cp ../models/mobile/gait_cnn_model.tflite assets/models/
cp ../models/mobile/gait_cnn_lstm_model.tflite assets/models/
```

---

## 🔧 Step 7: Update app.json

Add the following to `app.json`:

```json
{
  "expo": {
    "name": "Gait Recognition",
    "slug": "gait-recognition-expo",
    "version": "1.0.0",
    "orientation": "portrait",
    "icon": "./assets/icon.png",
    "userInterfaceStyle": "dark",
    "splash": {
      "image": "./assets/splash.png",
      "resizeMode": "contain",
      "backgroundColor": "#1a1a2e"
    },
    "assetBundlePatterns": [
      "assets/**/*"
    ],
    "android": {
      "package": "com.gaitrecognition.app",
      "adaptiveIcon": {
        "foregroundImage": "./assets/adaptive-icon.png",
        "backgroundColor": "#1a1a2e"
      }
    },
    "plugins": [
      [
        "expo-build-properties",
        {
          "android": {
            "minSdkVersion": 24
          }
        }
      ]
    ]
  }
}
```

---

## 🏗️ Step 8: Start the Build

```bash
eas build --profile development --platform android
```

**What happens:**
1. EAS uploads your code to their servers
2. Builds a custom APK with native modules
3. Takes ~15-20 minutes
4. Gives you a download link

**Output:**
```
✔ Build finished
📱 Install the build on your Android device:
   https://expo.dev/artifacts/eas/[your-build-id].apk
```

---

## 📲 Step 9: Install Development Build on Device

**Option A: Direct Download**
1. Open the link on your Android phone
2. Download the APK
3. Install it (allow installation from unknown sources)

**Option B: QR Code**
1. EAS will show a QR code
2. Scan with your phone camera
3. Download and install

---

## 🔄 Step 10: Update InferenceService for Real TFLite

Now that you have a development build, update the InferenceService:

```typescript
/**
 * Inference Service - Real TFLite Integration
 */

import {TensorflowModel} from 'react-native-fast-tflite';
import {ACTIVITY_NAMES} from '../utils/constants';

export interface Prediction {
  activity: string;
  confidence: number;
  probabilities: number[];
}

export class InferenceService {
  private model: TensorflowModel | null = null;
  private modelName: string = 'gru';
  private isReady: boolean = false;

  async initialize(): Promise<void> {
    console.log('TFLite inference service initialized');
  }

  async loadModel(modelName: string): Promise<void> {
    try {
      this.modelName = modelName;
      
      // Dispose previous model
      if (this.model) {
        this.model.dispose();
      }

      // Load model from assets
      const modelAsset = require(`../../assets/models/gait_${modelName}_model.tflite`);
      this.model = await TensorflowModel.create(modelAsset);
      
      this.isReady = true;
      console.log(`Model ${modelName} loaded successfully`);
    } catch (error) {
      console.error('Error loading model:', error);
      this.isReady = false;
      throw error;
    }
  }

  async predict(window: number[][][]): Promise<Prediction> {
    if (!this.model || !this.isReady) {
      throw new Error('Model not loaded');
    }

    try {
      // Flatten window to 1D array
      const flatInput = new Float32Array(window[0].flat());
      
      // Run inference
      const outputs = this.model.run([flatInput]);
      const probabilities = Array.from(outputs[0]);
      
      // Get predicted class
      const predictedClass = probabilities.indexOf(Math.max(...probabilities));
      const confidence = probabilities[predictedClass];

      return {
        activity: ACTIVITY_NAMES[predictedClass],
        confidence: confidence,
        probabilities: probabilities,
      };
    } catch (error) {
      console.error('Inference error:', error);
      throw error;
    }
  }

  getCurrentModel(): string {
    return this.modelName;
  }

  isModelReady(): boolean {
    return this.isReady;
  }

  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.isReady = false;
  }
}
```

---

## ▶️ Step 11: Run the App

After installing the development build:

```bash
# Start the development server
npx expo start --dev-client

# Scan QR code with your development build app
```

**Important:** Use the **development build app** you installed, NOT Expo Go!

---

## 🧪 Step 12: Test Real Inference

1. Open the app on your device
2. Tap "Start Monitoring"
3. Walk on flat ground → Should predict "Flat Walk"
4. Walk up stairs → Should predict "Up Stairs"
5. Walk down stairs → Should predict "Down Stairs"
6. Walk up slope → Should predict "Up Slope"
7. Walk down slope → Should predict "Down Slope"

---

## 🎯 Expected Results

- **Inference Speed:** 20-50ms per prediction
- **Accuracy:** ~90-92% (matching training)
- **Update Frequency:** ~1 prediction per second
- **Battery Usage:** ~5% per hour

---

## 🔧 Troubleshooting

### Build Fails

**Error:** "Build failed"

**Solution:**
```bash
# Check build logs
eas build:list

# Try again with verbose logging
eas build --profile development --platform android --clear-cache
```

### Model Not Loading

**Error:** "Cannot find module"

**Solution:**
- Ensure models are in `assets/models/`
- Check file names match: `gait_gru_model.tflite`, etc.
- Verify `assetBundlePatterns` in app.json

### App Crashes on Inference

**Error:** App crashes when tapping "Start"

**Solution:**
- Check model input shape matches [1, 200, 6]
- Verify preprocessing outputs correct format
- Check console logs for errors

---

## 📊 Build Status Tracking

Monitor your build:
```bash
# List all builds
eas build:list

# View specific build
eas build:view [build-id]
```

Or visit: https://expo.dev/accounts/[your-account]/projects/gait-recognition-expo/builds

---

## 🚀 Production Build (Later)

Once everything works, create a production APK:

```bash
eas build --profile production --platform android
```

This creates a signed, optimized APK for distribution.

---

## 💡 Tips

1. **First build takes longest** (~20 min), subsequent builds are faster
2. **Keep development build installed** - reuse it for testing
3. **Use `--dev-client` flag** when running `expo start`
4. **Check build queue** - builds may queue if many users building

---

## 📝 Quick Command Reference

```bash
# Login
eas login

# Configure
eas build:configure

# Build development
eas build --profile development --platform android

# Run app
npx expo start --dev-client

# List builds
eas build:list

# Production build
eas build --profile production --platform android
```

---

## ✅ Success Checklist

- [ ] EAS CLI installed
- [ ] Logged into Expo account
- [ ] `eas.json` configured
- [ ] Models copied to `assets/models/`
- [ ] `app.json` updated
- [ ] Development build started
- [ ] APK downloaded and installed
- [ ] InferenceService updated
- [ ] App running with `--dev-client`
- [ ] Real inference working!

---

**Ready to start?** Run the first command:

```bash
npm install -g eas-cli
eas login
```

Let me know when you're ready for the next step! 🚀
