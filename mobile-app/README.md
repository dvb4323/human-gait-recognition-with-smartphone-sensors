# Gait Recognition Mobile App

Real-time gait activity classification using smartphone sensors and TensorFlow Lite models.

## Features

- ✅ Real-time activity recognition (Flat Walk, Up/Down Stairs, Up/Down Slope)
- ✅ Swappable ML models (GRU, 1D CNN, CNN-LSTM)
- ✅ Live confidence scores and probability visualization
- ✅ 100 Hz sensor data collection
- ✅ Dark mode UI

## Project Structure

```
mobile-app/
├── src/
│   ├── components/         # UI components
│   │   ├── ActivityDisplay.tsx
│   │   └── ConfidenceBars.tsx
│   ├── services/          # Core services
│   │   ├── SensorService.ts
│   │   └── InferenceService.ts
│   ├── screens/           # App screens
│   │   └── HomeScreen.tsx
│   ├── utils/             # Utilities
│   │   ├── constants.ts
│   │   └── preprocessing.ts
│   └── assets/            # Models and resources
│       └── models/        # TFLite models (to be added)
├── android/               # Android configuration
├── App.tsx                # Main app component
└── package.json           # Dependencies
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd mobile-app
npm install
```

### 2. Add TensorFlow Lite Models

Copy the converted `.tflite` models to `src/assets/models/`:

```
src/assets/models/
├── gait_gru_model.tflite
├── gait_cnn_model.tflite
└── gait_cnn_lstm_model.tflite
```

**Note**: TensorFlow.js models need to be converted from `.tflite` format. Use the `tensorflowjs_converter` tool.

### 3. Android Setup

```bash
npx react-native run-android
```

### 4. Build APK

```bash
cd android
./gradlew assembleRelease
```

The APK will be located at:
`android/app/build/outputs/apk/release/app-release.apk`

## Usage

1. **Select Model**: Choose between GRU, 1D CNN, or CNN-LSTM
2. **Start Monitoring**: Tap "Start Monitoring" button
3. **Perform Activity**: Place phone in pocket and walk/climb stairs
4. **View Results**: See real-time activity predictions and confidence scores

## Model Information

| Model | Size | Accuracy | Speed |
|-------|------|----------|-------|
| GRU | 209 KB | 91.6% | Medium |
| 1D CNN | 351 KB | 90.3% | Fast |
| CNN-LSTM | 379 KB | ~90% | Slow |

## Requirements

- Node.js >= 18
- React Native 0.73
- Android SDK (for Android builds)
- Physical device recommended (sensors work better than emulator)

## Troubleshooting

### Sensors not working
- Ensure app has sensor permissions
- Use a physical device (emulator sensors are unreliable)

### Model loading fails
- Check that `.tflite` models are in `src/assets/models/`
- Verify models are converted to TensorFlow.js format

### Build errors
- Run `npm install` to ensure all dependencies are installed
- Clear cache: `npx react-native start --reset-cache`

## Next Steps

- [ ] Convert `.tflite` models to TensorFlow.js format
- [ ] Test on physical Android device
- [ ] Add activity history tracking
- [ ] Implement model performance metrics
- [ ] Add export functionality for predictions

## License

MIT
