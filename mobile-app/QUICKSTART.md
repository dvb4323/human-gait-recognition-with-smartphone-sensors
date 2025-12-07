# Quick Start Guide - Gait Recognition App

## Problem: React Native Setup is Complex

Setting up React Native with Android from scratch requires many configuration files. Here are **3 practical solutions**:

---

## ✅ **Option 1: Use Expo (EASIEST - Recommended)**

Expo is a framework that simplifies React Native development.

### Step 1: Install Expo CLI

```bash
npm install -g expo-cli
```

### Step 2: Create Expo Project

```bash
cd ..
npx create-expo-app gait-recognition-expo
cd gait-recognition-expo
```

### Step 3: Copy Our Code

```bash
# Copy our source files
cp -r ../mobile-app/src ./
cp ../mobile-app/App.tsx ./
```

### Step 4: Install Dependencies

```bash
npm install react-native-sensors
npm install react-native-picker-select
```

### Step 5: Run on Device

```bash
# Install Expo Go app on your Android phone
# Then run:
npx expo start

# Scan QR code with Expo Go app
```

**Advantages:**
- ✅ No Android Studio needed
- ✅ Works immediately
- ✅ Easy to test on device
- ✅ Can build APK with `eas build`

---

## ✅ **Option 2: Use Existing React Native Template**

### Step 1: Create New Project with Template

```bash
cd ..
npx react-native init GaitRecognition
cd GaitRecognition
```

### Step 2: Replace Files

```bash
# Copy our source code
rm -rf src
cp -r ../mobile-app/src ./
cp ../mobile-app/App.tsx ./App.tsx

# Copy package.json dependencies
```

### Step 3: Install Dependencies

```bash
npm install react-native-sensors
npm install react-native-picker-select  
npm install react-native-fs
```

### Step 4: Run

```bash
npx react-native run-android
```

---

## ✅ **Option 3: Web Demo First (Test Without Mobile)**

Test the logic in a web browser first, then port to mobile later.

### Create Simple Web Version

```bash
cd ../mobile-app
npm install
npm install -g serve

# Create a simple HTML file to test
```

I can create a web-based demo that:
- Shows the UI
- Simulates sensor data
- Tests the preprocessing logic
- Demonstrates model switching

---

## 🎯 **My Recommendation**

**Use Option 1 (Expo)** because:
1. ✅ Fastest to get running (5 minutes)
2. ✅ No Android Studio setup needed
3. ✅ Easy to test on real device
4. ✅ Can build APK later with `eas build`

### Quick Expo Setup (Copy-Paste)

```bash
# 1. Go back to parent directory
cd ..

# 2. Create Expo app
npx create-expo-app gait-recognition-expo --template blank-typescript

# 3. Go into new project
cd gait-recognition-expo

# 4. Install dependencies
npm install react-native-sensors @react-native-picker/picker

# 5. Copy our code
# (I'll provide updated files for Expo)

# 6. Run
npx expo start
```

---

## What Would You Like to Do?

**A)** Use Expo (I'll help set it up - 5 minutes)  
**B)** Use React Native CLI (I'll provide Android config files)  
**C)** Create web demo first (test in browser)  
**D)** Something else?

Let me know and I'll help you get it running! 🚀
