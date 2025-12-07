/**
 * Home Screen - Main screen for gait recognition
 */

import React, {useState, useEffect, useRef} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
} from 'react-native';
import RNPickerSelect from 'react-native-picker-select';
import ActivityDisplay from '../components/ActivityDisplay';
import ConfidenceBars from '../components/ConfidenceBars';
import {SensorService} from '../services/SensorService';
import {InferenceService} from '../services/InferenceService';
import {SlidingWindowBuffer, SensorSample} from '../utils/preprocessing';
import {MODEL_OPTIONS, ACTIVITY_NAMES} from '../utils/constants';

const HomeScreen: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentActivity, setCurrentActivity] = useState('Waiting...');
  const [confidence, setConfidence] = useState(0);
  const [probabilities, setProbabilities] = useState([0.2, 0.2, 0.2, 0.2, 0.2]);
  const [selectedModel, setSelectedModel] = useState('gru');
  const [bufferSize, setBufferSize] = useState(0);

  const sensorService = useRef(new SensorService()).current;
  const inferenceService = useRef(new InferenceService()).current;
  const windowBuffer = useRef(new SlidingWindowBuffer(200, 0.5)).current;

  useEffect(() => {
    // Initialize TensorFlow and load model
    initializeModel();

    return () => {
      // Cleanup
      sensorService.stop();
      inferenceService.dispose();
    };
  }, []);

  const initializeModel = async () => {
    try {
      await inferenceService.initialize();
      await inferenceService.loadModel(selectedModel);
      Alert.alert('Success', `Model ${selectedModel.toUpperCase()} loaded!`);
    } catch (error) {
      Alert.alert('Error', 'Failed to load model. Check console for details.');
      console.error(error);
    }
  };

  const handleModelChange = async (model: string) => {
    if (isRunning) {
      Alert.alert('Warning', 'Stop monitoring before changing models');
      return;
    }

    setSelectedModel(model);
    try {
      await inferenceService.loadModel(model);
      Alert.alert('Success', `Switched to ${model.toUpperCase()} model`);
    } catch (error) {
      Alert.alert('Error', 'Failed to load model');
      console.error(error);
    }
  };

  const handleSensorSample = async (sample: SensorSample) => {
    // Add sample to buffer
    const ready = windowBuffer.addSample(sample);
    setBufferSize(windowBuffer.getBufferSize());

    // Run inference if window is ready
    if (ready) {
      try {
        const window = windowBuffer.getNormalizedWindow();
        const prediction = await inferenceService.predict(window);

        setCurrentActivity(prediction.activity);
        setConfidence(prediction.confidence);
        setProbabilities(prediction.probabilities);
      } catch (error) {
        console.error('Inference error:', error);
      }
    }
  };

  const startMonitoring = () => {
    if (!inferenceService.isModelReady()) {
      Alert.alert('Error', 'Model not loaded yet');
      return;
    }

    windowBuffer.reset();
    sensorService.start(handleSensorSample);
    setIsRunning(true);
    setCurrentActivity('Collecting data...');
  };

  const stopMonitoring = () => {
    sensorService.stop();
    setIsRunning(false);
    setCurrentActivity('Stopped');
    setBufferSize(0);
  };

  const activityIndex = ACTIVITY_NAMES.indexOf(currentActivity);

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>🚶 Gait Recognition</Text>
        <Text style={styles.subtitle}>Real-time Activity Classification</Text>
      </View>

      {/* Model Selector */}
      <View style={styles.modelSelector}>
        <Text style={styles.label}>Model:</Text>
        <RNPickerSelect
          value={selectedModel}
          onValueChange={handleModelChange}
          items={MODEL_OPTIONS}
          style={pickerSelectStyles}
          disabled={isRunning}
        />
      </View>

      {/* Activity Display */}
      <ActivityDisplay
        activity={currentActivity}
        confidence={confidence}
        activityIndex={activityIndex}
      />

      {/* Confidence Bars */}
      {probabilities.some(p => p > 0) && (
        <ConfidenceBars probabilities={probabilities} />
      )}

      {/* Buffer Status */}
      <View style={styles.statusContainer}>
        <Text style={styles.statusText}>
          Buffer: {bufferSize} / 200 samples
        </Text>
        <Text style={styles.statusText}>
          Status: {isRunning ? '🟢 Running' : '🔴 Stopped'}
        </Text>
      </View>

      {/* Control Buttons */}
      <View style={styles.buttonContainer}>
        {!isRunning ? (
          <TouchableOpacity style={styles.startButton} onPress={startMonitoring}>
            <Text style={styles.buttonText}>▶️ Start Monitoring</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity style={styles.stopButton} onPress={stopMonitoring}>
            <Text style={styles.buttonText}>⏹️ Stop Monitoring</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Info */}
      <View style={styles.infoContainer}>
        <Text style={styles.infoText}>
          📱 Place phone in your pocket and start walking
        </Text>
        <Text style={styles.infoText}>
          ⏱️ Predictions update every second
        </Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 16,
    color: '#b0b0b0',
  },
  modelSelector: {
    backgroundColor: '#16213e',
    borderRadius: 10,
    padding: 15,
    marginBottom: 10,
  },
  label: {
    fontSize: 16,
    color: '#ffffff',
    marginBottom: 8,
  },
  statusContainer: {
    backgroundColor: '#16213e',
    borderRadius: 10,
    padding: 15,
    marginVertical: 10,
  },
  statusText: {
    fontSize: 14,
    color: '#b0b0b0',
    marginBottom: 5,
  },
  buttonContainer: {
    marginVertical: 20,
  },
  startButton: {
    backgroundColor: '#4CAF50',
    borderRadius: 15,
    padding: 18,
    alignItems: 'center',
    shadowColor: '#4CAF50',
    shadowOffset: {width: 0, height: 4},
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 8,
  },
  stopButton: {
    backgroundColor: '#F44336',
    borderRadius: 15,
    padding: 18,
    alignItems: 'center',
    shadowColor: '#F44336',
    shadowOffset: {width: 0, height: 4},
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 8,
  },
  buttonText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  infoContainer: {
    backgroundColor: '#16213e',
    borderRadius: 10,
    padding: 15,
    marginBottom: 30,
  },
  infoText: {
    fontSize: 14,
    color: '#b0b0b0',
    marginBottom: 8,
  },
});

const pickerSelectStyles = StyleSheet.create({
  inputIOS: {
    fontSize: 16,
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: '#4CAF50',
    borderRadius: 8,
    color: '#ffffff',
    backgroundColor: '#0f1419',
  },
  inputAndroid: {
    fontSize: 16,
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderWidth: 1,
    borderColor: '#4CAF50',
    borderRadius: 8,
    color: '#ffffff',
    backgroundColor: '#0f1419',
  },
});

export default HomeScreen;
