/**
 * Home Screen - Main screen for gait recognition (Simplified for Expo)
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
import ActivityDisplay from '../components/ActivityDisplay';
import ConfidenceBars from '../components/ConfidenceBars';
import {SensorService} from '../services/SensorService';
import {InferenceService} from '../services/InferenceService';
import {SlidingWindowBuffer, SensorSample} from '../utils/preprocessing';
import {ACTIVITY_NAMES} from '../utils/constants';

const HomeScreen: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentActivity, setCurrentActivity] = useState('Waiting...');
  const [confidence, setConfidence] = useState(0);
  const [probabilities, setProbabilities] = useState([0.2, 0.2, 0.2, 0.2, 0.2]);
  const [selectedModel, setSelectedModel] = useState('cnn'); // Start with CNN
  const [bufferSize, setBufferSize] = useState(0);

  const sensorService = useRef(new SensorService()).current;
  const inferenceService = useRef(new InferenceService()).current;
  const windowBuffer = useRef(new SlidingWindowBuffer(200, 0.5)).current;

  useEffect(() => {
    // Initialize model
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
      console.log(`Model ${selectedModel.toUpperCase()} loaded!`);
    } catch (error) {
      console.error('Error loading model:', error);
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

      {/* Model Selector - Simple Buttons */}
      <View style={styles.modelSelector}>
        <Text style={styles.label}>Model:</Text>
        <View style={styles.modelButtons}>
          <TouchableOpacity
            style={[
              styles.modelButton,
              selectedModel === 'gru' && styles.modelButtonActive,
            ]}
            onPress={() => handleModelChange('gru')}
            disabled={isRunning}>
            <Text style={styles.modelButtonText}>GRU</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.modelButton,
              selectedModel === 'cnn' && styles.modelButtonActive,
            ]}
            onPress={() => handleModelChange('cnn')}
            disabled={isRunning}>
            <Text style={styles.modelButtonText}>CNN</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.modelButton,
              selectedModel === 'cnn_lstm' && styles.modelButtonActive,
            ]}
            onPress={() => handleModelChange('cnn_lstm')}
            disabled={isRunning}>
            <Text style={styles.modelButtonText}>CNN-LSTM</Text>
          </TouchableOpacity>
        </View>
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
        <Text style={styles.statusText}>
          Model: {selectedModel.toUpperCase()}
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
        <Text style={styles.infoText}>
          🔬 Currently using mock predictions for testing
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
    marginTop: 20,
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
    marginBottom: 10,
  },
  modelButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  modelButton: {
    flex: 1,
    backgroundColor: '#0f1419',
    borderRadius: 8,
    padding: 12,
    marginHorizontal: 4,
    borderWidth: 2,
    borderColor: '#2a2a3e',
  },
  modelButtonActive: {
    borderColor: '#4CAF50',
    backgroundColor: '#1a3a1a',
  },
  modelButtonText: {
    color: '#ffffff',
    textAlign: 'center',
    fontWeight: 'bold',
    fontSize: 12,
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

export default HomeScreen;
