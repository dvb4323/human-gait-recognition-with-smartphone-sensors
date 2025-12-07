/**
 * Constants for the Gait Recognition App
 */

export const ACTIVITY_NAMES = [
  'Flat Walk',
  'Up Stairs',
  'Down Stairs',
  'Up Slope',
  'Down Slope',
];

export const ACTIVITY_COLORS = [
  '#4CAF50', // Flat Walk - Green
  '#2196F3', // Up Stairs - Blue
  '#FF9800', // Down Stairs - Orange
  '#9C27B0', // Up Slope - Purple
  '#F44336', // Down Slope - Red
];

export const ACTIVITY_ICONS = [
  '🚶', // Flat Walk
  '⬆️', // Up Stairs
  '⬇️', // Down Stairs
  '⛰️', // Up Slope
  '🏔️', // Down Slope
];

export const MODEL_OPTIONS = [
  { label: 'GRU (Best Accuracy)', value: 'gru' },
  { label: '1D CNN (Faster)', value: 'cnn' },
  { label: 'CNN-LSTM (Hybrid)', value: 'cnn_lstm' },
];

export const SENSOR_CONFIG = {
  WINDOW_SIZE: 200,
  SAMPLING_RATE: 100, // Hz
  OVERLAP: 0.5,
  UPDATE_INTERVAL: 10, // ms (100 Hz)
};

export const PREPROCESSING_CONFIG = {
  NORMALIZATION: 'z-score',
  FEATURES: ['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az'],
};
