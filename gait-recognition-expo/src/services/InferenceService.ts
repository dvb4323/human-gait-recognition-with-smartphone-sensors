/**
 * Inference Service - Real TFLite Integration
 */

import {loadTensorflowModel} from 'react-native-fast-tflite';
import {ACTIVITY_NAMES} from '../utils/constants';

export interface Prediction {
  activity: string;
  confidence: number;
  probabilities: number[];
}

export class InferenceService {
  private model: any = null;
  private modelName: string = 'cnn'; // Changed to CNN (no recurrent ops)
  private isReady: boolean = false;

  async initialize(): Promise<void> {
    console.log('TFLite inference service initialized');
  }

  async loadModel(modelName: string): Promise<void> {
    try {
      this.modelName = modelName;
      
      // Dispose previous model
      if (this.model) {
        this.model = null;
      }

      // Map model names to static requires
      // React Native bundler doesn't support dynamic requires
      const modelAssets: {[key: string]: any} = {
        'gru': require('../../assets/models/gait_gru_model.tflite'),
        'cnn': require('../../assets/models/gait_cnn_model.tflite'),
        'cnn_lstm': require('../../assets/models/gait_cnn_lstm_model.tflite'),
      };

      const modelAsset = modelAssets[modelName];
      if (!modelAsset) {
        throw new Error(`Model ${modelName} not found`);
      }

      // Load model from assets
      this.model = await loadTensorflowModel(modelAsset);
      
      this.isReady = true;
      console.log(`✅ Model ${modelName} loaded successfully`);
    } catch (error) {
      console.error('❌ Error loading model:', error);
      this.isReady = false;
      throw error;
    }
  }

  async predict(window: number[][][]): Promise<Prediction> {
    if (!this.model || !this.isReady) {
      throw new Error('Model not loaded');
    }

    try {
      // Flatten window to 1D Float32Array
      // Input shape: [1, 200, 6] -> [1200]
      const flatInput = new Float32Array(window[0].flat());
      
      // Run inference
      const outputs = await this.model.run([flatInput]);
      const probabilities = Array.from(outputs[0]) as number[];
      
      // Get predicted class
      const predictedClass = probabilities.indexOf(Math.max(...probabilities));
      const confidence = probabilities[predictedClass];

      console.log(`🎯 Prediction: ${ACTIVITY_NAMES[predictedClass]} (${(confidence * 100).toFixed(1)}%)`);

      return {
        activity: ACTIVITY_NAMES[predictedClass],
        confidence: confidence,
        probabilities: probabilities,
      };
    } catch (error) {
      console.error('❌ Inference error:', error);
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
      this.model = null;
    }
    this.isReady = false;
  }
}
