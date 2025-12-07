/**
 * Sensor Service - Collects accelerometer and gyroscope data (Expo Version)
 */

import {Accelerometer, Gyroscope} from 'expo-sensors';
import {SensorSample} from '../utils/preprocessing';
import {SENSOR_CONFIG} from '../utils/constants';

export class SensorService {
  private accelSubscription: any = null;
  private gyroSubscription: any = null;
  private currentAccel = {x: 0, y: 0, z: 0};
  private currentGyro = {x: 0, y: 0, z: 0};
  private onSampleCallback: ((sample: SensorSample) => void) | null = null;

  constructor() {
    // Set update interval to 100 Hz (10ms)
    Accelerometer.setUpdateInterval(SENSOR_CONFIG.UPDATE_INTERVAL);
    Gyroscope.setUpdateInterval(SENSOR_CONFIG.UPDATE_INTERVAL);
  }

  /**
   * Start collecting sensor data
   */
  start(onSample: (sample: SensorSample) => void): void {
    this.onSampleCallback = onSample;

    // Subscribe to accelerometer
    this.accelSubscription = Accelerometer.addListener(({x, y, z}) => {
      this.currentAccel = {x, y, z};
      this.emitSample();
    });

    // Subscribe to gyroscope
    this.gyroSubscription = Gyroscope.addListener(({x, y, z}) => {
      this.currentGyro = {x, y, z};
      this.emitSample();
    });
  }

  /**
   * Stop collecting sensor data
   */
  stop(): void {
    if (this.accelSubscription) {
      this.accelSubscription.remove();
      this.accelSubscription = null;
    }
    if (this.gyroSubscription) {
      this.gyroSubscription.remove();
      this.gyroSubscription = null;
    }
    this.onSampleCallback = null;
  }

  /**
   * Emit combined sensor sample
   */
  private emitSample(): void {
    if (this.onSampleCallback) {
      const sample: SensorSample = {
        Gx: this.currentGyro.x,
        Gy: this.currentGyro.y,
        Gz: this.currentGyro.z,
        Ax: this.currentAccel.x,
        Ay: this.currentAccel.y,
        Az: this.currentAccel.z,
      };
      this.onSampleCallback(sample);
    }
  }

  /**
   * Check if sensors are running
   */
  isRunning(): boolean {
    return this.accelSubscription !== null && this.gyroSubscription !== null;
  }
}
