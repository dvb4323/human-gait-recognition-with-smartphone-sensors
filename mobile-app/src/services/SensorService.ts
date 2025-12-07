/**
 * Sensor Service - Collects accelerometer and gyroscope data
 */

import {
  accelerometer,
  gyroscope,
  setUpdateIntervalForType,
  SensorTypes,
} from 'react-native-sensors';
import {Subscription} from 'rxjs';
import {SensorSample} from '../utils/preprocessing';
import {SENSOR_CONFIG} from '../utils/constants';

export class SensorService {
  private accelSubscription: Subscription | null = null;
  private gyroSubscription: Subscription | null = null;
  private currentAccel = {x: 0, y: 0, z: 0};
  private currentGyro = {x: 0, y: 0, z: 0};
  private onSampleCallback: ((sample: SensorSample) => void) | null = null;

  constructor() {
    // Set update interval to 100 Hz (10ms)
    setUpdateIntervalForType(SensorTypes.accelerometer, SENSOR_CONFIG.UPDATE_INTERVAL);
    setUpdateIntervalForType(SensorTypes.gyroscope, SENSOR_CONFIG.UPDATE_INTERVAL);
  }

  /**
   * Start collecting sensor data
   */
  start(onSample: (sample: SensorSample) => void): void {
    this.onSampleCallback = onSample;

    // Subscribe to accelerometer
    this.accelSubscription = accelerometer.subscribe(({x, y, z}) => {
      this.currentAccel = {x, y, z};
      this.emitSample();
    });

    // Subscribe to gyroscope
    this.gyroSubscription = gyroscope.subscribe(({x, y, z}) => {
      this.currentGyro = {x, y, z};
      this.emitSample();
    });
  }

  /**
   * Stop collecting sensor data
   */
  stop(): void {
    if (this.accelSubscription) {
      this.accelSubscription.unsubscribe();
      this.accelSubscription = null;
    }
    if (this.gyroSubscription) {
      this.gyroSubscription.unsubscribe();
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
