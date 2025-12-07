/**
 * Preprocessing utilities for sensor data
 * Ported from Python preprocessing module
 */

export interface SensorSample {
  Gx: number;
  Gy: number;
  Gz: number;
  Ax: number;
  Ay: number;
  Az: number;
}

/**
 * Normalize sensor data using z-score normalization
 */
export function normalizeSensorData(data: number[][]): number[][] {
  const numFeatures = data[0].length;
  const numSamples = data.length;

  // Calculate mean for each feature
  const means = new Array(numFeatures).fill(0);
  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      means[j] += data[i][j];
    }
  }
  means.forEach((_, idx) => {
    means[idx] /= numSamples;
  });

  // Calculate standard deviation for each feature
  const stds = new Array(numFeatures).fill(0);
  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      stds[j] += Math.pow(data[i][j] - means[j], 2);
    }
  }
  stds.forEach((_, idx) => {
    stds[idx] = Math.sqrt(stds[idx] / numSamples);
    // Avoid division by zero
    if (stds[idx] === 0) stds[idx] = 1;
  });

  // Normalize
  const normalized: number[][] = [];
  for (let i = 0; i < numSamples; i++) {
    const row: number[] = [];
    for (let j = 0; j < numFeatures; j++) {
      row.push((data[i][j] - means[j]) / stds[j]);
    }
    normalized.push(row);
  }

  return normalized;
}

/**
 * Convert SensorSample to array format [Gx, Gy, Gz, Ax, Ay, Az]
 */
export function sensorSampleToArray(sample: SensorSample): number[] {
  return [sample.Gx, sample.Gy, sample.Gz, sample.Ax, sample.Ay, sample.Az];
}

/**
 * Sliding window buffer for real-time inference
 */
export class SlidingWindowBuffer {
  private buffer: number[][] = [];
  private windowSize: number;
  private stepSize: number;
  private samplesSinceLastWindow: number = 0;

  constructor(windowSize: number = 200, overlap: number = 0.5) {
    this.windowSize = windowSize;
    this.stepSize = Math.floor(windowSize * (1 - overlap));
  }

  /**
   * Add a new sensor sample to the buffer
   * @returns true if ready for inference, false otherwise
   */
  addSample(sample: SensorSample): boolean {
    const array = sensorSampleToArray(sample);
    this.buffer.push(array);
    this.samplesSinceLastWindow++;

    // Check if ready for inference
    return (
      this.buffer.length >= this.windowSize &&
      this.samplesSinceLastWindow >= this.stepSize
    );
  }

  /**
   * Get current window for inference
   */
  getWindow(): number[][] {
    if (this.buffer.length < this.windowSize) {
      throw new Error('Buffer not ready for window extraction');
    }

    // Extract window (last windowSize samples)
    const window = this.buffer.slice(-this.windowSize);

    // Reset counter
    this.samplesSinceLastWindow = 0;

    // Trim buffer to prevent unlimited growth
    const maxBufferSize = this.windowSize + this.stepSize;
    if (this.buffer.length > maxBufferSize) {
      this.buffer = this.buffer.slice(-maxBufferSize);
    }

    return window;
  }

  /**
   * Get normalized window ready for model inference
   */
  getNormalizedWindow(): number[][][] {
    const window = this.getWindow();
    const normalized = normalizeSensorData(window);
    // Add batch dimension: [1, windowSize, numFeatures]
    return [normalized];
  }

  /**
   * Reset the buffer
   */
  reset(): void {
    this.buffer = [];
    this.samplesSinceLastWindow = 0;
  }

  /**
   * Get current buffer size
   */
  getBufferSize(): number {
    return this.buffer.length;
  }
}
