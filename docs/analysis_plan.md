# OU-SimilarGaitActivities Dataset Analysis Plan

## Objective
Thoroughly analyze the OU-SimilarGaitActivities dataset to understand its characteristics and prepare it for deep learning-based gait recognition and activity classification.

---

## Phase 1: Exploratory Data Analysis (EDA)

### 1.1 Data Loading & Basic Statistics
- [ ] Load sample files from all three sensor positions (Center, Left, Right)
- [ ] Verify data format consistency across all files
- [ ] Calculate basic statistics:
  - Number of samples per subject
  - Recording duration per subject
  - Missing data analysis (subjects without all 3 sensor positions)
  - Data type verification (all numeric values)

### 1.2 Sensor Data Distribution Analysis
- [ ] **Accelerometer Analysis**:
  - Distribution of Ax, Ay, Az values
  - Range and outliers detection
  - Mean, std, min, max for each axis
  - Correlation between axes
- [ ] **Gyroscope Analysis**:
  - Distribution of Gx, Gy, Gz values
  - Range and outliers detection
  - Mean, std, min, max for each axis
  - Correlation between axes
- [ ] **Cross-sensor Analysis**:
  - Compare distributions across Center, Left, Right positions
  - Identify position-specific patterns

### 1.3 Activity Class Analysis
- [ ] Class distribution (samples per class)
- [ ] Class balance analysis
- [ ] Average duration per activity class
- [ ] Transition patterns between activities
- [ ] Visualize activity sequences in recordings

### 1.4 Step Label Analysis
- [ ] Distribution of step labels vs non-step data (-1)
- [ ] Average steps per activity class
- [ ] Step duration statistics
- [ ] Gait cycle characteristics (if applicable)

### 1.5 Subject-Level Analysis
- [ ] Samples per subject distribution
- [ ] Activities performed by each subject
- [ ] Recording duration variability across subjects
- [ ] Identify subjects with incomplete data

---

## Phase 2: Data Quality Assessment

### 2.1 Data Integrity Checks
- [ ] Check for NaN/missing values
- [ ] Identify corrupted or incomplete files
- [ ] Verify timestamp continuity (100 Hz sampling)
- [ ] Detect sensor saturation or clipping

### 2.2 Noise & Artifact Analysis
- [ ] Identify static periods (no movement)
- [ ] Detect sudden spikes or anomalies
- [ ] Analyze signal-to-noise ratio
- [ ] Check for sensor drift

### 2.3 Label Quality
- [ ] Verify class label consistency
- [ ] Check for mislabeled segments
- [ ] Validate step label sequences
- [ ] Identify ambiguous transitions

---

## Phase 3: Signal Processing & Feature Exploration

### 3.1 Time-Domain Features
- [ ] Statistical features (mean, std, variance, skewness, kurtosis)
- [ ] Peak detection and counting
- [ ] Zero-crossing rate
- [ ] Signal magnitude area (SMA)
- [ ] Energy and power metrics

### 3.2 Frequency-Domain Features
- [ ] FFT analysis for each sensor axis
- [ ] Dominant frequency identification
- [ ] Power spectral density
- [ ] Spectral entropy
- [ ] Frequency band energy distribution

### 3.3 Time-Frequency Analysis
- [ ] Spectrogram visualization
- [ ] Wavelet transform analysis
- [ ] Short-time Fourier transform (STFT)

### 3.4 Sensor Fusion Features
- [ ] Magnitude of acceleration: √(Ax² + Ay² + Az²)
- [ ] Magnitude of angular velocity: √(Gx² + Gy² + Gz²)
- [ ] Jerk (derivative of acceleration)
- [ ] Cross-sensor correlations

---

## Phase 4: Visualization & Pattern Discovery

### 4.1 Raw Signal Visualization
- [ ] Plot sample time-series for each activity class
- [ ] Overlay accelerometer and gyroscope data
- [ ] Compare signals across sensor positions
- [ ] Visualize step-segmented data

### 4.2 Statistical Visualizations
- [ ] Distribution plots (histograms, KDE)
- [ ] Box plots for each sensor axis by activity
- [ ] Correlation heatmaps
- [ ] PCA/t-SNE for dimensionality reduction

### 4.3 Activity-Specific Patterns
- [ ] Identify distinguishing features per activity
- [ ] Analyze gait patterns for flat vs stairs vs slopes
- [ ] Compare upward vs downward movements
- [ ] Subject variability within same activity

---

## Phase 5: Deep Learning Preparation

### 5.1 Data Preprocessing Pipeline
- [ ] **Normalization/Standardization**:
  - Z-score normalization per sensor
  - Min-max scaling
  - Per-subject normalization
- [ ] **Filtering**:
  - Low-pass filter for noise reduction
  - High-pass filter for drift removal
  - Butterworth or median filtering
- [ ] **Resampling** (if needed):
  - Handle variable-length sequences
  - Interpolation for missing samples

### 5.2 Segmentation Strategy
- [ ] **Fixed-length windows**:
  - Determine optimal window size (e.g., 2s, 3s, 5s)
  - Overlap percentage (e.g., 50%, 75%)
  - Padding strategy for short sequences
- [ ] **Activity-based segmentation**:
  - Segment by activity transitions
  - Use step labels for gait cycle segmentation
- [ ] **Sliding window approach**:
  - Real-time inference simulation

### 5.3 Label Encoding
- [ ] One-hot encoding for activity classes
- [ ] Handle multi-label scenarios (if applicable)
- [ ] Create person ID labels for identification task
- [ ] Binary labels for verification task

### 5.4 Train/Validation/Test Split
- [ ] **Respect gallery/probe protocol**:
  - Gallery (252 subjects) → Training set
  - Probe (252 subjects) → Test set
- [ ] **Create validation set**:
  - Split gallery into train/val (e.g., 80/20)
  - Ensure subject-independent split
- [ ] **Cross-validation strategy**:
  - K-fold cross-validation on gallery set
  - Leave-one-subject-out (LOSO) validation

### 5.5 Data Augmentation Strategies
- [ ] **Time-domain augmentation**:
  - Random cropping
  - Time warping
  - Jittering (add random noise)
  - Scaling (amplitude variation)
- [ ] **Rotation augmentation**:
  - Simulate sensor orientation changes
- [ ] **Mixup/CutMix** for sequences
- [ ] **Synthetic minority oversampling** (if class imbalance)

### 5.6 Multi-Sensor Fusion Strategy
- [ ] **Early fusion**: Concatenate all sensor data
- [ ] **Late fusion**: Separate models per sensor, combine predictions
- [ ] **Intermediate fusion**: Multi-stream architecture
- [ ] Determine if all 3 positions are necessary or if Center alone suffices

---

## Phase 6: Baseline Establishment

### 6.1 Classical ML Baselines
- [ ] Extract handcrafted features
- [ ] Train traditional classifiers:
  - Random Forest
  - SVM
  - XGBoost
- [ ] Evaluate on gallery/probe split

### 6.2 Simple Deep Learning Baselines
- [ ] 1D CNN for time-series classification
- [ ] LSTM/GRU for sequence modeling
- [ ] Hybrid CNN-LSTM architecture
- [ ] Evaluate and compare with classical methods

---

## Phase 7: Documentation & Reporting

### 7.1 Analysis Report
- [ ] Summary statistics tables
- [ ] Key findings and insights
- [ ] Data quality assessment results
- [ ] Recommended preprocessing pipeline

### 7.2 Visualization Gallery
- [ ] Representative samples per activity
- [ ] Distribution plots
- [ ] Feature importance analysis
- [ ] Confusion matrices from baselines

### 7.3 Code Organization
- [ ] Data loading utilities
- [ ] Preprocessing functions
- [ ] Feature extraction modules
- [ ] Visualization scripts
- [ ] Baseline model implementations

---

## Deliverables

1. **EDA Notebook**: Comprehensive exploratory analysis with visualizations
2. **Data Processing Pipeline**: Reusable preprocessing code
3. **Dataset Statistics Report**: Detailed characterization document
4. **Baseline Results**: Performance benchmarks for comparison
5. **Recommendations**: Optimal strategies for deep learning training

---

## Tools & Libraries

- **Data Processing**: pandas, numpy, scipy
- **Signal Processing**: scipy.signal, pywt (wavelets)
- **Visualization**: matplotlib, seaborn, plotly
- **ML/DL**: scikit-learn, TensorFlow/PyTorch
- **Feature Extraction**: tsfresh, tsfel

---

## Timeline Estimate

- **Phase 1-2**: 2-3 days (EDA + Quality Assessment)
- **Phase 3-4**: 2-3 days (Feature Exploration + Visualization)
- **Phase 5**: 2-3 days (DL Preparation)
- **Phase 6**: 1-2 days (Baselines)
- **Phase 7**: 1 day (Documentation)

**Total**: ~8-12 days for thorough analysis
