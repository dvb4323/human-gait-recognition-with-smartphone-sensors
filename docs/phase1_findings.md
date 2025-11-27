# Phase 1 EDA - Key Findings & Insights

## Executive Summary
Phase 1 analysis of 10 sample subjects from the OU-SimilarGaitActivities dataset revealed important characteristics and challenges that will inform our preprocessing and model training strategy.

---

## 📊 Dataset Statistics

### File & Subject Coverage
- **Total Files**: 1,493 files across 3 sensor positions
  - Center: 503 files (100%)
  - Left: 496 files (98.6%)
  - Right: 494 files (98.2%)
- **Unique Subjects**: 504
- **Complete Data** (all 3 sensors): 490 subjects (97.2%)
- **Missing Data**:
  - 8 subjects missing Left sensor
  - 9 subjects missing Right sensor

### Recording Characteristics
- **Average Duration**: 36.3 seconds (±8.2s)
- **Average Samples**: 3,635 samples per recording
- **Sampling Rate**: 100 Hz (confirmed)
- **Duration Range**: ~20-52 seconds per recording

### Gallery/Probe Protocol
- ✅ **Gallery**: 252 subjects (training/enrollment)
- ✅ **Probe**: 252 subjects (testing/verification)
- ✅ **No overlap** between sets (proper evaluation protocol)

---

## 🔍 Sensor Data Characteristics

### Accelerometer (in g)
| Axis | Mean | Std | Min | Max | Range |
|------|------|-----|-----|-----|-------|
| **Ax** | 0.021 | 0.202 | -1.284 | 2.258 | 3.542 |
| **Ay** | -1.001 | 0.268 | -3.554 | -0.066 | 3.488 |
| **Az** | 0.002 | 0.216 | -0.750 | 1.968 | 2.718 |

**Key Observations**:
- **Ay (vertical)** has strong negative bias (~-1g) → sensor oriented with Y-axis pointing up
- Ay shows gravity component, indicating upright posture
- Ax and Az centered near zero → horizontal plane movements
- Reasonable ranges, no extreme outliers detected

### Gyroscope (in rad/s or deg/s)
| Axis | Mean | Std | Min | Max | Range |
|------|------|-----|-----|-----|-------|
| **Gx** | -0.013 | 0.305 | -1.927 | 3.970 | 5.897 |
| **Gy** | -0.066 | 1.002 | -4.415 | 5.950 | 10.365 |
| **Gz** | -0.008 | 0.313 | -1.978 | 2.102 | 4.080 |

**Key Observations**:
- **Gy** has highest variability (std=1.002) → primary rotation axis during walking
- All axes centered near zero (minimal drift)
- Gy range is largest → yaw/turning movements during gait
- No sensor saturation detected

---

## ⚠️ Critical Issues Identified

### 1. Severe Class Imbalance
```
Class Distribution:
  -1 (Unlabeled):     52.09% (18,933 samples) ⚠️
   0 (Flat walk):     29.53% (10,733 samples)
   1 (Up stairs):      2.83% (1,030 samples)  ⚠️ VERY LOW
   2 (Down stairs):    2.63% (957 samples)    ⚠️ VERY LOW
   3 (Up slope):       6.23% (2,266 samples)
   4 (Down slope):     6.68% (2,428 samples)
```

**Implications**:
- 🚨 **52% of data is unlabeled** (ClassLabel = -1) → likely transitions/standing
- 🚨 **Stairs classes severely underrepresented** (2.8% and 2.6%)
- Flat walking dominates labeled data (29.5%)
- **Action Required**: 
  - Consider removing unlabeled data OR treating as separate "transition" class
  - Use data augmentation heavily for stairs classes
  - Consider class weighting in loss function
  - May need to collect more stairs data or use SMOTE

### 2. Step Label Distribution
- **Non-step data**: 54.5% (19,794 samples)
- **Step data**: 45.5% (16,553 samples)
- **Average steps per recording**: 30.8 steps

**Implications**:
- Step segmentation could be useful for feature extraction
- Gait cycle-based windowing might improve performance
- Step labels can be used for semi-supervised learning

### 3. Activity Transitions
- **Average transitions per recording**: 10.3
- Multiple activity changes within single recording
- Transition periods likely contribute to unlabeled data

---

## 💡 Actionable Insights

### For Preprocessing (Phase 2)
1. **Handle Unlabeled Data**:
   - Option A: Remove all -1 labels (lose 52% of data)
   - Option B: Treat as "transition" class (6 classes total)
   - Option C: Use only step-labeled data (45.5% of data)
   - **Recommendation**: Start with Option A, evaluate Option B later

2. **Address Class Imbalance**:
   - Implement class weighting: `weight = 1 / class_frequency`
   - Use SMOTE or time-series augmentation for minority classes
   - Consider focal loss for training

3. **Sensor Normalization**:
   - Z-score normalization per sensor axis
   - Consider per-subject normalization to handle individual differences
   - Ay needs special handling due to gravity component

4. **Filtering Strategy**:
   - Low-pass filter (cutoff ~20 Hz) to remove high-frequency noise
   - Gy shows most variability → may benefit from smoothing

### For Segmentation (Phase 5)
1. **Window Size**:
   - Average duration: 36.3s → too long for single window
   - Recommended: 2-3 second windows (200-300 samples)
   - With 50% overlap: ~20-30 windows per recording

2. **Segmentation Approach**:
   - **Option A**: Fixed sliding windows (simple, standard)
   - **Option B**: Step-based segmentation (30.8 steps available)
   - **Option C**: Activity-based segmentation (10.3 transitions)
   - **Recommendation**: Start with fixed windows, explore step-based later

### For Model Training
1. **Multi-task Learning Opportunity**:
   - Task 1: Activity classification (5 classes, excluding -1)
   - Task 2: Person identification (504 subjects)
   - Task 3: Step detection (binary: step vs non-step)

2. **Data Split Strategy**:
   - Use gallery/probe protocol for person identification
   - For activity classification: stratified split respecting class imbalance
   - Ensure subject-independent splits (no subject in both train/test)

3. **Sensor Fusion**:
   - Start with Center sensor only (503 files, 100% coverage)
   - Later explore multi-sensor fusion (490 subjects with all 3)

---

## 🎯 Updated Priorities for Phase 2

### High Priority
1. ✅ **Data quality checks** (missing values, outliers, corrupted files)
2. ✅ **Handle unlabeled data** (decide on strategy)
3. ✅ **Noise analysis** (identify static periods, artifacts)
4. ✅ **Class imbalance quantification** (across all subjects, not just 10)

### Medium Priority
5. ⚠️ Sensor drift detection
6. ⚠️ Timestamp continuity verification
7. ⚠️ Cross-sensor synchronization check

### Low Priority (defer to later phases)
8. ⏸️ Label quality assessment (manual inspection)
9. ⏸️ Subject demographic analysis (if available)

---

## 📈 Next Steps

### Immediate Actions (Phase 2)
1. Run data quality assessment on **all 504 subjects** (not just 10)
2. Quantify class imbalance across entire dataset
3. Identify and flag corrupted/anomalous files
4. Detect static periods and sensor artifacts
5. Generate quality report with recommendations

### Subsequent Phases
- **Phase 3**: Signal processing and feature engineering
- **Phase 4**: Comprehensive visualizations
- **Phase 5**: Preprocessing pipeline implementation
- **Phase 6**: Baseline model training

---

## 🔑 Key Takeaways

1. ✅ **Dataset is well-structured** with proper gallery/probe split
2. ✅ **Sensor data looks reasonable** (no major issues detected)
3. ⚠️ **Severe class imbalance** requires special handling
4. ⚠️ **52% unlabeled data** needs decision on usage
5. ⚠️ **Stairs activities underrepresented** (need augmentation)
6. ✅ **Step labels available** for advanced segmentation
7. ✅ **Multi-sensor data** enables fusion experiments
