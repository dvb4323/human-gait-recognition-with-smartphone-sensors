# Phase 2 Quality Assessment - Quick Start Guide

## Overview
This script performs comprehensive data quality assessment on the OU-SimilarGaitActivities dataset based on findings from Phase 1.

## Key Findings from Phase 1
- ⚠️ **52% of data is unlabeled** (ClassLabel = -1)
- ⚠️ **Severe class imbalance**: Stairs activities only 2.8% and 2.6%
- ✅ 504 subjects with 36.3s average recording duration
- ✅ Proper gallery/probe split (252 subjects each)

## Usage

### Run Complete Phase 2 Analysis
```bash
python src/data_analysis/phase2_quality.py
```

By default, this analyzes the **first 100 files**. To analyze all 504 subjects, edit the script:
```python
quality_metrics = qa.run_all_phase2(max_files=None)  # Analyze ALL files
```

### Run Individual Assessments
```python
from src.data_analysis.phase2_quality import OUSimilarGaitQualityAssessment

# Initialize
qa = OUSimilarGaitQualityAssessment("data/raw/OU-SimilarGaitActivities")

# Run individual phases
qa.phase2_1_missing_values(max_files=100)
qa.phase2_2_outlier_detection(max_files=100)
qa.phase2_3_static_period_detection(max_files=50)
qa.phase2_4_sensor_drift_analysis(max_files=50)
qa.phase2_5_timestamp_continuity(max_files=50)
```

## Output
- Console output with detailed quality metrics
- JSON file: `reports/phase2_quality_report.json`

## What Gets Assessed

### 2.1 Missing Value Detection
- Scans all files for NaN/missing values
- Identifies corrupted or incomplete files
- Reports percentage of missing data per sensor

### 2.2 Outlier Detection
- Uses IQR (Interquartile Range) method
- Detects extreme outliers (3*IQR threshold)
- Reports outlier counts and percentages per sensor

### 2.3 Static Period Detection
- Identifies periods with no movement
- Uses rolling window (1 second) to detect low variance
- Flags files with >10% static data

### 2.4 Sensor Drift Analysis
- Compares first vs last 100 samples
- Detects gradual sensor drift over time
- Flags files with drift > 0.1

### 2.5 Timestamp Continuity
- Verifies consistent 100 Hz sampling
- Detects abnormally short (<10s) or long (>60s) recordings
- Reports sample count statistics

## Expected Results
Based on Phase 1 findings, we expect:
- ✅ Minimal missing values (data looks clean)
- ⚠️ Some outliers in gyroscope Y-axis (highest variability)
- ⚠️ Static periods in unlabeled data (-1 class)
- ✅ Minimal sensor drift (sensors appear stable)
- ✅ Consistent sampling rate (100 Hz verified)

## Next Steps
After Phase 2:
- **Phase 3**: Signal processing and feature extraction
- **Phase 4**: Comprehensive visualizations
- **Phase 5**: Preprocessing pipeline implementation
