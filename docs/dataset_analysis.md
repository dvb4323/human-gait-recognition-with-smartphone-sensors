# OU-SimilarGaitActivities Dataset Analysis

## Overview
The **OU-SimilarGaitActivities** dataset is part of the OU-ISIR Gait Database, specifically the "Similar Action Inertial Dataset". This dataset focuses on gait-based human identification using smartphone and wearable IMU (Inertial Measurement Unit) sensors.

## Dataset Source
- **Institution**: Osaka University, Institute of Scientific and Industrial Research (OU-ISIR)
- **Type**: Inertial sensor data for gait recognition
- **Purpose**: Human identification through gait analysis with similar activities

## Data Collection Setup

### Hardware
- **IMUZ Sensors**: 3 sensors, each containing:
  - Triaxial accelerometer
  - Triaxial gyroscope
- **Sampling Rate**: 100 Hz
- **Sensor Placement**: Around the subject's waist
  - **Center**: Back waist (center position)
  - **Left**: Left waist
  - **Right**: Right waist

### Subjects
- **Total Subjects**: 503 unique individuals
- **Data Coverage**:
  - Center position: 503 files
  - Left position: 496 files
  - Right position: 494 files

## Dataset Structure

### Directory Organization
```
OU-SimilarGaitActivities/
├── Center/                    # 503 files
│   └── T0_Id{ID}_ActLabelAndStepInfor.txt
├── Left/                      # 496 files
│   └── T0_Id{ID}_ActLabelAndStepInfor.txt
├── Right/                     # 494 files
│   └── T0_Id{ID}_ActLabelAndStepInfor.txt
├── PaperProtocol/
│   ├── gallery_list.txt       # 252 subjects
│   ├── probe_list.txt         # 252 subjects
│   └── note.txt
└── Example_Id066238_CenterIMUZ.xlsx
```

### File Format
Each `.txt` file contains time-series sensor data with the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| **Gx** | Gyroscope X-axis | rad/s or deg/s |
| **Gy** | Gyroscope Y-axis | rad/s or deg/s |
| **Gz** | Gyroscope Z-axis | rad/s or deg/s |
| **Ax** | Accelerometer X-axis | g (gravity) |
| **Ay** | Accelerometer Y-axis | g (gravity) |
| **Az** | Accelerometer Z-axis | g (gravity) |
| **ClassLabel** | Activity class | Integer (0-4) |
| **StepLabel** | Step/gait cycle label | Integer (-1 or positive) |

**Header**: `LineWidth: 8`
**Data Format**: Tab or space-separated values

### Activity Classes
Based on the Similar Action Inertial Dataset description and observed class labels (0-4), the five gait activities are likely:

- **Class 0**: Walking on flat ground / level walk
- **Class 1**: Walking up stairs
- **Class 2**: Walking down stairs
- **Class 3**: Walking up slope
- **Class 4**: Walking down slope

### Step Labels
- **-1**: Non-step data (transitions, standing, etc.)
- **Positive integers**: Individual step/gait cycle identifiers

## Data Characteristics

### Sample Statistics (Example: T0_Id066238)
- **Total samples**: 3,835 time points
- **Duration**: ~38.35 seconds (at 100 Hz)
- **Activities**: Multiple classes within single recording
- **Step annotations**: Detailed step-by-step labeling

### Data Quality
- **Synchronized**: All three sensor positions record simultaneously
- **Labeled**: Both activity class and step-level annotations
- **Continuous**: Time-series data with no gaps

## Experimental Protocol

### Gallery/Probe Split
The dataset includes a predefined evaluation protocol:
- **Gallery Set**: 252 subjects (for training/enrollment)
- **Probe Set**: 252 subjects (for testing/verification)
- **Purpose**: Standardized evaluation for gait recognition algorithms

This split enables:
- Person identification experiments
- Person verification experiments
- Cross-activity gait recognition (e.g., train on walking, test on stairs)

## Use Cases

1. **Gait-based Person Identification**: Identify individuals from their walking patterns
2. **Activity Recognition**: Classify different types of walking activities
3. **Step Detection**: Detect and segment individual gait cycles
4. **Multi-sensor Fusion**: Combine data from multiple body positions
5. **Cross-activity Analysis**: Study gait consistency across different activities
6. **Wearable Sensor Research**: Develop algorithms for smartphone/wearable applications

## Key Features

✅ **Large-scale**: 503 subjects with multiple recordings per subject
✅ **Multi-position**: 3 sensor placements for comprehensive analysis
✅ **Multi-activity**: 5 different gait activities including stairs and slopes
✅ **High sampling rate**: 100 Hz for detailed motion capture
✅ **Dual annotation**: Both activity-level and step-level labels
✅ **Standardized protocol**: Predefined gallery/probe split for fair comparison
✅ **6-axis IMU**: Full inertial measurement (accelerometer + gyroscope)

## Data Format Example

```
LineWidth:      8
Gx      Gy      Gz      Ax      Ay      Az      ClassLabel      StepLabel
0.316248        0.133915        0.0472754       0.166   -0.76   0.138     0       -1
0.300269        0.0593471       0.0845604       -0.068  -0.842  -0.05     0       -1
0.294942        -0.031201       0.0259704       -0.112  -0.874  -0.078    0       -1
...
```

## Research Applications

This dataset is particularly valuable for:
- **Biometric authentication** using gait patterns
- **Activity recognition** in daily living scenarios
- **Fall detection** and elderly care monitoring
- **Fitness tracking** and gait analysis
- **Smartphone-based** health monitoring systems
- **Multi-modal sensor fusion** research

## Technical Notes

- **Coordinate System**: Sensor-fixed coordinate frame
- **Data Type**: Floating-point values
- **Missing Data**: Some subjects may not have all three sensor positions
- **File Naming**: `T0_Id{6-digit ID}_ActLabelAndStepInfor.txt`
- **Example File**: `Example_Id066238_CenterIMUZ.xlsx` provided for reference
