# R Project 3 - Human Activity Recognition
# Classifying Physical Activity from Smartphone Sensor Data

This project classifies physical activities and postural transitions from raw smartphone accelerometer and gyroscope signals. The analysis converts 50 Hz triaxial sensor streams into fixed-length epochs, engineers time- and frequency-domain features, compares several multiclass classifiers using cross-validation, and generates predictions for held-out users.

## Research Question

How accurately can physical activities and postural transitions be classified from smartphone accelerometer and gyroscope signals using engineered time-series features and conventional machine-learning classifiers?

## Data

The project uses the **Smartphone-Based Recognition of Human Activities and Postural Transitions** dataset adapted for the University of Amsterdam Behavioural Data Analysis course. The packaged dataset contains raw accelerometer and gyroscope files, training labels, activity definitions, and the original dataset documentation.

The activity classes include standing, sitting, lying, walking, walking downstairs, walking upstairs, and six postural transitions.

## Analysis

The workflow consists of:

1. Loading and aligning raw accelerometer and gyroscope signals with training labels.
2. Splitting each signal into 128-sample epochs, approximately 2.56 seconds at 50 Hz.
3. Engineering descriptive, correlation, shape, energy, RMS, and spectral features.
4. Removing near-zero-variance and highly correlated predictors.
5. Comparing multinomial logistic regression, LDA, KNN, and scaled KNN using 10-fold cross-validation.
6. Applying the best model to held-out test recordings and generating competition-format predictions.

## Original Reported Results

| Model | Best CV accuracy |
| --- | ---: |
| Multinomial logistic regression | **89.21%** |
| Scaled KNN | 87.57% |
| LDA | 86.68% |
| KNN | 74.45% |

The full tuning-level results from the original notebook are stored in `results/original_cv_details.csv`.

## Code Cleaning

The portfolio script follows the original analytical design while correcting evident implementation issues in the coursework notebook, including repeated RMS variable names, duplicated cross-lag definitions, and inconsistent test-set predictor selection. Because of these corrections, rerunning `analysis.R` may produce results that differ somewhat from the historical notebook results.

## Repository Structure

```text
R Project 3 - Human Activity Recognition/
├── README.md
├── analysis.R
├── .gitignore
├── data/
│   ├── README.txt
│   ├── activity_labels.txt
│   ├── example_submission.csv
│   └── RawData/
│       ├── Train/
│       └── Test/
└── results/
    ├── README.md
    ├── original_model_comparison.csv
    ├── original_cv_details.csv
    └── validation_report.csv
```

## Reproducibility

Install the required R packages:

```r
install.packages(c("tidyverse", "caret", "nnet", "e1071"))
```

From the repository root, run:

```r
source("analysis.R")
```

The script writes the rerun model-comparison table and final test predictions to `results/`.

## Skills Demonstrated

- Time-series sensor data
- Human activity recognition
- Feature engineering
- Signal processing and spectral features
- Multiclass classification
- Cross-validation
- Feature selection
- Multinomial logistic regression
- Linear discriminant analysis
- K-nearest neighbours
- Reproducible R workflows

## Contributors

This project was originally completed collaboratively by **Laura** and **Yuxuan**.

| Component | Contributor(s) |
| --- | --- |
| Layout, design and text | Laura |
| Feature extraction | Yuxuan and Laura |
| Model selection | Yuxuan and Laura |

This repository is a cleaned portfolio presentation of the collaborative coursework and preserves the original attribution.

## Dataset Attribution

The included dataset documentation credits Jorge L. Reyes-Ortiz, Davide Anguita, Luca Oneto, Xavier Parra and collaborators. The original `data/README.txt` is included with the repository and should be consulted for the complete dataset description and citation requirements.
