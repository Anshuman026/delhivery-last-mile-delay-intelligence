# Delhivery Last-Mile Delay Intelligence Pipeline

**Capstone Project:** AI and Machine Learning Certification Program  
**Institution:** E&ICT Academy, IIT Roorkee  
**Project:** Delhivery Last-Mile Delay Intelligence Pipeline  
**Dataset:** Synthetic shipment data generated locally. No real Delhivery shipment records are used.

---

## 1. Project Overview

This project develops an end-to-end machine learning pipeline to analyze last-mile delivery performance, predict delivery delays, identify operationally challenging hubs, and recommend actions for new shipments.

The pipeline covers four major stages:

1. **Data Cleaning and Exploratory Data Analysis:** Clean shipment records, handle missing values and duplicates, engineer features, and identify delay patterns.
2. **Delivery Time Prediction:** Build and compare regression models to estimate delivery time in hours.
3. **Delay Classification:** Train classification models to identify shipments at risk of delay and optimize the prediction threshold.
4. **Hub Segmentation and Operational Recommendations:** Group hubs according to operational performance and generate recommended actions for new shipments.

The final workflow also includes model packaging, model reload validation, and batch prediction for 48 new shipments.

---

## 2. Project Structure

```text
delhivery-capstone/
│
├── data/
│   ├── hubs.csv
│   ├── raw_shipments.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
│
├── models/
│   ├── delay_model.pkl
│   ├── delay_threshold.csv
│   ├── feature_order.csv
│   └── impute_values.csv
│
├── outputs/
│   ├── part_a_checks.csv
│   ├── part_a_correlations.csv
│   ├── part_a_delay_by_group.csv
│   ├── part_b_checks.csv
│   ├── part_b_cv.csv
│   ├── part_b_metrics.csv
│   ├── part_b_tree_depth.csv
│   ├── part_b_gb_learning_rate.csv
│   ├── part_c_models.csv
│   ├── part_c_feature_scores.csv
│   ├── part_c_selectk.csv
│   ├── part_c_tuning.csv
│   ├── part_c_thresholds.csv
│   ├── part_c_final_config.csv
│   ├── part_c_final_test.csv
│   ├── hub_profile.csv
│   ├── hub_segments.csv
│   ├── segment_profile.csv
│   ├── tier_vs_segment_mismatch.csv
│   ├── part_d_separation.csv
│   ├── part_d_reload_check.csv
│   └── new_shipment_actions.csv
│
├── src/
│   └── shipment_prep.py
│
├── generate_data.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
├── predict_delay.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib / Seaborn, where used for analysis
- Git and GitHub

### Python Dependencies

The project uses the following pinned dependencies:

| Library | Version |
|---|---|
| pandas | 2.2.3 |
| numpy | 2.3.5 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |

---

## 4. How to Run the Project

### Prerequisites

- Python 3.10 or later, compatible with the pinned dependencies.
- pip
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the repository URL with the actual GitHub repository URL.

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Generate the dataset

```bash
python generate_data.py
```

Expected generator console output:

```text
24 1224 48
```

This generates:

- `data/hubs.csv`
- `data/raw_shipments.csv`
- `data/new_shipments.csv`

The supplied data generator should not be modified.

### Step 4: Run Part A – Data Cleaning and EDA

```bash
python part_a_clean_eda.py
```

This step performs:

- Duplicate removal.
- Missing-value treatment.
- Sentinel-value handling.
- Service-type standardization.
- Feature engineering.
- Hub information merging.
- Exploratory data analysis.
- Delay-rate analysis.

### Step 5: Run Part B – Regression

```bash
python part_b_regression.py
```

This step trains and evaluates regression models for delivery-time prediction.

### Step 6: Run Part C – Classification

```bash
python part_c_classification.py
```

This step performs:

- Classification model comparison.
- Feature selection.
- Hyperparameter tuning.
- Threshold optimization.
- Final test evaluation.

### Step 7: Run Part D – Hub Segmentation

```bash
python part_d_segments.py
```

This step generates hub profiles, clusters, segment assignments, and tier-versus-segment mismatch analysis.

### Step 8: Package the model

```bash
python package_model.py
```

This generates the serialized model and supporting preprocessing configuration.

### Step 9: Predict new shipment delays

```bash
python predict_delay.py
```

This generates:

```text
outputs/new_shipment_actions.csv
```

**Important:** Run the scripts in the above order from the repository root. `package_model.py` regenerates the packaged classifier and its supporting configuration. Use the pinned dependency versions for reproducibility.

---

## 5. Part A – Data Cleaning and Exploratory Data Analysis

### Dataset Cleaning

The raw dataset initially contained 1,224 shipment records.

| Metric | Result |
|---|---:|
| Raw shipment records | 1,224 |
| Duplicate records removed | 24 |
| Records after deduplication | 1,200 |
| Sentinel weight records | 5 |
| Records after hub merge | 1,200 |
| Nulls in final features and target | 0 |

Missing values were handled using median imputation for numerical columns and appropriate treatment for categorical columns.

Service-type spellings were standardized to three consistent categories:

- Economy
- Express
- Standard

### Overall Delivery Performance

- Total shipments: **1,200**
- Delayed shipments: **278**
- Overall delay rate: **23.17%**

### Key Business Findings

**1. Service Type Analysis**

Express shipments have the highest delay rate.

| Service Type | Shipments | Delayed | Delay Rate |
|---|---:|---:|---:|
| Economy | 315 | 27 | 8.57% |
| Express | 243 | 119 | 48.97% |
| Standard | 642 | 132 | 20.56% |

Express shipments require additional attention because their delay rate is substantially higher than the other service categories.

**2. Hub Tier Analysis**

| Hub Tier | Shipments | Delayed | Delay Rate |
|---|---:|---:|---:|
| Tier 1 | 400 | 16 | 4.00% |
| Tier 2 | 400 | 86 | 21.50% |
| Tier 3 | 400 | 176 | 44.00% |

Delay rates increase significantly from Tier 1 to Tier 3.

**3. Distance Analysis**

| Distance Band | Shipments | Delayed | Delay Rate |
|---|---:|---:|---:|
| 0–200 km | 274 | 6 | 2.19% |
| 201–400 km | 563 | 77 | 13.68% |
| 401–600 km | 275 | 120 | 43.64% |
| 601+ km | 88 | 75 | 85.23% |

Longer delivery distances are associated with significantly higher delay rates.

**4. Correlation Analysis**

The three strongest absolute correlations with delivery time, excluding delivery time itself, are:

| Feature | Correlation |
|---|---:|
| promised_hours | +0.8372 |
| distance_km | +0.4013 |
| n_stops | +0.3247 |

These features show meaningful associations with delivery time. Correlation does not establish causation.

**5. Declared Value Transformation**

The median-filled declared-value distribution is right-skewed.

| Metric | Original Value | Log-Transformed |
|---|---:|---:|
| Mean | 1533.0542 | 7.0277 |
| Median | 1121.5000 | 7.0224 |

The natural-log transformation reduces the influence of high declared values and produces a more balanced distribution.

---

## 6. Part B – Delivery Time Regression

The objective of Part B is to predict delivery time in hours.

### Dataset Split

| Dataset | Records |
|---|---:|
| Training | 840 |
| Validation | 180 |
| Testing | 180 |
| Total | 1,200 |

Feature scaling was fitted on the training data only to avoid data leakage.

### Model Comparison

Five approaches were evaluated:

- Mean Baseline
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Validation results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean Baseline | 15.7952 | 18.8552 | -0.0212 |
| Linear Regression | 3.9222 | 4.8912 | 0.9313 |
| Decision Tree | 5.0971 | 6.2796 | 0.8867 |
| Random Forest | 4.2882 | 5.5095 | 0.9128 |
| Gradient Boosting | 3.9166 | 4.9165 | 0.9306 |

### Selected Regression Model

**Linear Regression** was selected because it achieved the lowest validation RMSE.

Final test performance:

| Metric | Test Result |
|---|---:|
| MAE | 3.5671 |
| RMSE | 4.5235 |
| R² | 0.9493 |

The test R² indicates that the model explains approximately 94.93% of the variation in delivery time on the held-out test dataset.

### Decision Tree Depth Experiment

| Maximum Depth | Training RMSE | Validation RMSE |
|---|---:|---:|
| 1 | 11.6334 | 12.1324 |
| 2 | 9.3687 | 10.4114 |
| 3 | 7.5295 | 8.7033 |
| 4 | 6.0999 | 7.3896 |
| 6 | 4.7114 | 6.2796 |
| 8 | 3.3360 | 6.7901 |
| 12 | 1.3613 | 7.3540 |
| Unlimited | 0.0000 | 7.8368 |

A shallow tree underfits the data, while an unrestricted tree overfits the training data.

Depth 6 provides the lowest validation RMSE in this experiment.

### Gradient Boosting Learning Rate

| Learning Rate | Validation RMSE | Validation R² |
|---|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | 4.9165 | 0.9306 |
| 1.0 | 7.3701 | 0.8440 |

A learning rate of 0.1 gives the best validation performance among the tested values.

### Cross-Validation

| Model | Mean R² | Standard Deviation |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | 0.9353 | 0.0045 |

Cross-validation slightly favors Gradient Boosting. However, Linear Regression is retained because it achieved the lowest validation RMSE and performed strongly on the untouched test set.

---

## 7. Part C – Delay Classification

The objective is to classify shipments as delayed or not delayed.

### Classification Dataset Split

| Dataset | Delayed Shipments |
|---|---:|
| Training | 195 |
| Validation | 42 |
| Testing | 41 |

The classification split is stratified to preserve the target distribution.

### Classification Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Blind All-On-Time Baseline | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision Tree (Depth 6) | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Random Forest | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

The all-on-time baseline achieves relatively high accuracy but fails to identify delayed shipments. Therefore, accuracy alone is not sufficient for this classification problem.

### Feature Selection

The feature-selection experiment evaluated five feature counts.

| Selected Features (k) | Validation F1 |
|---|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

Seven features achieved the highest validation F1 in this experiment.

The top seven features by F-score are:

1. distance_km
2. hub_tier
3. hub_daily_capacity
4. n_stops
5. promised_hours
6. rain_flag
7. sale_event_flag

### Noise Feature Analysis

Five bookkeeping/noise columns were evaluated.

Their combined random-forest importance is **0.2245**, approximately 22.45%.

All five are outside the top seven by F-score:

| Feature | F-Score Rank |
|---|---:|
| label_print_batch | 8 |
| tape_roll_no | 12 |
| sticker_serial_mod | 14 |
| shelf_slot_no | 16 |
| scanner_build_no | 17 |

Each has an F-score below 10.

### Hyperparameter Tuning

Two search strategies were evaluated:

| Search | Candidates | CV Folds | Total Fits | Best CV Accuracy |
|---|---:|---:|---:|---:|
| Grid Search | 27 | 5 | 135 | 0.8952 |
| Randomized Search | 9 | 5 | 45 | 0.8905 |

The untuned random forest using the selected seven features achieved validation F1 of 0.8333, compared with 0.7750 for grid search and 0.7848 for randomized search.

The untuned seven-feature model was therefore retained.

### Threshold Optimization

The selected threshold is **0.5**.

It is the highest tested threshold that maintains validation recall of at least 0.80.

At this threshold:

- Validation precision: 0.7955
- Validation recall: 0.8333
- Validation F1: 0.8140

### Final Classification Model

Selected configuration:

| Parameter | Value |
|---|---|
| Feature count | 7 |
| Model | Random Forest |
| Number of estimators | 100 |
| Maximum depth | Unlimited |
| Minimum samples per leaf | 1 |
| Classification threshold | 0.5 |

### Final Test Performance

| Metric | Test Result |
|---|---:|
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1 Score | 0.7209 |

Confusion matrix:

| | Predicted On-Time | Predicted Delayed |
|---|---:|---:|
| Actual On-Time | 125 | 14 |
| Actual Delayed | 10 | 31 |

On the test set, the classifier identified 31 delayed shipments and missed 10 delayed shipments.

Approximately 69% of its positive delay predictions were correct.

The test set contains only 41 delayed shipments, so the reported precision and recall should not be treated as guaranteed long-term production performance.

---

## 8. Part D – Hub Segmentation

The objective is to identify operationally similar hubs and categorize them into actionable groups.

### Hub Profiling

The analysis covers:

- 24 hubs.
- 50 historical shipments per hub.
- Mean delivery hours.
- Delay rate.
- Mean delivery distance.
- Mean number of stops.

### Cluster Separation Experiment

| Number of Clusters (K) | Across/Within Ratio | Cluster Sizes |
|---|---:|---|
| 2 | 2.8940 | 15-9 |
| 3 | 5.3928 | 9-8-7 |
| 4 | 6.3047 | 8-7-5-4 |
| 5 | 6.8866 | 8-7-4-3-2 |

K=3 was selected because the business requires three practical operating playbooks.

Although K=4 and K=5 achieve higher separation ratios, they produce smaller clusters, including groups of four or fewer hubs.

### Final Hub Segments

| Segment | Number of Hubs | Mean Delivery Hours | Delay Rate |
|---|---:|---:|---:|
| Standard | 7 | 47.2396 | 2.57% |
| Watch | 8 | 56.8137 | 15.50% |
| Escalate | 9 | 64.6504 | 46.00% |

The Escalate segment has the highest average delivery time, distance, number of stops, and delay rate.

### Tier-versus-Segment Mismatches

The following hubs have a mismatch between their assigned hub tier and their observed operational segment:

| Hub ID | City | Hub Tier | Operational Segment |
|---|---|---:|---|
| HUB07 | Kolkata | 1 | Watch |
| HUB13 | Nagpur | 2 | Escalate |
| HUB15 | Kochi | 2 | Escalate |
| HUB22 | Dehradun | 3 | Watch |

This demonstrates that hub tier alone does not fully represent actual delivery performance.

Operational decisions should consider measured performance in addition to the existing hub-tier classification.

---

## 9. Model Packaging and Reload Validation

The classification model is packaged using Joblib.

The following supporting artifacts are generated:

- `models/delay_model.pkl`
- `models/delay_threshold.csv`
- `models/feature_order.csv`
- `models/impute_values.csv`

The model reload check evaluates whether predictions from the reloaded model match the in-memory model.

Validation results:

| Metric | Result |
|---|---:|
| Test rows scored | 180 |
| Reload maximum absolute probability difference | 0.0000000000 |
| Threshold | 0.5000000000 |

The probability difference of zero confirms consistent predictions between the packaged and reloaded model in this validation run.

---

## 10. New Shipment Predictions and Operational Actions

The final pipeline processes 48 new shipments and generates an operational action for each shipment.

Output file:

`outputs/new_shipment_actions.csv`

The output includes:

- Shipment ID
- Hub ID
- Hub segment
- Delay probability
- Flagged status
- Recommended action

### Action Distribution

| Recommended Action | Shipment Count |
|---|---:|
| No action | 40 |
| Escalate to hub manager | 6 |
| Proactive customer call | 2 |
| **Total** | **48** |

Eight shipments were flagged for intervention, representing 16.67% of the new shipment batch.

The action logic prioritizes shipments requiring hub-manager intervention or proactive customer communication.

### Example Operational Recommendations

- High-risk shipments at Escalate hubs should receive hub-manager attention.
- Flagged shipments in Watch hubs should receive proactive customer communication.
- Shipments below the intervention threshold continue through normal operations.

---

## 11. Production Readiness and Monitoring Plan

The following measures would support a production implementation.

### 11.1 Model Accuracy Monitoring

Track:

- Regression MAE.
- Regression RMSE.
- Regression R².
- Classification precision.
- Classification recall.
- Classification F1.

Compare production performance with the current baseline before promoting model updates.

### 11.2 Data Quality Monitoring

Validate:

- Required feature availability.
- Missing values.
- Duplicate shipment IDs.
- Shipment counts between input and output.
- Feature ordering.
- Imputation consistency.

Fail the batch if required features are missing or if shipment IDs are duplicated.

### 11.3 Reproducibility

Maintain:

- Fixed random seeds.
- Pinned dependency versions.
- Documented execution order.
- Versioned output CSVs.
- Packaged model artifacts.
- Model reload validation.

### 11.4 Performance Monitoring

Record prediction runtime and batch size.

For the reference 48-shipment batch, investigate if prediction runtime exceeds 10 seconds in the target environment. Define production SLAs after measuring realistic workloads.

### 11.5 Feature Drift Monitoring

Track changes in:

- Delivery distance.
- Number of stops.
- Rain frequency.
- Service-type distribution.
- Hub distribution.
- Sale-event frequency.

### 11.6 Initial Monitoring Triggers

| Metric | Initial Trigger |
|---|---|
| Weekly delayed share | Above 30% or an increase of more than 5 percentage points against the trailing four-week baseline |
| Flagged shipment share | Above 35% or below 5% |
| Recall on matured shipments | Below 0.70 over at least 50 delayed shipments |
| Feature drift | Weekly mean distance or stop count changes by more than 20% |
| Missing feature rate | Above 2% before imputation |

These are initial investigation thresholds and should be recalibrated using actual production history.

### 11.7 Retraining Strategy

- Retrain monthly initially.
- Trigger an earlier review when monitoring thresholds are breached.
- Validate model performance before promoting a new version.
- Compare sale-event periods with previous sale-event periods rather than ordinary operating weeks.

For the 15–28 September sale window, performance should be compared with equivalent promotional periods because sale-event activity can create a different operational distribution.

---

## 12. Limitations

1. **Synthetic Data:** The dataset is generated locally and reflects the generator's assumptions. It does not represent verified real-world Delhivery shipment behavior.
2. **Small Test Set:** The classification test set contains only 41 delayed shipments. A small number of outcomes can materially affect precision and recall.
3. **Threshold Selection:** The classification threshold was selected using one validation split and a grid of nine candidate thresholds.
4. **Hub Segmentation:** Hub profiles use a limited set of aggregate metrics and 50 historical shipments per hub. Segment assignments should not be treated as permanent labels.
5. **Probability Calibration:** The classifier's predicted probabilities are not guaranteed to be calibrated. Probability calibration and cost-sensitive evaluation should be considered before production deployment.
6. **External Validity:** Model performance on synthetic data does not guarantee equivalent performance on actual logistics operations.

---

## 13. Conclusion

The Delhivery Last-Mile Delay Intelligence Pipeline demonstrates an end-to-end machine learning workflow covering data preparation, exploratory analysis, regression, classification, feature selection, model tuning, hub segmentation, model packaging, and operational recommendations.

Key outcomes include:

- A cleaned modelling dataset containing 1,200 shipment records.
- A Linear Regression model achieving test R² of 0.9493.
- A Random Forest classifier achieving test F1 of 0.7209.
- Seven selected classification features.
- Three operational hub segments: Standard, Watch, and Escalate.
- Four tier-versus-segment mismatches identified.
- A packaged model with a successful reload consistency check.
- Operational recommendations generated for 48 new shipments.

The project provides a reproducible foundation for identifying delivery risks and supporting data-driven last-mile logistics decisions.

---

**Project:** Delhivery Last-Mile Delay Intelligence Pipeline  
**Program:** AI and Machine Learning Certification Program  
**Institution:** E&ICT Academy, IIT Roorkee

**Note:** All results are based on the supplied synthetic dataset and should be validated against real operational data before production deployment.
