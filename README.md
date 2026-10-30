# Delhivery Last-Mile Delay Intelligence Pipeline

**Capstone:** AI and Machine Learning Certification Program · E&ICT Academy, IIT Roorkee  
**Dataset:** Synthetic shipments generated locally; no real Delhivery shipment records are used.

This repository implements one connected workflow: clean historical shipments, build reusable features, estimate delivery hours, classify late parcels, segment hubs, package the classifier, and generate an action list for 48 new shipments.

## How to run

Run all commands from the repository root in this order, on a clean clone:

```bash
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`generate_data.py` writes `data/hubs.csv`, `data/raw_shipments.csv`, and `data/new_shipments.csv`; its expected console output is `24 1224 48`. It is committed as the supplied generator and should not be edited. Part A writes `data/modelling_table.csv` and its evidence CSVs. Parts B and C write regression and classification diagnostics. Part D writes the hub profiles and segments. `package_model.py` regenerates `models/delay_model.pkl`, the threshold, feature-order, and imputation files. `predict_delay.py` writes `outputs/new_shipment_actions.csv`.

The committed pickle is tied to the scikit-learn version pinned in `requirements.txt`. Rebuild it by running `python package_model.py` in the pinned environment; if the installed scikit-learn version differs, regenerate rather than relying on the committed pickle. All CSV outputs are written without DataFrame indexes except the correlation matrix, where the row labels are intentionally retained as the first column.

## Data findings

- **Service promise:** Express shipments have the highest delay rate: **0.4897** (119 delayed out of 243). Standard is 0.2056 and Economy is 0.0857 (`outputs/part_a_delay_by_group.csv`).
- **Hub tier:** Delay rate rises from **0.0400** for Tier 1 to **0.2150** for Tier 2 and **0.4400** for Tier 3 (`outputs/part_a_delay_by_group.csv`).
- **Distance:** Delay rate rises across bands: **0.0219** at 0–200 km, **0.1368** at 201–400 km, **0.4364** at 401–600 km, and **0.8523** above 600 km (`outputs/part_a_delay_by_group.csv`).
- **Correlations with delivery time:** the three strongest absolute correlations excluding `delivery_hours` itself are `promised_hours` **+0.8372**, `distance_km` **+0.4013**, and `n_stops` **+0.3247** (`outputs/part_a_correlations.csv`). Correlation is association, not proof of causation.
- **Declared value transformation:** the median-filled `declared_value_inr` has mean **INR 1,533.0542** and median **INR 1,121.5000**, showing a right-skewed distribution. After natural-log transformation, the mean is **7.0277** and median **7.0224**, which are much closer; the log reduces the influence of high declared values (`outputs/part_a_checks.csv`).
- **Cleaning evidence:** the 1,224-row raw file contains 24 exact duplicate rows; 1,200 unique rows remain. Five weight sentinels are treated as missing before median imputation. The final modelling table has zero nulls across the pinned 17 features and the target columns (`outputs/part_a_checks.csv`).

## Model results (regression)

The split is **840 training / 180 validation / 180 test**. The training-scaled maximum absolute feature mean is effectively zero, while the test-scaled maximum absolute mean is **0.2187**, consistent with fitting the scaler on training rows only (`outputs/part_b_checks.csv`).

- The mean baseline has validation **R² = -0.0212**: predicting one constant mean is slightly worse than predicting the validation target mean and does not explain useful variation.
- **Linear regression** wins the validation comparison with RMSE **4.8912 hours** and R² **0.9313**. Gradient boosting is close at RMSE 4.9165 and R² 0.9306. Linear regression is the selected final regression model and has test RMSE **4.5235 hours**, test R² **0.9493** (`outputs/part_b_metrics.csv`).
- Tree depth 1 has high bias: training RMSE 11.6334 and validation RMSE 12.1324 are both high. The unlimited tree has high variance: training RMSE 0.0000 versus validation RMSE 7.8368. Depth 6 has the lowest validation RMSE, **6.2796** (`outputs/part_b_tree_depth.csv`).
- The best tested gradient-boosting learning rate is **0.1** (validation R² 0.9306). At 0.01, learning is too slow for 100 estimators (R² 0.7127); at 1.0, the larger steps overshoot useful improvements (R² 0.8440) (`outputs/part_b_gb_learning_rate.csv`).
- Cross-validation on the combined 1,020 training-plus-validation rows slightly favors gradient boosting (mean R² **0.9353**, std 0.0045) over linear regression (mean R² 0.9261, std 0.0048). For this repository, linear regression is retained because it wins the held-out validation comparison and performs strongly on the untouched test set; the cross-validation difference is small and both are stable (`outputs/part_b_cv.csv`).

## Model results (classification)

The stratified split has **195 / 42 / 41** delayed parcels across training, validation, and test, respectively. The all-on-time baseline reaches validation accuracy **0.7667**, but recall, precision, and F1 are all zero: it never identifies a delayed parcel, so accuracy alone is misleading (`outputs/part_c_models.csv`).

- The five bookkeeping/noise columns have combined random-forest importance **0.2245** (approximately 21.7%). All five are outside the top seven by F-score; their ranks are 8, 9, 11, 13, and 14, and each has F-score below 10 (`outputs/part_c_feature_scores.csv`).
- The feature-count experiment gives validation F1 **0.5714** for k=5, **0.6743** for k=7, **0.6437** for k=10, **0.6509** for k=12, and **0.6163** for k=17. The required k=7 choice is a useful compact feature set (`outputs/part_c_selectk.csv`).
- Tuning did not beat the untuned forest on this 180-row validation set: the untuned forest on the selected seven features has F1 **0.8333**, compared with 0.7750 for grid search and 0.7848 for randomized search. This illustrates why tuning should be judged on held-out performance, not only cross-validation accuracy (`outputs/part_c_tuning.csv`).
- The selected model uses k=7, a 100-tree forest with unlimited depth and minimum leaf size 1. The alert threshold is **0.5**, the highest tested threshold with validation recall at least 0.80. At this threshold validation precision is 0.7955 and recall is 0.8333 (`outputs/part_c_thresholds.csv`, `outputs/part_c_final_config.csv`).
- On the untouched test set, precision is **0.6889**, recall **0.7561**, and F1 **0.7209**. That is roughly 7 of every 10 flagged parcels truly delayed, and about 8 of every 10 delayed parcels caught. The test set contains only 41 delayed parcels, so recall is uncertain and should not be treated as a precise long-run guarantee (`outputs/part_c_final_test.csv`).

## Operations recommendation

Use the three hub segments as playbooks, alongside parcel-level risk probabilities. The **Escalate** segment contains nine hubs and has mean hub delay rate **0.4600**, mean delivery time 64.6504 hours, mean distance 470.4757 km, and mean route stops 84.3444. **Watch** has eight hubs and delay rate 0.1550; **Standard** has seven hubs and delay rate 0.0257 (`outputs/segment_profile.csv`).

Tier labels do not fully reflect current operating behavior. The mismatch file identifies **HUB07 Kolkata** (Tier 1, Watch), **HUB13 Nagpur** (Tier 2, Escalate), **HUB15 Kochi** (Tier 2, Escalate), and **HUB22 Dehradun** (Tier 3, Watch). Treat the tier as a planning label, not a substitute for measured hub performance, and review these four hubs operationally (`outputs/tier_vs_segment_mismatch.csv`).

At threshold 0.5, test precision is 0.6889 and recall is 0.7561. In the generated batch of 48 new parcels, **8 are flagged** and **6 of those require hub-manager escalation**; the remaining two flagged parcels receive proactive customer calls. Non-flagged parcels receive no action (`outputs/new_shipment_actions.csv`).

K=3 is retained because the business needs exactly three playbooks and it yields a clear across-to-within distance ratio of **5.3928**, with cluster sizes **9-8-7**. The ratio rises at K=4 (6.3047) and K=5 (6.8866), but those solutions create clusters of only four or fewer hubs, too small to justify independent operating playbooks (`outputs/part_d_separation.csv`).

## Production plan

- **Accuracy:** keep regression MAE/RMSE/R² and classifier precision/recall/F1 in the versioned CSV outputs; compare them against the current baseline before promotion.
- **Reliability:** fail the batch if any of the 17 model features is null, if shipment IDs are duplicated, or if the scoring output is not exactly one row per input shipment.
- **Reproducibility:** use fixed random seeds, pinned dependency versions, a documented command order, and committed output CSVs. `package_model.py` checks that serialized and in-memory probabilities match.
- **Performance:** record runtime and batch size for `predict_delay.py`; alert if the 48-row reference batch takes more than 10 seconds in the target environment, then set a production-specific SLA after measuring realistic volumes.
- **Maintainability:** centralize cleaning and feature engineering in `src/shipment_prep.py`; the training and scoring scripts call the same functions.
- **Consistency:** persist the exact feature order and imputation medians in `models/`; never independently reimplement preprocessing in production scoring.
- **Monitorability:** store each batch's probability distribution, flagged share, hub segment, and eventual observed outcomes with a timestamp so delayed outcomes can be evaluated later.

Suggested initial monitoring triggers (review and recalibrate after enough production history):

1. **Weekly delayed share:** investigate if the share of delivered parcels that are late rises above **30%** or increases by more than **5 percentage points** versus the trailing four-week baseline.
2. **Flagged share:** investigate if more than **35%** or fewer than **5%** of dispatched parcels are flagged in a week; this can indicate a data or distribution shift.
3. **Recall on matured parcels:** once outcomes are known, alert if weekly recall falls below **0.70** over at least 50 delayed parcels; if fewer are available, report the sample size and avoid over-interpreting the metric.
4. **Feature drift:** investigate if the weekly mean distance or stop count shifts by more than **20%** from the training baseline, or if any required feature's missingness exceeds **2%** before imputation.

Retrain monthly at first, and sooner only if monitored performance or drift triggers are breached and the investigation confirms a sustained change. For the **15–28 September sale window**, compare against prior sale-event periods or an equivalent promotion window, not ordinary weeks, because `sale_event_flag` represents a distinct operating regime. Data drift may appear as changes in distance, stop count, rain frequency, service mix, hub mix, or sale-event frequency; monitor both feature distributions and delayed outcomes.

## Limitations

1. The data is synthetic and reflects the generator's assumptions, not real-world Delhivery shipment behavior; external validity is unknown.
2. The test set has only 41 delayed parcels, so a few outcomes can materially change precision and recall.
3. The threshold was selected from one validation split and a coarse grid of nine thresholds; it may not be optimal under changed operating costs or class prevalence.
4. Hub segments are based on a limited set of four aggregate profile metrics and 50 historical parcels per hub; they are operational summaries, not permanent labels.
5. The classifier's probability values are not guaranteed to be calibrated; calibration and cost-sensitive evaluation should be considered before production deployment.
