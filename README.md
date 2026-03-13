# NYC Taxi Trip Duration — End-to-End TFX ML Pipeline

A production-grade machine learning pipeline that predicts NYC taxi trip duration (in seconds) using TensorFlow Extended (TFX). The pipeline covers the full ML lifecycle from raw data ingestion through to model deployment, with artifact lineage tracking via ML Metadata.

---

## Pipeline Overview

```
Raw Data
   └─→ Feature Selection (SelectKBest)
         └─→ ExampleGen         → ingest + split data
               └─→ StatisticsGen    → compute dataset statistics
                     └─→ SchemaGen      → infer feature schema
                           └─→ Schema Curation  → set domains + environments
                                 └─→ ImportSchemaGen  → register schema in MLMD
                                       └─→ StatisticsGen v2 → recompute with schema
                                             └─→ ExampleValidator → detect anomalies
                                                   └─→ Transform   → feature engineering
                                                         └─→ Trainer    → train Keras DNN
                                                               └─→ Evaluator  → validate + bless
                                                                     └─→ Pusher  → deploy if blessed
```

---

## Dataset

A synthetic 50,000-row dataset modelled after the [NYC TLC Taxi Trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Generated automatically on first run if no CSV is present.

| Feature | Type | Description |
|---------|------|-------------|
| `passenger_count` | float | Number of passengers (1–6) |
| `trip_distance` | float | Miles traveled (0–100) |
| `pickup_hour` | float | Hour of day (0–23) |
| `pickup_weekday` | float | Day of week (0=Mon, 6=Sun) |
| `payment_type` | string | Credit Card, Cash, No Charge, Dispute |
| `pickup_location_id` | string | TLC pickup zone ID (1–263) |
| `dropoff_location_id` | string | TLC dropoff zone ID (1–263) |
| `trip_duration` | **int** | **Target** — trip duration in seconds |

---

## Model

A two-hidden-layer Keras DNN trained for regression:

```
Input features (all transformed)
    → Dense(128, ReLU) + BatchNorm + Dropout(0.2)
    → Dense(64,  ReLU) + BatchNorm + Dropout(0.2)
    → Dense(1)   ← linear output
```

- **Loss**: Mean Squared Error
- **Metric**: Mean Absolute Error
- **Optimizer**: Adam (lr=1e-3)
- The `transform_graph` is embedded in the SavedModel's `serving_default` signature — raw untransformed features can be passed directly at inference time, eliminating train/serve skew.

---

## Feature Engineering

| Feature | Transformation |
|---------|---------------|
| `trip_distance`, `passenger_count` | `tft.scale_by_min_max` → [0, 1] |
| `pickup_hour`, `pickup_weekday` | `tft.scale_to_0_1` |
| `payment_type` | `tft.compute_and_apply_vocabulary` |
| `pickup_location_id`, `dropoff_location_id` | `tft.hash_strings` (300 buckets) |
| `trip_duration` (label) | `tft.scale_to_0_1` |

---

## Schema Curation

The auto-inferred schema is curated with domain knowledge before being used in the pipeline:

- Explicit min/max domains set for all numeric features
- Two schema environments defined:
  - `TRAINING` — all features including the label are required
  - `SERVING` — label is excluded (the model predicts it)

---

## Model Validation

The `Evaluator` component computes metrics on:
- The overall eval split
- Per `payment_type` slice

A model is **blessed** (approved for deployment) only if:
- MAE < 0.15 on the scaled [0, 1] label
- MAE is within 10% of the baseline model

`Pusher` skips deployment automatically if the model is not blessed.

---

## Project Structure

```
├── taxi_constants.py     # feature definitions, domains, label key
├── taxi_transform.py     # preprocessing_fn for TFX Transform
├── taxi_trainer.py       # Keras model + TFX run_fn + serving signature
├── pipeline.py           # full pipeline runner (all 14 steps)
├── infer.py              # standalone inference test on pushed model
└── README.md
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install tfx==1.13.0
```

### 2. Run the pipeline

```bash
python pipeline.py
```

This will:
- Generate synthetic data automatically if none exists
- Run all 14 pipeline steps end to end
- Print artifact URIs and ML Metadata lineage at the end

### 3. Run inference

```bash
python infer.py
```

Loads the pushed SavedModel and runs three test predictions with raw input features.
