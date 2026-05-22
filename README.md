# 🏠 Home Price Prediction v3 — Dynamic Non-Linear Model

A machine learning project for predicting Thai residential property prices using an ensemble of **XGBoost** and **LightGBM** with automatic hyperparameter tuning via **Optuna**.

---

## Overview

This project builds a home price prediction model trained on Thai housing data. It upgrades from a simple linear regression baseline (v2) to a non-linear gradient boosting ensemble with Bayesian hyperparameter optimization.

| Feature | v2 (Baseline) | v3 (This Project) |
|---|---|---|
| Algorithm | Linear Regression | **XGBoost + LightGBM** |
| Hyperparameters | Fixed (manual) | **Dynamic — Optuna TPE Search** |
| Non-linearity | ❌ | ✅ Captures thresholds & interactions |
| Target transform | MinMaxScaler | **log1p** (handles skewed price distribution) |
| Ensemble | ❌ | ✅ Weighted average of XGB + LGB |
| Dataset | `house_train.csv` | **`house_train_augmented.csv`** (+3 engineered features) |

---

## How It Works

### 1. Data & Features

The model uses augmented housing data with 15 features:

**Original Features**
- `zone` — location zone (urban / suburban inner / suburban outer / metro fringe)
- `house_type` — detached house / townhouse / semi-detached
- `usable_area` — usable floor area (sqm)
- `n_rooms`, `n_bathrooms`, `n_floors`, `parking_slots`, `has_pool`
- `land_size` — land area in square wah (ตร.วา)
- `land_price_per_wa` — land price per square wah
- `bts_dist_km` — distance to nearest BTS station (km)
- `age_years` — age of the property

**Engineered Features (New in v3)**
- `land_size_sqm` — land area converted to square metres (`land_size × 4`)
- `total_area_sqm` — total combined area (`usable_area + land_size_sqm`)
- `is_new_house` — binary flag: 1 if property age ≤ 3 years

### 2. Preprocessing

- Categorical features (`zone`, `house_type`) are encoded with `OrdinalEncoder` — appropriate for tree-based models
- No feature scaling needed — gradient boosting trees are scale-invariant
- Target (`price`) is **log1p-transformed** to normalize the right-skewed price distribution

### 3. Hyperparameter Tuning with Optuna

Instead of grid search, the project uses **Optuna's Tree-structured Parzen Estimator (TPE)** — a Bayesian optimization method that learns from previous trials to focus on promising parameter regions.

Both XGBoost and LightGBM are independently tuned over 60 trials, searching across parameters such as learning rate, tree depth, regularization, subsampling, and more.

### 4. Ensemble

Final predictions are a weighted average:

```
Ensemble = 0.6 × XGBoost + 0.4 × LightGBM
```

Predictions are inverse-transformed (`expm1`) back to Thai Baht.

---

## Project Structure

```
├── Home_price_prediction.ipynb   # Main notebook
├── house_train_augmented.csv     # Training data
├── house_val_augmented.csv       # Validation data
├── house_test_augmented.csv      # Test data
└── models/
    ├── xgb_model.joblib          # Saved XGBoost model
    ├── lgb_model.joblib          # Saved LightGBM model
    ├── ordinal_encoder.joblib    # Saved encoder
    └── feature_config.json       # Feature & category metadata
```

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm optuna joblib
```

### Run the Notebook

1. Clone the repository
2. Place the CSV data files in the root directory
3. Open and run `Home_price_prediction.ipynb` from top to bottom

### Predict a New House

You can predict the price of any property by filling in a dictionary like this at the end of the notebook:

```python
new_house = pd.DataFrame([{
    'zone':               'ชานเมืองชั้นใน',
    'house_type':         'บ้านเดี่ยว',
    'usable_area':        220,
    'n_rooms':            4,
    'n_bathrooms':        3,
    'land_size':          65,
    'land_price_per_wa':  55000,
    'bts_dist_km':        1.8,
    'age_years':          5,
    'n_floors':           2,
    'parking_slots':      2,
    'has_pool':           0,
    'land_size_sqm':      65 * 4,
    'total_area_sqm':     220 + (65 * 4),
    'is_new_house':       0,
}])
```

The notebook will output predictions from XGBoost, LightGBM, and the ensemble in Thai Baht.

---

## Evaluation Metrics

The model is evaluated on three metrics computed on actual (non-log) prices:

- **R²** — proportion of variance explained
- **RMSE** — root mean squared error in Baht
- **MAPE** — mean absolute percentage error

---

## Zone & House Type Values

| Field | Valid Values |
|---|---|
| `zone` | `ใจกลางเมือง`, `ชานเมืองชั้นใน`, `ชานเมืองชั้นนอก`, `ปริมณฑล` |
| `house_type` | `บ้านเดี่ยว`, `ทาวน์เฮาส์`, `บ้านแฝด` |

---

## Tech Stack

- **Python 3.x**
- [XGBoost](https://xgboost.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [Optuna](https://optuna.org/) — hyperparameter optimization
- [scikit-learn](https://scikit-learn.org/) — preprocessing & metrics
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
- [Matplotlib](https://matplotlib.org/) — visualization