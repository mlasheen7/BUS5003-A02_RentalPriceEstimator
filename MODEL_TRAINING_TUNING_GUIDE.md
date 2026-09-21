# PREDICTION MODEL TRAINING & TUNING GUIDE
## For PERSON B (ML Engineer)

---

## 📋 OVERVIEW

**Goal:** Train an XGBoost model to predict rental prices with:
- RMSE target: ±$50/week
- Fairness audit: <±15% error variance across SEIFA groups
- Feature importance analysis with SHAP
- Model validation and cross-validation

**Timeline:**
- Week 4: Build pipeline with mock data (don't wait for PERSON A)
- Thursday Week 4: Swap mock data → real data from PERSON A
- Week 5: Hyperparameter tuning (Optuna, 50 trials)
- Week 6: Fairness audit, feature importance, model comparison
- Week 7+: Final model deployment

---

## 🚀 WEEK 4: SETUP & BASELINE MODEL

### **Step 1: Install Dependencies**

```bash
# Activate venv
venv\Scripts\Activate.ps1

# Install packages
pip install xgboost pycaret optuna scikit-learn pandas numpy shap matplotlib seaborn

# Verify XGBoost
python -c "import xgboost; print(xgboost.__version__)"
```

### **Step 2: Create Project Structure**

```bash
mkdir -p src/models
mkdir -p models/
mkdir -p notebooks/
mkdir -p reports/
```

### **Step 3: Create Training Pipeline**

File: `src/models/train.py`

```python
"""
XGBoost training pipeline for rental price prediction
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("RENTAL PRICE ESTIMATOR - MODEL TRAINING PIPELINE")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[STEP 1] Loading training data...")

try:
    X = pd.read_pickle('data/processed/X_train.pkl')
    y = pd.read_pickle('data/processed/y_train.pkl')
    print(f"  ✓ X shape: {X.shape}")
    print(f"  ✓ y shape: {y.shape}")
except FileNotFoundError:
    print("  ✗ ERROR: Data files not found. Run pipeline first!")
    exit(1)

# Load feature names for reference
with open('data/processed/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

print(f"  Features ({len(feature_names)}): {feature_names[:5]}...")

# ============================================================================
# STEP 2: EXPLORE DATA
# ============================================================================

print("\n[STEP 2] Exploring data distribution...")

print(f"  Target variable (y) statistics:")
print(f"    - Mean: ${y.mean():.2f}/week")
print(f"    - Std: ${y.std():.2f}")
print(f"    - Min: ${y.min():.2f}")
print(f"    - Max: ${y.max():.2f}")
print(f"    - Median: ${y.median():.2f}")

# Check for nulls
print(f"\n  Data quality:")
print(f"    - Null values in X: {X.isnull().sum().sum()}")
print(f"    - Null values in y: {y.isnull().sum()}")

if X.isnull().sum().sum() > 0:
    print(f"  ⚠️ WARNING: Found nulls in X, filling with median")
    X = X.fillna(X.median())

# ============================================================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================================================

print("\n[STEP 3] Splitting data into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"  ✓ Train set: {X_train.shape[0]} samples")
print(f"  ✓ Test set: {X_test.shape[0]} samples")
print(f"  ✓ Train/test split: 80/20")

# ============================================================================
# STEP 4: BASELINE MODEL (DEFAULT HYPERPARAMETERS)
# ============================================================================

print("\n[STEP 4] Training baseline XGBoost model...")

# Initialize XGBoost regressor
xgb_baseline = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    verbose=0
)

# Train
xgb_baseline.fit(X_train, y_train)

# Evaluate on train set
y_train_pred_baseline = xgb_baseline.predict(X_train)
train_rmse_baseline = np.sqrt(mean_squared_error(y_train, y_train_pred_baseline))
train_r2_baseline = r2_score(y_train, y_train_pred_baseline)
train_mae_baseline = mean_absolute_error(y_train, y_train_pred_baseline)

# Evaluate on test set
y_test_pred_baseline = xgb_baseline.predict(X_test)
test_rmse_baseline = np.sqrt(mean_squared_error(y_test, y_test_pred_baseline))
test_r2_baseline = r2_score(y_test, y_test_pred_baseline)
test_mae_baseline = mean_absolute_error(y_test, y_test_pred_baseline)

print(f"\n  Baseline Model Performance:")
print(f"    Train RMSE: ${train_rmse_baseline:.2f}/week")
print(f"    Train MAE: ${train_mae_baseline:.2f}/week")
print(f"    Train R²: {train_r2_baseline:.4f}")
print(f"\n    Test RMSE: ${test_rmse_baseline:.2f}/week")
print(f"    Test MAE: ${test_mae_baseline:.2f}/week")
print(f"    Test R²: {test_r2_baseline:.4f}")

# Check vs target
target_rmse = 50
print(f"\n  Target RMSE: ${target_rmse}/week")
if test_rmse_baseline <= target_rmse:
    print(f"  ✅ ACHIEVED! (${test_rmse_baseline:.2f} ≤ ${target_rmse})")
else:
    print(f"  ⚠️ Not yet achieved (${test_rmse_baseline:.2f} > ${target_rmse})")

# ============================================================================
# STEP 5: CROSS-VALIDATION
# ============================================================================

print("\n[STEP 5] Running 5-fold cross-validation...")

cv_scores = cross_val_score(
    xgb_baseline,
    X_train, y_train,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

cv_rmse = np.sqrt(-cv_scores)
print(f"  CV RMSE: ${cv_rmse.mean():.2f} ± ${cv_rmse.std():.2f}")

# ============================================================================
# STEP 6: FEATURE IMPORTANCE (BUILT-IN)
# ============================================================================

print("\n[STEP 6] Computing feature importance...")

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_baseline.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  Top 10 most important features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"    {row['feature']:35s}: {row['importance']:.4f}")

# ============================================================================
# STEP 7: SAVE BASELINE MODEL
# ============================================================================

print("\n[STEP 7] Saving baseline model...")

# Save model
with open('models/xgboost_v1.0.pkl', 'wb') as f:
    pickle.dump(xgb_baseline, f)

# Save model metadata
metadata = {
    'model_type': 'XGBoost Regressor',
    'trained_date': datetime.now().isoformat(),
    'train_rmse': float(train_rmse_baseline),
    'test_rmse': float(test_rmse_baseline),
    'test_r2': float(test_r2_baseline),
    'test_mae': float(test_mae_baseline),
    'cv_rmse_mean': float(cv_rmse.mean()),
    'cv_rmse_std': float(cv_rmse.std()),
    'n_features': len(feature_names),
    'features': feature_names,
    'hyperparameters': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
}

with open('models/xgboost_v1.0.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✓ Saved: models/xgboost_v1.0.pkl")
print(f"  ✓ Saved: models/xgboost_v1.0.json")

# ============================================================================
# STEP 8: SAVE PERFORMANCE REPORT
# ============================================================================

print("\n[STEP 8] Generating performance report...")

report = f"""
RENTAL PRICE ESTIMATOR - MODEL PERFORMANCE REPORT
{'=' * 80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA:
  Training samples: {len(X_train)}
  Test samples: {len(X_test)}
  Total features: {len(feature_names)}

BASELINE MODEL PERFORMANCE:
  
  Training Set:
    RMSE: ${train_rmse_baseline:.2f}/week
    MAE: ${train_mae_baseline:.2f}/week
    R²: {train_r2_baseline:.4f}
  
  Test Set:
    RMSE: ${test_rmse_baseline:.2f}/week
    MAE: ${test_mae_baseline:.2f}/week
    R²: {test_r2_baseline:.4f}
  
  Cross-Validation (5-fold):
    RMSE: ${cv_rmse.mean():.2f} ± ${cv_rmse.std():.2f}/week

TARGET PERFORMANCE:
  RMSE target: ${target_rmse}/week
  Status: {'✅ ACHIEVED' if test_rmse_baseline <= target_rmse else '⚠️ NOT YET ACHIEVED'}

TOP 10 FEATURES:
"""

for idx, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    report += f"  {idx:2d}. {row['feature']:35s} ({row['importance']:.4f})\n"

report += f"""

NEXT STEPS:
  Week 5: Hyperparameter tuning with Optuna (50 trials)
  Week 6: Fairness audit (SEIFA groups), SHAP analysis
  Week 7: Model deployment and integration
"""

with open('reports/model_performance.md', 'w') as f:
    f.write(report)

print(f"  ✓ Saved: reports/model_performance.md")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ BASELINE MODEL TRAINING COMPLETE!")
print("=" * 80)

print(f"\n📊 SUMMARY:")
print(f"  Model: XGBoost (baseline)")
print(f"  Test RMSE: ${test_rmse_baseline:.2f}/week")
print(f"  Test R²: {test_r2_baseline:.4f}")
print(f"  Target: ${target_rmse}/week")
print(f"  Status: {'✅ ACHIEVED' if test_rmse_baseline <= target_rmse else '⚠️ NOT YET ACHIEVED'}")

print(f"\n🚀 NEXT STEPS:")
print(f"  1. Run hyperparameter tuning: python src/models/tune_hyperparameters.py")
print(f"  2. Analyze feature importance with SHAP")
print(f"  3. Run fairness audit")
print(f"  4. Compare model variants")

print("\n" + "=" * 80)
```

---

## 🎯 WEEK 5: HYPERPARAMETER TUNING

### **Step 4: Hyperparameter Tuning with Optuna**

File: `src/models/tune_hyperparameters.py`

```python
"""
Hyperparameter tuning with Optuna for XGBoost
"""

import pandas as pd
import numpy as np
import pickle
import optuna
from optuna.trial import TrialState
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import json
from datetime import datetime

print("=" * 80)
print("HYPERPARAMETER TUNING WITH OPTUNA")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1] Loading data...")

X = pd.read_pickle('data/processed/X_train.pkl')
y = pd.read_pickle('data/processed/y_train.pkl')

with open('data/processed/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  ✓ Loaded {X_train.shape[0]} training samples")

# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def objective(trial):
    """
    Optuna objective function - minimize test RMSE
    """
    
    # Hyperparameters to tune
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 0.0, 10.0),  # L2 regularization
        'alpha': trial.suggest_float('alpha', 0.0, 10.0),   # L1 regularization
        'objective': 'reg:squarederror',
        'random_state': 42,
        'verbosity': 0
    }
    
    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return rmse

# ============================================================================
# OPTIMIZE
# ============================================================================

print("\n[2] Running Optuna optimization (50 trials)...")
print("  This may take 5-10 minutes...\n")

# Create study
study = optuna.create_study(direction='minimize')

# Optimize
study.optimize(objective, n_trials=50, show_progress_bar=True)

# ============================================================================
# RESULTS
# ============================================================================

print("\n[3] Optimization complete!")

best_trial = study.best_trial

print(f"\n  Best RMSE: ${best_trial.value:.2f}/week")
print(f"\n  Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# ============================================================================
# TRAIN FINAL MODEL WITH BEST PARAMS
# ============================================================================

print("\n[4] Training final model with best hyperparameters...")

best_params = best_trial.params
best_params['objective'] = 'reg:squarederror'
best_params['random_state'] = 42
best_params['verbosity'] = 0

model_final = xgb.XGBRegressor(**best_params)
model_final.fit(X_train, y_train)

# Evaluate
y_train_pred = model_final.predict(X_train)
y_test_pred = model_final.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"  Train RMSE: ${train_rmse:.2f}/week")
print(f"  Test RMSE: ${test_rmse:.2f}/week")

# ============================================================================
# SAVE
# ============================================================================

print("\n[5] Saving tuned model...")

with open('models/xgboost_tuned.pkl', 'wb') as f:
    pickle.dump(model_final, f)

metadata_tuned = {
    'model_type': 'XGBoost (Tuned)',
    'trained_date': datetime.now().isoformat(),
    'train_rmse': float(train_rmse),
    'test_rmse': float(test_rmse),
    'n_trials': 50,
    'best_trial': int(best_trial.number),
    'features': feature_names,
    'hyperparameters': best_params
}

with open('models/xgboost_tuned.json', 'w') as f:
    json.dump(metadata_tuned, f, indent=2)

print(f"  ✓ Saved: models/xgboost_tuned.pkl")
print(f"  ✓ Saved: models/xgboost_tuned.json")

print(f"\n✅ TUNING COMPLETE! Best RMSE: ${test_rmse:.2f}/week")
```

---

## 📊 WEEK 6: FAIRNESS AUDIT & FEATURE ANALYSIS

### **Step 5: Fairness Audit**

File: `src/models/fairness_audit.py`

```python
"""
Fairness audit - check model accuracy across SEIFA groups
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error

print("=" * 80)
print("FAIRNESS AUDIT - CHECKING MODEL BIAS")
print("=" * 80)

# Load model and data
with open('models/xgboost_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

X = pd.read_pickle('data/processed/X_train.pkl')
y = pd.read_pickle('data/processed/y_train.pkl')

merged = pd.read_csv('data/processed/merged_rental_data.csv')

# Get affluence groups
merged['affluence'] = pd.qcut(merged['IRSAD_Score'], q=5, 
                              labels=['very_low', 'low', 'medium', 'high', 'very_high'])

# Predictions
y_pred = model.predict(X)

# Calculate RMSE by affluence group
print("\nModel accuracy by affluence group:\n")

rmse_by_group = {}
for group in ['very_low', 'low', 'medium', 'high', 'very_high']:
    mask = merged['affluence_bin'] == group
    group_rmse = np.sqrt(mean_squared_error(y[mask], y_pred[mask]))
    rmse_by_group[group] = group_rmse
    
    print(f"  {group:10s}: ${group_rmse:6.2f}/week (n={mask.sum()})")

# Check fairness (variance < 15%)
mean_rmse = np.mean(list(rmse_by_group.values()))
rmse_variance = np.std(list(rmse_by_group.values())) / mean_rmse * 100

print(f"\nVariance across groups: {rmse_variance:.1f}%")
print(f"Target: <15%")

if rmse_variance < 15:
    print(f"✅ FAIRNESS CHECK PASSED")
else:
    print(f"⚠️ FAIRNESS CHECK FAILED - Model may be biased")

print("\n" + "=" * 80)
```

### **Step 6: SHAP Feature Importance**

File: `src/models/shap_analysis.py`

```python
"""
SHAP analysis for feature importance and interpretability
"""

import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

print("Computing SHAP values (this may take a few minutes)...")

# Load model and data
with open('models/xgboost_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

X = pd.read_pickle('data/processed/X_train.pkl')

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Plot feature importance
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('reports/shap_feature_importance.png', dpi=150)
print("✓ Saved: reports/shap_feature_importance.png")

# Save SHAP values for later use
import pickle
with open('models/shap_values.pkl', 'wb') as f:
    pickle.dump((shap_values, explainer.expected_value), f)

print("✅ SHAP analysis complete!")
```

---

## 🚀 TRAINING EXECUTION

### **Week 4: Run baseline**

```bash
python src/models/train.py
```

Expected output:
```
[STEP 1] Loading training data...
  ✓ X shape: (1800, 18)
  ✓ y shape: (1800,)

[STEP 4] Training baseline XGBoost model...
  Baseline Model Performance:
    Test RMSE: $65.32/week
    Test R²: 0.7841

[STEP 5] Cross-Validation (5-fold):
    RMSE: $62.14 ± $8.32/week

✅ BASELINE MODEL TRAINING COMPLETE!
```

### **Week 5: Run hyperparameter tuning**

```bash
python src/models/tune_hyperparameters.py
```

### **Week 6: Run fairness audit**

```bash
python src/models/fairness_audit.py
python src/models/shap_analysis.py
```

---

## 📋 MODEL COMPARISON

Compare baseline vs tuned vs other approaches:

```python
import json

# Load metadata
with open('models/xgboost_v1.0.json') as f:
    baseline = json.load(f)

with open('models/xgboost_tuned.json') as f:
    tuned = json.load(f)

print(f"Baseline RMSE: ${baseline['test_rmse']:.2f}")
print(f"Tuned RMSE: ${tuned['test_rmse']:.2f}")
print(f"Improvement: ${baseline['test_rmse'] - tuned['test_rmse']:.2f}")
```

---

## ✅ VALIDATION CHECKLIST

- [ ] Baseline model trains successfully
- [ ] Test RMSE < $70/week (initial target)
- [ ] Cross-validation RMSE within 10% of test RMSE
- [ ] No overfitting (train RMSE ≈ test RMSE)
- [ ] Hyperparameter tuning improves RMSE
- [ ] Final RMSE < $50/week (target)
- [ ] Fairness audit: variance < 15%
- [ ] Feature importance makes sense
- [ ] Model serializes correctly (pickle)
- [ ] Can load and make predictions

---

## 📊 DEPENDENCIES

```
xgboost>=1.7
optuna>=3.0
scikit-learn>=1.0
pandas>=1.3
numpy>=1.20
shap>=0.41
matplotlib>=3.3
```

---

**Ready to train? Run `python src/models/train.py`!** 🚀
