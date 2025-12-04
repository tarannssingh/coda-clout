"""
Baseline Models for Predicting Posthumous Wikipedia Legacy
- Logistic Regression (baseline)
- XGBoost (nonlinear baseline)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Force XGBoost import (run pip install xgboost if it fails)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"XGBoost not available: {type(e).__name__}")
    print("If XGBoostError, run: brew install libomp (macOS) or install OpenMP")
    XGBOOST_AVAILABLE = False

# Set random seed
np.random.seed(42)

print("="*60)
print("BASELINE MODELS: Predicting Posthumous Wikipedia Legacy")
print("="*60)

# Load data
print("\n1. Loading data...")
print("   Using BALANCED dataset (equal sampling per year)...")
df = pd.read_csv("../setup/modeling_data_balanced.csv")
print(f"   Loaded {len(df)} samples")

# Filter to dead people only (for classification)
df = df[df['date_of_death'].notna()].copy()
print(f"   Dead people: {len(df)}")

# Temporal split — NO LEAKAGE (this is the entire point of the project)
train_df = df[df['split'] == 'train'].copy()   # 2018–2022
val_df   = df[df['split'] == 'val'].copy()     # 2023–2024

# Get sample weights (fixes 2018 over-representation)
sample_weights_train = train_df['sample_weight'].fillna(1.0).values

print(f"\n   Train: {len(train_df)} samples")
print(f"   Val: {len(val_df)} samples")
print(f"   Train legend rate: {train_df['is_legend'].mean():.1%}")
print(f"   Val legend rate: {val_df['is_legend'].mean():.1%}")

# Feature engineering
print("\n2. Feature engineering...")

def prepare_features(df):
    """Prepare features for modeling with enhanced feature engineering"""
    df = df.copy()
    
    # Get ALL available features (log, occupations, demographics, composite)
    feature_cols = []
    
    # ALL log-transformed features (use all that exist, EXCEPT post-death ones)
    log_features = [col for col in df.columns if col.startswith('log_')]
    # EXCLUDE only post-death features (data leakage!) - include log_edits_past_year
    exclude_features = ['log_sustained_ratio_31_365', 'log_post_30_365_avg_daily']
    log_features = [f for f in log_features if f not in exclude_features]
    
    for feat in log_features:
        df[feat] = df[feat].fillna(0)  # log1p(0) = 0
        feature_cols.append(feat)
    
    # Age at death
    if 'age_at_death' in df.columns:
        df['age_at_death'] = df['age_at_death'].fillna(df['age_at_death'].median())
        feature_cols.append('age_at_death')
    
    # Death year (as numeric)
    if 'death_year' in df.columns:
        df['death_year'] = df['death_year'].fillna(df['death_year'].median())
        feature_cols.append('death_year')
    
    # ALL one-hot encoded occupations (use all that exist, EXCEPT <1% importance ones)
    occ_cols = [col for col in df.columns if col.startswith('occ_')]
    # Remove occupations with <1% importance (from feature importance analysis)
    exclude_occs = ['occ_singer', 'occ_writer', 'occ_musician', 'occ_film_director', 
                    'occ_composer', 'occ_lawyer', 'occ_journalist', 'occ_American_football_player',
                    'occ_screenwriter', 'occ_association_football_player', 'occ_university_teacher',
                    'occ_television_actor']
    occ_cols = [col for col in occ_cols if col not in exclude_occs]
    for col in occ_cols:
        df[col] = df[col].fillna(0).astype(int)
        feature_cols.append(col)
    
    # Cause of death categories (ONE-HOT ENCODE - was missing!)
    # Note: We'll handle this separately to ensure train/val alignment
    # Store cause_category for later processing
    if 'cause_category' in df.columns:
        df['_cause_category'] = df['cause_category'].fillna('unknown')
    
    # Fame proxy (if available)
    if 'fame_proxy' in df.columns:
        df['fame_proxy'] = df['fame_proxy'].fillna(df['fame_proxy'].median())
        feature_cols.append('fame_proxy')
    
    # INTERACTION FEATURES (key interactions that might matter)
    if 'age_at_death' in df.columns and 'fame_proxy' in df.columns:
        df['age_x_fame'] = df['age_at_death'] * df['fame_proxy']
        feature_cols.append('age_x_fame')
    
    if 'age_at_death' in df.columns and 'death_year' in df.columns:
        df['age_x_year'] = df['age_at_death'] * (df['death_year'] - 2018)  # Years since 2018
        feature_cols.append('age_x_year')
    
    # Ratio features (views per sitelink, etc.)
    if 'log_avg_views_pre_death_10d' in df.columns and 'log_sitelinks' in df.columns:
        # Use log features to avoid division issues
        df['views_per_sitelink'] = df['log_avg_views_pre_death_10d'] - df['log_sitelinks']
        feature_cols.append('views_per_sitelink')
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df['is_legend'].astype(int)
    
    return X, y, feature_cols

# Prepare features (without cause categories first)
X_train, y_train, feature_cols = prepare_features(train_df)
X_val, y_val, _ = prepare_features(val_df)

# Now add cause categories with proper alignment
if '_cause_category' in train_df.columns:
    # Get all unique causes from train set
    all_causes = train_df['_cause_category'].fillna('unknown').unique()
    # Top 6 most common
    cause_counts = train_df['_cause_category'].fillna('unknown').value_counts()
    top_causes = cause_counts.head(6).index.tolist()
    
    # One-hot encode for train
    cause_dummies_train = pd.get_dummies(train_df['_cause_category'].fillna('unknown'), prefix='cause')
    # One-hot encode for val
    cause_dummies_val = pd.get_dummies(val_df['_cause_category'].fillna('unknown'), prefix='cause')
    
    # Align columns (add missing, remove extra)
    for cause in top_causes:
        col = f'cause_{cause}'
        if col not in cause_dummies_train.columns:
            cause_dummies_train[col] = 0
        if col not in cause_dummies_val.columns:
            cause_dummies_val[col] = 0
    
    # Keep only top causes
    cause_dummies_train = cause_dummies_train[[f'cause_{c}' for c in top_causes if f'cause_{c}' in cause_dummies_train.columns]]
    cause_dummies_val = cause_dummies_val[[f'cause_{c}' for c in top_causes if f'cause_{c}' in cause_dummies_val.columns]]
    
    # Add to feature sets
    X_train = pd.concat([X_train, cause_dummies_train], axis=1)
    X_val = pd.concat([X_val, cause_dummies_val], axis=1)
    feature_cols.extend(cause_dummies_train.columns.tolist())
    
    print(f"   Added {len(cause_dummies_train.columns)} cause category features")

# Save feature columns for deployment
import json
import os
feature_cols_path = os.path.join(os.path.dirname(__file__), 'detector', 'feature_columns.json')
with open(feature_cols_path, 'w') as f:
    json.dump(feature_cols, f)
print(f"   💾 Saved feature columns to {feature_cols_path}")

print(f"   Selected {len(feature_cols)} features")
print(f"   Features: {', '.join(feature_cols[:10])}...")

# Ensure X_train and X_val have same columns in same order
X_train = X_train[feature_cols]
X_val = X_val[feature_cols]

# Save feature columns for deployment (after finalization)
import json
import os
feature_cols_path = os.path.join(os.path.dirname(__file__), 'detector', 'feature_columns.json')
with open(feature_cols_path, 'w') as f:
    json.dump(feature_cols, f)
print(f"   💾 Saved feature columns to {feature_cols_path}")

# Standardize features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\n3. Training models...")

# Model 1: Logistic Regression
print("\n   Model 1: Logistic Regression")
lr_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    class_weight=None,
    max_iter=1000,
    solver='lbfgs',
    random_state=42
)
lr_model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)

# Model 2: Random Forest (unleash the beast)
print("   Model 2: Random Forest")
rf_model = RandomForestClassifier(
    n_estimators=500,  # More trees for stability
    max_depth=None,  # Let trees grow fully
    min_samples_leaf=1,
    min_samples_split=2,
    class_weight=None,
    random_state=42,
    n_jobs=-1,
    bootstrap=True
)
rf_model.fit(X_train, y_train, sample_weight=sample_weights_train)

# Model 3: XGBoost (the real one)
xgb_model = None
if XGBOOST_AVAILABLE:
    print("   Model 3: XGBoost")
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,  # deeper than RF but controlled
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        min_child_weight=1,
        scale_pos_weight=1,  # DON'T use class_weight, just let sample_weight do it
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,  # Use sample weights (fixes 2018 bias)
        eval_set=[(X_val, y_val)],
        verbose=False
    )
else:
    print("   Model 3: XGBoost (SKIPPED - not available)")

# DEGENERATE MODE - Try everything
print("\n   DEGENERATE MODE ACTIVATED — trying everything...")

# Model 4: CatBoost (usually the winner on tabular) - WITH HYPERPARAMETER TUNING
cat_model = None
try:
    from catboost import CatBoostClassifier
    print("   Model 4: CatBoost (tuning hyperparameters...)")
    
    # Quick hyperparameter search (try 3 best combinations)
    best_auc = 0
    best_params = None
    best_model = None
    
    param_combos = [
        {'iterations': 1500, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 1},
        {'iterations': 2000, 'depth': 7, 'learning_rate': 0.03, 'l2_leaf_reg': 3},
        {'iterations': 1500, 'depth': 6, 'learning_rate': 0.07, 'l2_leaf_reg': 2},
    ]
    
    for i, params in enumerate(param_combos):
        temp_model = CatBoostClassifier(
            iterations=params['iterations'],
            depth=params['depth'],
            learning_rate=params['learning_rate'],
            l2_leaf_reg=params['l2_leaf_reg'],
            random_seed=42,
            verbose=False,
            loss_function='Logloss',
            eval_metric='AUC',
            early_stopping_rounds=100,
            use_best_model=True
        )
        temp_model.fit(X_train, y_train, sample_weight=sample_weights_train,
                      eval_set=(X_val, y_val), verbose=False)
        
        # Evaluate
        y_proba = temp_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        
        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = temp_model
    
    if best_model is not None:
        cat_model = best_model
        print(f"      ✅ CatBoost trained (best params: depth={best_params['depth']}, lr={best_params['learning_rate']}, AUC={best_auc:.3f})")
    else:
        # Fallback to default
        cat_model = CatBoostClassifier(
            iterations=1500, depth=6, learning_rate=0.05, l2_leaf_reg=1,
            random_seed=42, verbose=False, loss_function='Logloss',
            eval_metric='AUC', early_stopping_rounds=100, use_best_model=True
        )
        cat_model.fit(X_train, y_train, sample_weight=sample_weights_train,
                      eval_set=(X_val, y_val), verbose=False)
        print("      ✅ CatBoost trained (default params)")
    
    # Save model for deployment
    if cat_model is not None:
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'detector', 'catboost_model.cbm')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        cat_model.save_model(model_path)
        print(f"      💾 Model saved to {model_path}")
except Exception as e:
    print(f"   Model 4: CatBoost (SKIPPED - {type(e).__name__})")

# Model 5: LightGBM
lgb_model = None
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    print("   Model 5: LightGBM")
    lgb_model = LGBMClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train, sample_weight=sample_weights_train,
                  eval_set=[(X_val, y_val)], eval_metric='auc',
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    print("      ✅ LightGBM trained")
except Exception as e:
    print(f"   Model 5: LightGBM (SKIPPED - {type(e).__name__}: {str(e)[:50]})")

# Model 6: HistGradientBoosting (sklearn, no install needed)
print("   Model 6: HistGradientBoosting")
from sklearn.ensemble import HistGradientBoostingClassifier
hgb_model = HistGradientBoostingClassifier(
    max_iter=1000,
    learning_rate=0.05,
    max_depth=None,
    random_state=42
)
hgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)
print("      ✅ HistGradientBoosting trained")

print("\n4. Evaluating models...")

def evaluate_model(model, X, y, model_name, scaled=False, threshold=0.5):
    """Evaluate model and return metrics"""
    if scaled:
        y_pred_proba = model.predict_proba(X)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Use custom threshold instead of default 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'pr_auc': average_precision_score(y, y_pred_proba),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics

def find_best_threshold(model, X, y, scaled=False, proba_array=None, optimize_for='f1'):
    """Find threshold that maximizes F1 score or recall (with min precision)"""
    if proba_array is not None:
        y_pred_proba = proba_array
    elif scaled:
        y_pred_proba = model.predict_proba(X)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X)[:, 1]
    
    thresholds = np.arange(0.05, 0.9, 0.05)  # Start lower for recall optimization
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if optimize_for == 'recall':
            # Optimize for recall, but require minimum 20% precision
            rec = recall_score(y, y_pred, zero_division=0)
            prec = precision_score(y, y_pred, zero_division=0)
            score = rec if prec >= 0.20 else 0  # Must have at least 20% precision
        else:
            score = f1_score(y, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold

def evaluate_model_lgb(lgb_model, X, y, threshold=0.5):
    """Evaluate LightGBM model"""
    y_pred_proba = lgb_model.predict_proba(X)[:, 1]  # Fix: use predict_proba, not predict
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'pr_auc': average_precision_score(y, y_pred_proba),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics

# Find best thresholds on validation set
print("   Finding optimal thresholds...")
lr_threshold = find_best_threshold(lr_model, X_val_scaled, y_val, scaled=True)
rf_threshold = find_best_threshold(rf_model, X_val, y_val, scaled=False)
print(f"   LR best threshold: {lr_threshold:.3f}")
print(f"   RF best threshold: {rf_threshold:.3f}")

# Evaluate all models with optimal thresholds
lr_metrics = evaluate_model(lr_model, X_val_scaled, y_val, "Logistic Regression", scaled=True, threshold=lr_threshold)
rf_metrics = evaluate_model(rf_model, X_val, y_val, "Random Forest", scaled=False, threshold=rf_threshold)

# Evaluate XGBoost if available
xgb_metrics = None
if XGBOOST_AVAILABLE and xgb_model is not None:
    xgb_threshold = find_best_threshold(xgb_model, X_val, y_val, scaled=False)
    print(f"   XGB best threshold: {xgb_threshold:.3f}")
    xgb_metrics = evaluate_model(xgb_model, X_val, y_val, "XGBoost", scaled=False, threshold=xgb_threshold)
    
    # Save XGBoost model for deployment
    import pickle
    model_path = os.path.join(os.path.dirname(__file__), 'detector', 'xgb_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"   💾 XGBoost model saved to {model_path}")

# Evaluate degenerate mode models
cat_metrics = None
if cat_model is not None:
    cat_threshold = find_best_threshold(cat_model, X_val, y_val, scaled=False)
    print(f"   CatBoost best threshold: {cat_threshold:.3f}")
    cat_metrics = evaluate_model(cat_model, X_val, y_val, "CatBoost", scaled=False, threshold=cat_threshold)

lgb_metrics = None
if lgb_model is not None:
    lgb_threshold = find_best_threshold(lgb_model, X_val, y_val, scaled=False)
    print(f"   LightGBM best threshold: {lgb_threshold:.3f}")
    lgb_metrics = evaluate_model_lgb(lgb_model, X_val, y_val, threshold=lgb_threshold)

hgb_metrics = None
hgb_threshold = find_best_threshold(hgb_model, X_val, y_val, scaled=False)
print(f"   HistGB best threshold: {hgb_threshold:.3f}")
hgb_metrics = evaluate_model(hgb_model, X_val, y_val, "HistGradientBoosting", scaled=False, threshold=hgb_threshold)

# ENSEMBLE: Combine top 3 models (CatBoost + XGBoost + LightGBM)
print("\n   Creating ensemble...")
ensemble_probas = []
ensemble_weights = []

if cat_metrics is not None:
    ensemble_probas.append(cat_metrics['y_pred_proba'])
    ensemble_weights.append(0.5)  # CatBoost gets highest weight (best model)

if xgb_metrics is not None:
    ensemble_probas.append(xgb_metrics['y_pred_proba'])
    ensemble_weights.append(0.3)

if lgb_metrics is not None:
    ensemble_probas.append(lgb_metrics['y_pred_proba'])
    ensemble_weights.append(0.2)

if len(ensemble_probas) >= 2:
    # Normalize weights
    total_weight = sum(ensemble_weights)
    ensemble_weights = [w / total_weight for w in ensemble_weights]
    
    # Weighted average
    ensemble_proba = np.zeros(len(y_val))
    for proba, weight in zip(ensemble_probas, ensemble_weights):
        ensemble_proba += proba * weight
    
    # Find best threshold for ensemble
    ensemble_threshold = find_best_threshold(None, None, y_val, scaled=False, proba_array=ensemble_proba)
    print(f"   Ensemble best threshold: {ensemble_threshold:.3f}")
    
    # Evaluate ensemble
    ensemble_pred = (ensemble_proba >= ensemble_threshold).astype(int)
    ensemble_metrics = {
        'precision': precision_score(y_val, ensemble_pred, zero_division=0),
        'recall': recall_score(y_val, ensemble_pred, zero_division=0),
        'f1': f1_score(y_val, ensemble_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, ensemble_proba),
        'pr_auc': average_precision_score(y_val, ensemble_proba),
        'y_pred': ensemble_pred,
        'y_pred_proba': ensemble_proba
    }
    print(f"      ✅ Ensemble ROC-AUC: {ensemble_metrics['roc_auc']:.3f}")
else:
    ensemble_metrics = None

# Print results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Build results dataframe
results_dict = {
    'Logistic Regression': [
        f"{lr_metrics['precision']:.3f}",
        f"{lr_metrics['recall']:.3f}",
        f"{lr_metrics['f1']:.3f}",
        f"{lr_metrics['roc_auc']:.3f}",
        f"{lr_metrics['pr_auc']:.3f}"
    ],
    'Random Forest': [
        f"{rf_metrics['precision']:.3f}",
        f"{rf_metrics['recall']:.3f}",
        f"{rf_metrics['f1']:.3f}",
        f"{rf_metrics['roc_auc']:.3f}",
        f"{rf_metrics['pr_auc']:.3f}"
    ]
}

if xgb_metrics is not None:
    results_dict['XGBoost'] = [
        f"{xgb_metrics['precision']:.3f}",
        f"{xgb_metrics['recall']:.3f}",
        f"{xgb_metrics['f1']:.3f}",
        f"{xgb_metrics['roc_auc']:.3f}",
        f"{xgb_metrics['pr_auc']:.3f}"
    ]

if cat_metrics is not None:
    results_dict['CatBoost'] = [
        f"{cat_metrics['precision']:.3f}",
        f"{cat_metrics['recall']:.3f}",
        f"{cat_metrics['f1']:.3f}",
        f"{cat_metrics['roc_auc']:.3f}",
        f"{cat_metrics['pr_auc']:.3f}"
    ]

if lgb_metrics is not None:
    results_dict['LightGBM'] = [
        f"{lgb_metrics['precision']:.3f}",
        f"{lgb_metrics['recall']:.3f}",
        f"{lgb_metrics['f1']:.3f}",
        f"{lgb_metrics['roc_auc']:.3f}",
        f"{lgb_metrics['pr_auc']:.3f}"
    ]

if hgb_metrics is not None:
    results_dict['HistGradientBoosting'] = [
        f"{hgb_metrics['precision']:.3f}",
        f"{hgb_metrics['recall']:.3f}",
        f"{hgb_metrics['f1']:.3f}",
        f"{hgb_metrics['roc_auc']:.3f}",
        f"{hgb_metrics['pr_auc']:.3f}"
    ]

if ensemble_metrics is not None:
    results_dict['ENSEMBLE'] = [
        f"{ensemble_metrics['precision']:.3f}",
        f"{ensemble_metrics['recall']:.3f}",
        f"{ensemble_metrics['f1']:.3f}",
        f"{ensemble_metrics['roc_auc']:.3f}",
        f"{ensemble_metrics['pr_auc']:.3f}"
    ]

results_df = pd.DataFrame(results_dict, index=['Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'])

print("\nValidation Set Metrics:")
print(results_df.to_string())

# Save results
os.makedirs('figures', exist_ok=True)
results_df.to_csv('figures/results_table.csv', index=True)
print("\n   Saved: figures/results_table.csv")

# Create visualizations
print("\n5. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion matrices (show best model: CatBoost > XGBoost > RF)
best_model_name = None
best_cm = None
if cat_metrics is not None:
    best_model_name = "CatBoost"
    best_cm = confusion_matrix(y_val, cat_metrics['y_pred'])
elif xgb_metrics is not None:
    best_model_name = "XGBoost"
    best_cm = confusion_matrix(y_val, xgb_metrics['y_pred'])
else:
    best_model_name = "Random Forest"
    best_cm = confusion_matrix(y_val, rf_metrics['y_pred'])

cm_lr = confusion_matrix(y_val, lr_metrics['y_pred'])

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Logistic Regression - Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title(f'{best_model_name} - Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# ROC curves (all models)
fpr_lr, tpr_lr, _ = roc_curve(y_val, lr_metrics['y_pred_proba'])
fpr_rf, tpr_rf, _ = roc_curve(y_val, rf_metrics['y_pred_proba'])

axes[1, 0].plot(fpr_lr, tpr_lr, label=f"LR (AUC={lr_metrics['roc_auc']:.3f})", linewidth=2)
axes[1, 0].plot(fpr_rf, tpr_rf, label=f"RF (AUC={rf_metrics['roc_auc']:.3f})", linewidth=2)

if xgb_metrics is not None:
    fpr_xgb, tpr_xgb, _ = roc_curve(y_val, xgb_metrics['y_pred_proba'])
    axes[1, 0].plot(fpr_xgb, tpr_xgb, label=f"XGB (AUC={xgb_metrics['roc_auc']:.3f})", linewidth=2)

if cat_metrics is not None:
    fpr_cat, tpr_cat, _ = roc_curve(y_val, cat_metrics['y_pred_proba'])
    axes[1, 0].plot(fpr_cat, tpr_cat, label=f"Cat (AUC={cat_metrics['roc_auc']:.3f})", linewidth=2)

if lgb_metrics is not None:
    fpr_lgb, tpr_lgb, _ = roc_curve(y_val, lgb_metrics['y_pred_proba'])
    axes[1, 0].plot(fpr_lgb, tpr_lgb, label=f"LGB (AUC={lgb_metrics['roc_auc']:.3f})", linewidth=2)

if hgb_metrics is not None:
    fpr_hgb, tpr_hgb, _ = roc_curve(y_val, hgb_metrics['y_pred_proba'])
    axes[1, 0].plot(fpr_hgb, tpr_hgb, label=f"HGB (AUC={hgb_metrics['roc_auc']:.3f})", linewidth=2)

if ensemble_metrics is not None:
    fpr_ens, tpr_ens, _ = roc_curve(y_val, ensemble_metrics['y_pred_proba'])
    axes[1, 0].plot(fpr_ens, tpr_ens, label=f"ENSEMBLE (AUC={ensemble_metrics['roc_auc']:.3f})", 
                    linewidth=3, linestyle='--', color='red')

axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curves')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# PR curves (all models)
precision_lr, recall_lr, _ = precision_recall_curve(y_val, lr_metrics['y_pred_proba'])
precision_rf, recall_rf, _ = precision_recall_curve(y_val, rf_metrics['y_pred_proba'])

axes[1, 1].plot(recall_lr, precision_lr, label=f"LR (AUC={lr_metrics['pr_auc']:.3f})", linewidth=2)
axes[1, 1].plot(recall_rf, precision_rf, label=f"RF (AUC={rf_metrics['pr_auc']:.3f})", linewidth=2)

if xgb_metrics is not None:
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_val, xgb_metrics['y_pred_proba'])
    axes[1, 1].plot(recall_xgb, precision_xgb, label=f"XGB (AUC={xgb_metrics['pr_auc']:.3f})", linewidth=2)

if cat_metrics is not None:
    precision_cat, recall_cat, _ = precision_recall_curve(y_val, cat_metrics['y_pred_proba'])
    axes[1, 1].plot(recall_cat, precision_cat, label=f"Cat (AUC={cat_metrics['pr_auc']:.3f})", linewidth=2)

if lgb_metrics is not None:
    precision_lgb, recall_lgb, _ = precision_recall_curve(y_val, lgb_metrics['y_pred_proba'])
    axes[1, 1].plot(recall_lgb, precision_lgb, label=f"LGB (AUC={lgb_metrics['pr_auc']:.3f})", linewidth=2)

if hgb_metrics is not None:
    precision_hgb, recall_hgb, _ = precision_recall_curve(y_val, hgb_metrics['y_pred_proba'])
    axes[1, 1].plot(recall_hgb, precision_hgb, label=f"HGB (AUC={hgb_metrics['pr_auc']:.3f})", linewidth=2)

if ensemble_metrics is not None:
    precision_ens, recall_ens, _ = precision_recall_curve(y_val, ensemble_metrics['y_pred_proba'])
    axes[1, 1].plot(recall_ens, precision_ens, label=f"ENSEMBLE (AUC={ensemble_metrics['pr_auc']:.3f})", 
                    linewidth=3, linestyle='--', color='red')

axes[1, 1].axhline(y=y_val.mean(), color='k', linestyle='--', label='Baseline')
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].set_title('Precision-Recall Curves')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/model_evaluation.png', dpi=150, bbox_inches='tight')
print("   Saved: figures/model_evaluation.png")

# Feature importance (use best model: CatBoost > XGBoost > RF)
print("\n6. Feature importance...")
if cat_model is not None:
    print("   Using CatBoost feature importance (best model)")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': cat_model.feature_importances_
    }).sort_values('importance', ascending=False)
    model_name = "CatBoost"
elif xgb_model is not None:
    print("   Using XGBoost feature importance")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    model_name = "XGBoost"
else:
    print("   Using Random Forest feature importance")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    model_name = "Random Forest"

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

feature_importance.to_csv('figures/feature_importance.csv', index=False)
print("\n   Saved: figures/feature_importance.csv")

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title(f'{model_name} - Top 15 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight')
print("   Saved: figures/feature_importance.png")

print("\n" + "="*60)
print("DONE!")
print("="*60)
print("\nGenerated files:")
print("  - figures/results_table.csv")
print("  - figures/model_evaluation.png")
print("  - figures/feature_importance.csv")
print("  - figures/feature_importance.png")

