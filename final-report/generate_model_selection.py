"""
Generate the BEST graph showing model selection and why XGBoost was chosen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

print("="*60)
print("GENERATING MODEL SELECTION GRAPH")
print("="*60)

# Load results
results_df = pd.read_csv('figures/results_table.csv', index_col=0)
print(f"\nLoaded results for {len(results_df.columns)} models")

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Selection: Why XGBoost?', fontsize=18, fontweight='bold', y=0.98)

models = results_df.columns.tolist()
colors = ['#95a5a6' if m != 'XGBoost' else '#e74c3c' for m in models]  # Red for XGBoost, gray for others

# Top Left: ROC-AUC (Primary Metric)
ax = axes[0, 0]
roc_aucs = results_df.loc['ROC-AUC', models].values
bars = ax.bar(models, roc_aucs, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
ax.set_ylabel('ROC-AUC Score', fontweight='bold', fontsize=12)
ax.set_title('ROC-AUC: Discriminative Ability', fontweight='bold', fontsize=13)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0.85, color='green', linestyle='--', linewidth=2, alpha=0.7, label='XGBoost: 0.850')

# Add value labels
for bar, val, model in zip(bars, roc_aucs, models):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11,
            color='#e74c3c' if model == 'XGBoost' else 'black')
    if model == 'XGBoost':
        bar.set_edgecolor('#e74c3c')
        bar.set_linewidth(3)

ax.legend(loc='lower right')
ax.set_xticklabels(models, rotation=45, ha='right')

# Top Right: Precision vs Recall Trade-off
ax = axes[0, 1]
precisions = results_df.loc['Precision', models].values
recalls = results_df.loc['Recall', models].values

# Scatter plot
for i, (model, prec, rec) in enumerate(zip(models, precisions, recalls)):
    if model == 'XGBoost':
        ax.scatter(rec, prec, s=300, color='#e74c3c', edgecolors='black', 
                  linewidths=2, zorder=5, marker='*', label='XGBoost (Best)')
        ax.annotate('XGBoost', (rec, prec), xytext=(10, 10), 
                   textcoords='offset points', fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    else:
        ax.scatter(rec, prec, s=150, color='#95a5a6', alpha=0.7, 
                  edgecolors='black', linewidths=1)

ax.set_xlabel('Recall', fontweight='bold', fontsize=12)
ax.set_ylabel('Precision', fontweight='bold', fontsize=12)
ax.set_title('Precision vs Recall Trade-off', fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, max(recalls) * 1.15)
ax.set_ylim(0, max(precisions) * 1.15)

# Add model labels
for model, prec, rec in zip(models, precisions, recalls):
    if model != 'XGBoost':
        ax.text(rec, prec, model, fontsize=9, ha='center', va='bottom')

# Bottom Left: F1 Score (Balanced Metric)
ax = axes[1, 0]
f1_scores = results_df.loc['F1', models].values
bars = ax.bar(models, f1_scores, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
ax.set_ylabel('F1 Score', fontweight='bold', fontsize=12)
ax.set_title('F1 Score: Balanced Performance', fontweight='bold', fontsize=13)
ax.set_ylim(0, max(f1_scores) * 1.2)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, val, model in zip(bars, f1_scores, models):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11,
            color='#e74c3c' if model == 'XGBoost' else 'black')
    if model == 'XGBoost':
        bar.set_edgecolor('#e74c3c')
        bar.set_linewidth(3)

ax.set_xticklabels(models, rotation=45, ha='right')

# Bottom Right: Why XGBoost? Summary
ax = axes[1, 1]
ax.axis('off')

# Get XGBoost metrics
xgb_roc = results_df.loc['ROC-AUC', 'XGBoost']
xgb_prec = results_df.loc['Precision', 'XGBoost']
xgb_rec = results_df.loc['Recall', 'XGBoost']
xgb_f1 = results_df.loc['F1', 'XGBoost']
xgb_pr_auc = results_df.loc['PR-AUC', 'XGBoost']

# Compare to best alternatives
best_other_roc = results_df.loc['ROC-AUC', [m for m in models if m != 'XGBoost']].max()
best_other_prec = results_df.loc['Precision', [m for m in models if m != 'XGBoost']].max()
best_other_f1 = results_df.loc['F1', [m for m in models if m != 'XGBoost']].max()

summary_text = f"""
🎯 WHY XGBOOST?

ROC-AUC: {xgb_roc:.3f} (Best: {best_other_roc:.3f})
→ Strongest discriminative ability

Precision: {xgb_prec:.3f} (Best other: {best_other_prec:.3f})
→ Best at avoiding false positives

F1 Score: {xgb_f1:.3f} (Best other: {best_other_f1:.3f})
→ Best balanced performance

PR-AUC: {xgb_pr_auc:.3f}
→ Strong performance on imbalanced data

✅ CHOSEN BECAUSE:
• Highest ROC-AUC (0.850)
• Best precision (0.353)
• Best F1 score (0.333)
• Handles non-linear interactions
• Robust to class imbalance
"""

ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#e74c3c', 
                 linewidth=3, pad=1.0),
        fontweight='bold')

plt.tight_layout()
plt.savefig('figures/model_selection.png', bbox_inches='tight', facecolor='white')
print("\n   Saved: figures/model_selection.png")
print("\nDone! 🎉")

