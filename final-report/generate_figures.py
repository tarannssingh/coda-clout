"""
Generate publication-quality figures for the final report
Creates visualizations of model performance, feature importance, and dataset statistics
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
print("GENERATING FIGURES FOR FINAL REPORT")
print("="*60)

# Load data
print("\n1. Loading data...")
df = pd.read_csv("../setup/modeling_data_balanced.csv")
df = df[df['date_of_death'].notna()].copy()
print(f"   Loaded {len(df)} samples")

# Load results
results_df = pd.read_csv('figures/results_table.csv', index_col=0)
feature_importance = pd.read_csv('figures/feature_importance.csv')

print("\n2. Generating visualizations...")

# Create figures directory if it doesn't exist
Path('figures').mkdir(exist_ok=True)

# ============================================================================
# FIGURE 1: Model Performance Comparison (Bar Chart)
# ============================================================================
print("   Creating Figure 1: Model Performance Comparison...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)

metrics = ['Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
models = results_df.columns.tolist()
colors = sns.color_palette("husl", len(models))

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    values = results_df.loc[metric, models].values
    bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_title(f'{metric}', fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Rotate x-axis labels
    ax.set_xticklabels(models, rotation=45, ha='right')

# Remove empty subplot
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig('figures/1_model_performance.png', bbox_inches='tight', facecolor='white')
print("      Saved: figures/1_model_performance.png")

# ============================================================================
# FIGURE 2: Feature Importance (Horizontal Bar Chart)
# ============================================================================
print("   Creating Figure 2: Feature Importance...")
fig, ax = plt.subplots(figsize=(10, 8))

# Sort by importance
feature_importance_sorted = feature_importance.sort_values('importance', ascending=True)

# Create horizontal bar chart
bars = ax.barh(feature_importance_sorted['feature'], feature_importance_sorted['importance'],
               color=plt.cm.viridis(feature_importance_sorted['importance'] / feature_importance_sorted['importance'].max()),
               edgecolor='black', linewidth=0.8)

ax.set_xlabel('Feature Importance', fontweight='bold')
ax.set_title('Top Features Predicting Posthumous Wikipedia Legacy', fontweight='bold', fontsize=14)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (idx, row) in enumerate(feature_importance_sorted.iterrows()):
    ax.text(row['importance'] + 0.005, i, f'{row["importance"]:.3f}',
            va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('figures/2_feature_importance.png', bbox_inches='tight', facecolor='white')
print("      Saved: figures/2_feature_importance.png")

# ============================================================================
# FIGURE 3: Class Distribution
# ============================================================================
print("   Creating Figure 3: Class Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Overall class distribution
legend_counts = df['is_legend'].value_counts()
colors_pie = ['#ff6b6b', '#4ecdc4']

# Get counts
total_people = len(df)
non_legend_count = legend_counts.get(0, 0)
legend_count = legend_counts.get(1, 0)

# Create pie chart
axes[0].pie(legend_counts.values, labels=['Not Legend', 'Legend'], autopct='%1.1f%%',
            colors=colors_pie, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[0].set_title('Overall Class Distribution', fontweight='bold', fontsize=13)

# Add total counts text below pie chart
count_text = f'Total: {total_people:,} people\n'
count_text += f'Not Legend: {non_legend_count:,} ({non_legend_count/total_people*100:.1f}%)\n'
count_text += f'Legend: {legend_count:,} ({legend_count/total_people*100:.1f}%)'

axes[0].text(0, -1.3, count_text, ha='center', va='top', fontsize=10, 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', 
            alpha=0.9, edgecolor='black', pad=0.5))

# By year (2017 onward only) - Stacked bar chart showing legends vs non-legends
if 'death_year' in df.columns:
    # Filter to 2017 onward
    df_yearly = df[df['death_year'] >= 2017].copy()
    yearly_legend = df_yearly.groupby('death_year')['is_legend'].agg(['sum', 'count'])
    yearly_legend['non_legend'] = yearly_legend['count'] - yearly_legend['sum']
    yearly_legend['pct'] = (yearly_legend['sum'] / yearly_legend['count'] * 100).round(1)
    
    # Create stacked bar chart
    years = yearly_legend.index
    legend_counts = yearly_legend['sum'].values
    non_legend_counts = yearly_legend['non_legend'].values
    
    # Stack: non-legends on bottom, legends on top
    bars1 = axes[1].bar(years, non_legend_counts, color='#ff6b6b', alpha=0.8, 
                        edgecolor='black', linewidth=1.2, label='Not Legend')
    bars2 = axes[1].bar(years, legend_counts, bottom=non_legend_counts, color='#4ecdc4', 
                        alpha=0.8, edgecolor='black', linewidth=1.2, label='Legend')
    
    axes[1].set_xlabel('Death Year', fontweight='bold')
    axes[1].set_ylabel('Number of Samples', fontweight='bold')
    axes[1].set_title('Samples per Year: Legends vs Non-Legends (2017-2025)', fontweight='bold', fontsize=13)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_xlim(2016.5, yearly_legend.index.max() + 0.5)
    axes[1].set_xticks(years)
    axes[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    # Add sample count inside red box and percentage on top
    for year, total, pct, non_legend_count in zip(years, yearly_legend['count'], yearly_legend['pct'], non_legend_counts):
        # Show sample count in the middle of the red (Not Legend) portion
        mid_red = non_legend_count / 2
        axes[1].text(year, mid_red, f'n={total}', 
                    ha='center', va='center', fontsize=10, fontweight='bold', 
                    color='white', bbox=dict(boxstyle='round', facecolor='#ff6b6b', 
                    alpha=0.8, edgecolor='black', pad=0.3))
        # Show percentage on top if there are legends
        if pct > 0:
            axes[1].text(year, total + 3, f'{pct:.1f}%', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2d3436')

plt.tight_layout()
plt.savefig('figures/3_class_distribution.png', bbox_inches='tight', facecolor='white')
print("      Saved: figures/3_class_distribution.png")

# ============================================================================
# FIGURE 4: Key Feature Distributions (Legends vs Non-Legends)
# ============================================================================
print("   Creating Figure 4: Feature Distributions...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feature Distributions: Legends vs Non-Legends', fontsize=16, fontweight='bold')

# Log views
if 'log_avg_views_pre_death_10d' in df.columns:
    legend_views = df[df['is_legend'] == 1]['log_avg_views_pre_death_10d']
    non_legend_views = df[df['is_legend'] == 0]['log_avg_views_pre_death_10d']
    
    axes[0, 0].hist(non_legend_views, bins=30, alpha=0.6, label='Not Legend', color='#ff6b6b', edgecolor='black')
    axes[0, 0].hist(legend_views, bins=30, alpha=0.6, label='Legend', color='#4ecdc4', edgecolor='black')
    axes[0, 0].set_xlabel('Log(Average Daily Views)', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Pre-Death Attention (Top Predictor)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

# Age at death
if 'age_at_death' in df.columns:
    legend_age = df[df['is_legend'] == 1]['age_at_death']
    non_legend_age = df[df['is_legend'] == 0]['age_at_death']
    
    axes[0, 1].hist(non_legend_age, bins=30, alpha=0.6, label='Not Legend', color='#ff6b6b', edgecolor='black')
    axes[0, 1].hist(legend_age, bins=30, alpha=0.6, label='Legend', color='#4ecdc4', edgecolor='black')
    axes[0, 1].set_xlabel('Age at Death (years)', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Age at Death (3rd Most Important)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

# Page length
if 'log_page_len_bytes' in df.columns:
    legend_page = df[df['is_legend'] == 1]['log_page_len_bytes']
    non_legend_page = df[df['is_legend'] == 0]['log_page_len_bytes']
    
    axes[1, 0].hist(non_legend_page, bins=30, alpha=0.6, label='Not Legend', color='#ff6b6b', edgecolor='black')
    axes[1, 0].hist(legend_page, bins=30, alpha=0.6, label='Legend', color='#4ecdc4', edgecolor='black')
    axes[1, 0].set_xlabel('Log(Page Length in bytes)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Wikipedia Page Size (4th Most Important)', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

# Edits past year
if 'log_edits_past_year' in df.columns:
    legend_edits = df[df['is_legend'] == 1]['log_edits_past_year']
    non_legend_edits = df[df['is_legend'] == 0]['log_edits_past_year']
    
    axes[1, 1].hist(non_legend_edits, bins=30, alpha=0.6, label='Not Legend', color='#ff6b6b', edgecolor='black')
    axes[1, 1].hist(legend_edits, bins=30, alpha=0.6, label='Legend', color='#4ecdc4', edgecolor='black')
    axes[1, 1].set_xlabel('Log(Edits in Past Year)', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title('Editorial Activity (2nd Most Important)', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/4_feature_distributions.png', bbox_inches='tight', facecolor='white')
print("      Saved: figures/4_feature_distributions.png")

# ============================================================================
# FIGURE 5: Temporal Analysis
# ============================================================================
print("   Creating Figure 5: Temporal Analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if 'death_year' in df.columns:
    # Sample count by year
    yearly_counts = df['death_year'].value_counts().sort_index()
    axes[0].bar(yearly_counts.index, yearly_counts.values, color='#667eea', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Death Year', fontweight='bold')
    axes[0].set_ylabel('Number of Samples', fontweight='bold')
    axes[0].set_title('Dataset: Samples per Year (Temporally Balanced)', fontweight='bold', fontsize=13)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Legend count by year
    yearly_legend_counts = df.groupby('death_year')['is_legend'].sum().sort_index()
    axes[1].bar(yearly_legend_counts.index, yearly_legend_counts.values, color='#4ecdc4', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Death Year', fontweight='bold')
    axes[1].set_ylabel('Number of Legends', fontweight='bold')
    axes[1].set_title('Legends Identified per Year', fontweight='bold', fontsize=13)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figures/5_temporal_analysis.png', bbox_inches='tight', facecolor='white')
print("      Saved: figures/5_temporal_analysis.png")

# ============================================================================
# FIGURE 6: Model Comparison (Radar/Spider Chart)
# ============================================================================
print("   Creating Figure 6: Model Comparison Radar Chart...")
from math import pi

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Normalize metrics to 0-1 scale for radar chart
metrics_normalized = results_df.copy()
for metric in metrics_normalized.index:
    max_val = metrics_normalized.loc[metric].max()
    if max_val > 0:
        metrics_normalized.loc[metric] = metrics_normalized.loc[metric] / max_val

# Number of metrics
categories = list(metrics_normalized.index)
N = len(categories)

# Compute angle for each metric
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Complete the circle

# Plot each model
for model in models:
    values = list(metrics_normalized.loc[:, model].values)
    values += values[:1]  # Complete the circle
    ax.plot(angles, values, 'o-', linewidth=2, label=model)
    ax.fill(angles, values, alpha=0.15)

# Add category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.set_title('Model Performance Comparison (Normalized)', fontweight='bold', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig('figures/6_model_radar.png', bbox_inches='tight', facecolor='white')
print("      Saved: figures/6_model_radar.png")

# ============================================================================
# FIGURE 7: Summary Statistics Table
# ============================================================================
print("   Creating Figure 7: Summary Statistics...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Create summary statistics
summary_data = []
summary_data.append(['Total Samples', f'{len(df):,}'])
summary_data.append(['Legends', f'{df["is_legend"].sum():,} ({df["is_legend"].mean()*100:.1f}%)'])
summary_data.append(['Non-Legends', f'{(df["is_legend"]==0).sum():,} ({(df["is_legend"]==0).mean()*100:.1f}%)'])

if 'death_year' in df.columns:
    summary_data.append(['Year Range', f'{df["death_year"].min()}-{df["death_year"].max()}'])
    summary_data.append(['Avg Samples/Year', f'{df.groupby("death_year").size().mean():.0f}'])

summary_data.append(['Features Used', '16 engineered features'])
summary_data.append(['Best Model', 'XGBoost'])
summary_data.append(['ROC-AUC', f'{results_df.loc["ROC-AUC", "XGBoost"]:.3f}'])
summary_data.append(['Precision', f'{results_df.loc["Precision", "XGBoost"]:.3f}'])
summary_data.append(['Recall', f'{results_df.loc["Recall", "XGBoost"]:.3f}'])

table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                colLabels=['Metric', 'Value'], colWidths=[0.4, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Style the table
for i in range(len(summary_data) + 1):
    for j in range(2):
        cell = table[(i, j)]
        if i == 0:  # Header
            cell.set_facecolor('#667eea')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
            cell.set_text_props(weight='bold' if j == 0 else 'normal')

ax.set_title('Dataset and Model Summary Statistics', fontweight='bold', fontsize=14, pad=20)

plt.savefig('figures/7_summary_stats.png', bbox_inches='tight', facecolor='white')
print("      Saved: figures/7_summary_stats.png")

# ============================================================================
# FIGURE 8: Top Features Comparison (Box Plot)
# ============================================================================
print("   Creating Figure 8: Top Features Box Plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Top 4 Features: Legends vs Non-Legends', fontsize=16, fontweight='bold')

top_features = [
    ('log_avg_views_pre_death_10d', 'Pre-Death Attention (16.7%)'),
    ('log_edits_past_year', 'Editorial Activity (12.6%)'),
    ('age_at_death', 'Age at Death (11.2%)'),
    ('log_page_len_bytes', 'Page Size (7.2%)')
]

for idx, (feature, title) in enumerate(top_features):
    ax = axes[idx // 2, idx % 2]
    if feature in df.columns:
        data_to_plot = [df[df['is_legend'] == 0][feature].dropna(),
                       df[df['is_legend'] == 1][feature].dropna()]
        bp = ax.boxplot(data_to_plot, labels=['Not Legend', 'Legend'], 
                       patch_artist=True, widths=0.6)
        
        # Color the boxes
        colors_box = ['#ff6b6b', '#4ecdc4']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Feature Value', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figures/8_top_features_boxplot.png', bbox_inches='tight', facecolor='white')
print("      Saved: figures/8_top_features_boxplot.png")

# ============================================================================
# FIGURE 9: Feature Correlations with Legend Status
# ============================================================================
print("   Creating Figure 9: Feature Correlations...")
fig, axes = plt.subplots(2, 1, figsize=(14, 12))
fig.suptitle('What Makes Someone a Legend? Feature Correlations', fontsize=16, fontweight='bold', y=0.995)

# Get numeric features that exist in the dataset
feature_mapping = {
    'log_avg_views_pre_death_10d': 'Pre-Death Attention',
    'log_edits_past_year': 'Editorial Activity',
    'age_at_death': 'Age at Death',
    'log_page_len_bytes': 'Page Size',
    'log_sitelinks': 'Global Recognition',
    'log_award_count': 'Awards',
    'death_year': 'Death Year',
    'fame_proxy': 'Fame Proxy',
    'views_per_sitelink': 'Attention Efficiency',
    'age_x_fame': 'Age × Fame',
    'age_x_year': 'Age × Year'
}

numeric_features = []
for col in feature_mapping.keys():
    if col in df.columns:
        numeric_features.append(col)

# Calculate correlations with is_legend
correlations = []
for feat in numeric_features:
    if df[feat].dtype in ['int64', 'float64']:
        corr = df[feat].corr(df['is_legend'])
        if not np.isnan(corr):
            correlations.append({
                'feature': feat,
                'name': feature_mapping[feat],
                'correlation': corr
            })

corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=True)

# Top plot: Horizontal bar chart of correlations
colors_corr = ['#4ecdc4' if x > 0 else '#ff6b6b' for x in corr_df['correlation']]
bars = axes[0].barh(range(len(corr_df)), corr_df['correlation'], color=colors_corr, 
                    alpha=0.85, edgecolor='black', linewidth=1.5, height=0.7)
axes[0].axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.5)
axes[0].set_yticks(range(len(corr_df)))
axes[0].set_yticklabels(corr_df['name'], fontsize=11, fontweight='bold')
axes[0].set_xlabel('Correlation with Legend Status', fontweight='bold', fontsize=12)
axes[0].set_title('Feature Correlations: Higher Values → More/Less Likely to be Legend', 
                  fontweight='bold', fontsize=13, pad=15)
axes[0].set_xlim(-0.3, 0.3)
axes[0].grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)

# Add value labels and interpretation
for i, (idx, row) in enumerate(corr_df.iterrows()):
    corr_val = row['correlation']
    label_x = corr_val + (0.015 if corr_val > 0 else -0.015)
    axes[0].text(label_x, i, f'{corr_val:.3f}',
                va='center', ha='left' if corr_val > 0 else 'right',
                fontweight='bold', fontsize=10)
    
    # Add interpretation text
    if abs(corr_val) > 0.1:
        interpretation = 'Strong' if abs(corr_val) > 0.15 else 'Moderate'
        direction = 'More Legend' if corr_val > 0 else 'Less Legend'
        axes[0].text(0.32, i, f'→ {interpretation} predictor of {direction.lower()}', 
                    va='center', fontsize=9, style='italic', color='#495057')

# Bottom plot: Key insight - Edits comparison (most important feature)
if 'log_edits_past_year' in df.columns:
    legend_edits = df[df['is_legend'] == 1]['log_edits_past_year'].dropna()
    non_legend_edits = df[df['is_legend'] == 0]['log_edits_past_year'].dropna()
    
    # Convert back to actual edit counts for easier interpretation
    legend_edits_actual = np.expm1(legend_edits)
    non_legend_edits_actual = np.expm1(non_legend_edits)
    
    # Box plot with actual values
    bp = axes[1].boxplot([non_legend_edits_actual, legend_edits_actual], 
                        labels=['Not Legend', 'Legend'],
                        patch_artist=True, widths=0.6, showmeans=True)
    
    # Color the boxes
    colors_box = ['#ff6b6b', '#4ecdc4']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style means and medians
    for element in ['means', 'medians']:
        if element in bp:
            for line in bp[element]:
                line.set_color('black')
                line.set_linewidth(2)
    
    axes[1].set_ylabel('Number of Edits in Year Before Death', fontweight='bold', fontsize=12)
    axes[1].set_title('Does More Page Edits = More Likely to be a Legend?', 
                      fontweight='bold', fontsize=14, pad=15)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add clear statistics
    non_legend_mean = non_legend_edits_actual.mean()
    legend_mean = legend_edits_actual.mean()
    diff = legend_mean - non_legend_mean
    pct_diff = (diff / non_legend_mean * 100) if non_legend_mean > 0 else 0
    
    # Answer the question: Does high edits = legend?
    answer = "YES" if legend_mean > non_legend_mean else "NO"
    answer_color = '#28a745' if legend_mean > non_legend_mean else '#dc3545'
    
    stats_text = f'📊 ANSWER: High edits = Legend? {answer}\n\n'
    stats_text += f'Not Legend: {non_legend_mean:.0f} edits/year\n'
    stats_text += f'Legend: {legend_mean:.0f} edits/year\n'
    stats_text += f'Difference: {diff:+.0f} ({pct_diff:+.1f}% higher)\n\n'
    stats_text += f'This shows: People whose Wikipedia pages\n'
    stats_text += f'were actively edited before death are\n'
    stats_text += f'more likely to become legends.'
    
    axes[1].text(0.98, 0.02, stats_text, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, 
                         edgecolor=answer_color, linewidth=2, pad=0.8),
                fontweight='bold', color=answer_color)

plt.tight_layout()
plt.savefig('figures/9_feature_correlations.png', bbox_inches='tight', facecolor='white')
print("      Saved: figures/9_feature_correlations.png")

print("\n" + "="*60)
print("ALL FIGURES GENERATED SUCCESSFULLY!")
print("="*60)
print("\nGenerated files:")
for i in range(1, 10):
    print(f"  - figures/{i}_*.png")

print("\nDone! 🎉")

