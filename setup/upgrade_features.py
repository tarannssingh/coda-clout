#!/usr/bin/env python3
"""
Coda Clout — Feature Engineering & Label Upgrade

Implements all the critical upgrades:
1. Better legend labeling (ratio > 3.0 + post views > 500/day)
2. Age at death feature
3. Cause of death categories
4. Pre-death fame proxy
5. Log transformations
6. Year-based sampling weights
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import re

DB_PATH = Path(__file__).resolve().parent / "wikipedia_clout.db"
OUTPUT_CSV = Path(__file__).resolve().parent / "modeling_data_upgraded.csv"

# Cause of death categories (map from Wikidata cause strings)
CAUSE_CATEGORIES = {
    'cancer': ['cancer', 'carcinoma', 'tumor', 'leukemia', 'lymphoma', 'malignancy'],
    'heart_disease': ['heart', 'cardiac', 'myocardial', 'stroke', 'cerebrovascular'],
    'accident': ['accident', 'crash', 'collision', 'fall'],
    'suicide': ['suicide', 'self-inflicted'],
    'violence': ['murder', 'homicide', 'assassination', 'killing'],
    'old_age': ['old age', 'natural causes', 'age-related'],
    'covid': ['covid', 'coronavirus', 'sars-cov-2'],
    'overdose': ['overdose', 'drug overdose', 'opioid'],
    'other': []  # catch-all
}

def categorize_cause_of_death(cause_str):
    """Categorize cause of death into groups."""
    if pd.isna(cause_str) or not cause_str:
        return 'unknown'
    
    cause_lower = str(cause_str).lower()
    
    for category, keywords in CAUSE_CATEGORIES.items():
        if category == 'other':
            continue
        if any(keyword in cause_lower for keyword in keywords):
            return category
    
    return 'other'

def compute_sustained_metrics(df_views, page_id, date_of_death):
    """Compute sustained ratio and post-death average views."""
    if pd.isna(date_of_death):
        return None, None, None
    
    page_views = df_views[df_views['page_id'] == page_id].copy()
    if len(page_views) == 0:
        return None, None, None
    
    # Convert date_of_death to datetime
    death_date = pd.to_datetime(date_of_death)
    page_views['date'] = pd.to_datetime(page_views['date'])
    page_views['day_since_death'] = (page_views['date'] - death_date).dt.days
    
    # Pre-death year average (days -365 to -1)
    pre_views = page_views[
        (page_views['day_since_death'] >= -365) & 
        (page_views['day_since_death'] < 0)
    ]['views']
    
    # Post-death sustained average (days 30-365)
    post_views = page_views[
        (page_views['day_since_death'] >= 30) & 
        (page_views['day_since_death'] <= 365)
    ]['views']
    
    if len(pre_views) == 0 or len(post_views) == 0:
        return None, None, None
    
    avg_pre = pre_views.mean()
    avg_post = post_views.mean()
    
    if avg_pre == 0 or pd.isna(avg_pre):
        return None, None, None
    
    sustained_ratio = avg_post / avg_pre
    post_avg_daily = avg_post
    
    # Approximate days in window (335 days for 30-365)
    days_in_window = len(post_views)
    
    return sustained_ratio, post_avg_daily, days_in_window

def main():
    print("=" * 60)
    print("Coda Clout — Feature Engineering & Label Upgrade")
    print("=" * 60)
    
    # Connect to database
    print("\n1. Loading data from database...")
    conn = sqlite3.connect(DB_PATH)
    
    # Load pages
    df_pages = pd.read_sql_query("""
        SELECT 
            id, name, page_title, date_of_death,
            avg_views_pre_death_10d, wikidata_qid,
            sitelinks, birth_year, award_count,
            page_len_bytes, page_watchers, edits_past_year,
            num_editors, cause_of_death, page_created
        FROM wikipedia_page
    """, conn)
    
    print(f"   Loaded {len(df_pages)} pages")
    
    # Load view data
    print("2. Loading view data...")
    df_views = pd.read_sql_query("""
        SELECT page_id, views, date, day_since_death
        FROM wikipedia_daily_clout
    """, conn)
    print(f"   Loaded {len(df_views)} view records")
    
    # Load occupations
    print("3. Loading occupations...")
    df_occupations = pd.read_sql_query("""
        SELECT page_id, occupation
        FROM wikipedia_occupation
    """, conn)
    print(f"   Loaded {len(df_occupations)} occupation records")
    
    conn.close()
    
    # Start feature engineering
    print("\n4. Computing sustained metrics and new labels...")
    df = df_pages.copy()
    
    # Compute sustained ratios for dead people
    sustained_ratios = []
    post_avg_views = []
    days_in_window = []
    
    for idx, row in df.iterrows():
        if pd.notna(row['date_of_death']):
            ratio, post_avg, days = compute_sustained_metrics(
                df_views, row['id'], row['date_of_death']
            )
            sustained_ratios.append(ratio)
            post_avg_views.append(post_avg)
            days_in_window.append(days)
        else:
            sustained_ratios.append(None)
            post_avg_views.append(None)
            days_in_window.append(None)
    
    df['sustained_ratio_31_365'] = sustained_ratios
    df['post_30_365_avg_daily'] = post_avg_views
    df['days_in_post_window'] = days_in_window
    
    print(f"   Computed sustained ratios for {df['sustained_ratio_31_365'].notna().sum()} people")
    
    # NEW LABEL: Use percentile-based (top 15%) OR ratio > 2.5 + post views > 50
    print("\n5. Computing upgraded legend labels...")
    
    # First, compute percentile-based for dead people only
    dead_mask = df['date_of_death'].notna()
    dead_ratios = df.loc[dead_mask, 'sustained_ratio_31_365'].dropna()
    
    if len(dead_ratios) > 0:
        # Top 15% by sustained ratio
        p85_threshold = np.percentile(dead_ratios, 85)
        
        # FIXED THRESHOLD: Objective, time-invariant criteria (NO temporal leakage)
        # This is the CORRECT method - labels can be determined in real-time without future data
        df['high_volume_sustained'] = (
            (df['sustained_ratio_31_365'] > 2.5) & 
            (df['post_30_365_avg_daily'] > 50)
        )
        df['is_legend'] = df['high_volume_sustained'].astype(int)
        method = f"ratio > 2.5 + views > 50 (objective, time-invariant)"
        
        df.loc[df['date_of_death'].isna(), 'is_legend'] = None
        
        legend_count = df.loc[dead_mask, 'is_legend'].sum()
        non_legend_count = (df.loc[dead_mask, 'is_legend'] == 0).sum()
        print(f"   Method: {method}")
        print(f"   Legends: {legend_count}")
        print(f"   Non-legends: {non_legend_count}")
        print(f"   Legend rate: {legend_count / (legend_count + non_legend_count):.1%}")
    else:
        print("   Warning: No dead people with sustained ratios")
    
    # Age at death
    print("\n6. Computing age at death...")
    df['death_year'] = pd.to_datetime(df['date_of_death']).dt.year
    df['age_at_death'] = df['death_year'] - df['birth_year']
    df.loc[df['date_of_death'].isna(), 'age_at_death'] = None
    print(f"   Computed age for {df['age_at_death'].notna().sum()} people")
    print(f"   Age range: {df['age_at_death'].min():.0f} - {df['age_at_death'].max():.0f}")
    
    # Cause of death categories
    print("\n7. Categorizing causes of death...")
    df['cause_category'] = df['cause_of_death'].apply(categorize_cause_of_death)
    cause_counts = df['cause_category'].value_counts()
    print(f"   Cause categories:\n{cause_counts}")
    
    # One-hot encode top occupations
    print("\n8. Encoding occupations...")
    top_occupations = df_occupations['occupation'].value_counts().head(15).index.tolist()
    
    for occ in top_occupations:
        df[f'occ_{occ.replace(" ", "_").replace("/", "_")}'] = 0
    
    for page_id in df['id']:
        page_occs = df_occupations[df_occupations['page_id'] == page_id]['occupation'].tolist()
        for occ in page_occs:
            if occ in top_occupations:
                col_name = f'occ_{occ.replace(" ", "_").replace("/", "_")}'
                df.loc[df['id'] == page_id, col_name] = 1
    
    print(f"   Encoded {len(top_occupations)} top occupations")
    
    # Log transformations
    print("\n9. Applying log transformations...")
    log_features = [
        'avg_views_pre_death_10d', 'sitelinks', 'page_len_bytes',
        'page_watchers', 'edits_past_year', 'num_editors', 'award_count',
        'post_30_365_avg_daily', 'sustained_ratio_31_365'
    ]
    
    for feat in log_features:
        if feat in df.columns:
            df[f'log_{feat}'] = np.log1p(df[feat].fillna(0))
    
    print(f"   Applied log1p to {len(log_features)} features")
    
    # Pre-death fame proxy (FIXED: subtract page_len to account for complexity)
    # Already super-famous people (long pages) can't spike as much
    # New formula: log(views) + log(sitelinks) + awards - 0.5*log(page_len)
    print("\n10. Computing pre-death fame proxy (FIXED formula)...")
    df['fame_proxy'] = (
        np.log1p(df['avg_views_pre_death_10d'].fillna(0)) +
        np.log1p(df['sitelinks'].fillna(0)) +
        df['award_count'].fillna(0) -
        np.log1p(df['page_len_bytes'].fillna(0)) * 0.5  # Subtract complexity (weighted)
    )
    print(f"   Fame proxy range: {df['fame_proxy'].min():.2f} - {df['fame_proxy'].max():.2f}")
    
    # Year-based sampling weights (fix 2018 over-representation)
    print("\n11. Computing year-based sampling weights...")
    df['death_year'] = pd.to_datetime(df['date_of_death']).dt.year
    year_counts = df[df['date_of_death'].notna()]['death_year'].value_counts()
    target_per_year = year_counts.median()  # Use median as target
    
    df['sample_weight'] = 1.0
    for year in year_counts.index:
        count = year_counts[year]
        if count > target_per_year:
            weight = target_per_year / count
            df.loc[df['death_year'] == year, 'sample_weight'] = weight
    
    print(f"   Target per year: {target_per_year:.0f}")
    print(f"   Weights range: {df['sample_weight'].min():.2f} - {df['sample_weight'].max():.2f}")
    
    # Train/val split indicator
    print("\n12. Creating train/val split...")
    df['split'] = 'train'
    df.loc[
        (df['death_year'] >= 2023) & (df['date_of_death'].notna()),
        'split'
    ] = 'val'
    df.loc[df['date_of_death'].isna(), 'split'] = 'control'
    
    train_count = (df['split'] == 'train').sum()
    val_count = (df['split'] == 'val').sum()
    print(f"   Train (2018-2022): {train_count}")
    print(f"   Val (2023-2024): {val_count}")
    print(f"   Control (living): {(df['split'] == 'control').sum()}")
    
    # Filter to dead people only for modeling
    df_model = df[df['date_of_death'].notna()].copy()
    
    # Save
    print(f"\n13. Saving to {OUTPUT_CSV}...")
    df_model.to_csv(OUTPUT_CSV, index=False)
    print(f"   Saved {len(df_model)} rows")
    
    # Final summary
    print("\n" + "=" * 60)
    print("UPGRADE COMPLETE!")
    print("=" * 60)
    print(f"\nFinal dataset:")
    print(f"  Total: {len(df_model)} people")
    print(f"  Legends: {df_model['is_legend'].sum()} ({df_model['is_legend'].mean():.1%})")
    print(f"  Non-legends: {(df_model['is_legend'] == 0).sum()}")
    print(f"  Train: {(df_model['split'] == 'train').sum()}")
    print(f"  Val: {(df_model['split'] == 'val').sum()}")
    print(f"\nFeatures added:")
    print(f"  - sustained_ratio_31_365")
    print(f"  - post_30_365_avg_daily")
    print(f"  - age_at_death")
    print(f"  - cause_category")
    print(f"  - fame_proxy")
    print(f"  - sample_weight")
    print(f"  - {len(top_occupations)} occupation one-hots")
    print(f"  - Log transforms for {len(log_features)} features")
    print(f"\nReady for modeling!")

if __name__ == "__main__":
    main()

