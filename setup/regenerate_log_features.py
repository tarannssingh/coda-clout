"""
Regenerate log features from the fixed CSV with historical values.
This updates log_page_len_bytes and log_num_editors based on the new historical values.
"""

import pandas as pd
import numpy as np
from pathlib import Path

CSV_IN = Path(__file__).parent / "modeling_data_balanced_fixed.csv"
CSV_OUT = Path(__file__).parent / "modeling_data_balanced.csv"

def main():
    print("="*60)
    print("REGENERATING LOG FEATURES FROM FIXED CSV")
    print("="*60)
    
    if not CSV_IN.exists():
        print(f"ERROR: {CSV_IN} not found!")
        return
    
    print(f"\nReading: {CSV_IN}")
    df = pd.read_csv(CSV_IN)
    print(f"Loaded {len(df)} rows")
    
    # Regenerate log features that depend on historical values
    print("\nRegenerating log features...")
    
    # Log transformations for features that might have changed
    log_features = [
        'page_len_bytes',  # This was updated to historical
        'num_editors',     # This was updated to historical
        'page_watchers',   # Keep current (can't get historical)
        'avg_views_pre_death_10d',
        'sitelinks',
        'award_count',
        'post_30_365_avg_daily',
        'sustained_ratio_31_365'
    ]
    
    for feat in log_features:
        if feat in df.columns:
            df[f'log_{feat}'] = np.log1p(df[feat].fillna(0))
            print(f"   ✓ log_{feat}")
    
    # Regenerate fame_proxy with updated page_len_bytes
    print("\nRegenerating fame_proxy with historical page_len_bytes...")
    df['fame_proxy'] = (
        np.log1p(df['avg_views_pre_death_10d'].fillna(0)) +
        np.log1p(df['sitelinks'].fillna(0)) +
        df['award_count'].fillna(0) -
        np.log1p(df['page_len_bytes'].fillna(0)) * 0.5
    )
    print(f"   Fame proxy range: {df['fame_proxy'].min():.2f} - {df['fame_proxy'].max():.2f}")
    
    # Save
    print(f"\nSaving to: {CSV_OUT}")
    df.to_csv(CSV_OUT, index=False)
    print(f"✓ Saved {len(df)} rows")
    print("\nReady for training!")

if __name__ == "__main__":
    main()

