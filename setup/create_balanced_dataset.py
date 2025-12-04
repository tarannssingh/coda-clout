#!/usr/bin/env python3
"""
Create a balanced dataset by sampling equally from each year
Keeps original data intact, creates new balanced CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_CSV = Path(__file__).resolve().parent / "modeling_data_upgraded.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "modeling_data_balanced.csv"

print("="*60)
print("Creating Balanced Dataset")
print("="*60)

# Load original data
print("\n1. Loading original dataset...")
df = pd.read_csv(INPUT_CSV)
print(f"   Original: {len(df)} people")

# Check year distribution
print("\n2. Current year distribution:")
year_counts = df['death_year'].value_counts().sort_index()
print(year_counts)
min_per_year = year_counts.min()
print(f"\n   Min per year: {min_per_year}")
print(f"   Max per year: {year_counts.max()}")

# Target: sample min_per_year from each year (or a reasonable target)
# Let's use the median to get a good balance
target_per_year = int(year_counts.median())
print(f"\n   Target per year: {target_per_year}")

# Sample equally from each year
print("\n3. Sampling equally from each year...")
balanced_samples = []

for year in sorted(df['death_year'].dropna().unique()):
    year_data = df[df['death_year'] == year].copy()
    
    if len(year_data) >= target_per_year:
        # Sample target_per_year randomly
        sampled = year_data.sample(n=target_per_year, random_state=42)
    else:
        # If year has fewer than target, take all
        sampled = year_data
    
    balanced_samples.append(sampled)
    print(f"   {int(year)}: {len(sampled)}/{len(year_data)} sampled")

# Combine all samples
df_balanced = pd.concat(balanced_samples, ignore_index=True)

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n4. Balanced dataset created:")
print(f"   Total: {len(df_balanced)} people")
print(f"   Legends: {df_balanced['is_legend'].sum()} ({df_balanced['is_legend'].mean():.1%})")

# Check new year distribution
print("\n5. New year distribution:")
new_year_counts = df_balanced['death_year'].value_counts().sort_index()
print(new_year_counts)
print(f"\n   Min per year: {new_year_counts.min()}")
print(f"   Max per year: {new_year_counts.max()}")
print(f"   Std dev: {new_year_counts.std():.1f}")

# Check train/val split
print("\n6. Train/Val split:")
print(df_balanced['split'].value_counts())
train_legends = df_balanced[df_balanced['split'] == 'train']['is_legend'].sum()
val_legends = df_balanced[df_balanced['split'] == 'val']['is_legend'].sum()
print(f"   Train legends: {train_legends}")
print(f"   Val legends: {val_legends}")

# Save
print(f"\n7. Saving balanced dataset to {OUTPUT_CSV}...")
df_balanced.to_csv(OUTPUT_CSV, index=False)
print(f"   ✅ Saved {len(df_balanced)} rows")

print("\n" + "="*60)
print("BALANCED DATASET CREATED!")
print("="*60)
print(f"\nOriginal: {len(df)} people")
print(f"Balanced: {len(df_balanced)} people")
print(f"Reduction: {len(df) - len(df_balanced)} people ({100*(len(df)-len(df_balanced))/len(df):.1f}%)")
print(f"\n✅ Original data preserved in: {INPUT_CSV.name}")
print(f"✅ Balanced data saved to: {OUTPUT_CSV.name}")

