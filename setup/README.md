# Setup Pipeline

## What I'm Doing

I'm collecting Wikipedia data for deceased and living people, then creating a modeling-ready dataset to predict posthumous cultural legacy.

## Why

I want to predict which people will become "legends" (sustained post-death Wikipedia views) using only pre-death features.

## How to Run

### Quick Start (Full Pipeline)

```bash
cd setup
bash run_full_pipeline.sh
```

This runs all steps automatically. **Note**: Requires database and data already collected (steps 1-2 below).

### Manual Steps (If Starting Fresh)

#### 1. Create Database

```bash
python3 create_clout.py
```

Creates empty database tables.

#### 2. Collect People & Pageviews

```bash
python3 populate_clout.py
```

- Fetches deceased people (2017-2025) from Wikidata
- Downloads Wikipedia pageview data
- **Takes 2-4 hours** (lots of API calls)
- **Output**: Database with people and pageviews

#### 3. Add Metadata

```bash
python3 enrich_wikidata.py
```

- Adds sitelinks, birth year, awards, occupations
- **Takes 30-60 minutes**

#### 4. Create Modeling Dataset

```bash
python3 upgrade_features.py
```

- Engineers features (log transforms, interactions, one-hot encodings)
- Creates legend labels (ratio > 2.5 AND views > 50)
- **Output**: `modeling_data_upgraded.csv`
- **Takes 1-2 minutes**

#### 5. Create Balanced Dataset

```bash
python3 create_balanced_dataset.py
```

- Creates temporally balanced dataset (equal samples per year)
- **Output**: `modeling_data_balanced.csv`
- **Takes <1 minute**

#### 6. Fix Historical Data (Critical - Prevents Data Leakage)

```bash
python3 fix_historical_page_length.py
```

- Fixes data leakage: updates `page_len_bytes` and `num_editors` to historical values (day before death)
- Reads: `modeling_data_balanced.csv`
- **Output**: `modeling_data_balanced_fixed.csv`
- **Takes 1-2 hours** (API calls for each person)

#### 7. Regenerate Log Features

```bash
python3 regenerate_log_features.py
```

- Recomputes log-transformed features after historical fix
- Reads: `modeling_data_balanced_fixed.csv`
- **Output**: Updates `modeling_data_balanced.csv` with corrected features **← Use this for training**
- **Takes <1 minute**

## Output

- `modeling_data_upgraded.csv` - Original dataset (imbalanced years)
- `modeling_data_balanced.csv` - **Final dataset for training** (balanced, with historical fixes) **← Recommended**

**Important**: Always use `modeling_data_balanced.csv` for training (it has historical fixes and balanced sampling).

## Files

- `create_clout.py` - Database schema
- `populate_clout.py` - Data collection
- `enrich_wikidata.py` - Metadata enrichment
- `upgrade_features.py` - Feature engineering & labels
- `create_balanced_dataset.py` - Create balanced year distribution (optional but recommended)

That's it. Run them in order.
