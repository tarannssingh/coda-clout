#!/bin/bash
# Full data enrichment and model retraining pipeline

set -e  # Exit on error

echo "======================================================================"
echo "FULL LEGEND DETECTION PIPELINE"
echo "======================================================================"
echo ""
echo "NOTE: This assumes database and initial data collection are complete."
echo "If starting fresh, run:"
echo "  1. python3 create_clout.py"
echo "  2. python3 populate_clout.py"
echo ""

# Activate virtual environment
cd /Users/taran/CSC466
source .venv/bin/activate
cd coda-clout/setup

# Step 1: Enrich Wikidata
echo "Step 1/7: Enriching Wikidata (occupations, awards, sitelinks)..."
python enrich_wikidata.py
echo ""

# Step 2: Upgrade features
echo "Step 2/7: Upgrading features (Wikipedia page metrics, pageviews)..."
python upgrade_features.py
echo ""

# Step 3: Create balanced dataset
echo "Step 3/7: Creating balanced dataset..."
python create_balanced_dataset.py
echo ""

# Step 4: Fix historical data (CRITICAL - prevents data leakage)
echo "Step 4/7: Fixing historical page length and num_editors (prevents data leakage)..."
echo "  This may take 1-2 hours (API calls for each person)..."
python fix_historical_page_length.py
echo ""

# Step 5: Regenerate log features after historical fix
echo "Step 5/7: Regenerating log features with corrected historical data..."
python regenerate_log_features.py
echo ""

# Step 6: Train models
echo "Step 6/7: Training models..."
cd ../final-report
python train_baselines.py
echo ""

# Step 7: Copy model to detector
echo "Step 7/7: Copying model to detector app..."
cp xgb_model.pkl detector/ 2>/dev/null || echo "  (Model already in detector/)"
cp feature_columns.json detector/ 2>/dev/null || echo "  (Feature columns already in detector/)"
echo ""

echo "======================================================================"
echo "PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Check model performance in final-report/figures/"
echo "2. Test detector: cd detector && streamlit run legend_detector.py"
echo ""
