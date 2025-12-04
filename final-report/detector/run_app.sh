#!/bin/bash
# Run Streamlit app with cache cleared

cd "$(dirname "$0")"
cd ../..

source venv/bin/activate
cd final-report/detector

echo "Starting Streamlit app..."
echo "Cache will be cleared automatically"
echo ""
echo "App will open at: http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

streamlit run legend_detector.py

