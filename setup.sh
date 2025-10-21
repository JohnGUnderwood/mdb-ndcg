#!/bin/bash

# Setup script for NDCG evaluation with MongoDB
# This script installs dependencies and sets up sample data

echo "Setting up NDCG Evaluation Environment..."

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install pymongo

echo ""
echo "Setup complete!"
echo ""
echo "Virtual environment created and activated at: .venv/"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Run 'python create_sample_data.py' to create sample data"
# Ask user if they want to create sample data
read -p "Do you want to create sample data now? (y/n): " create_sample

if [ "$create_sample" == "y" ]; then
    echo "Creating sample data..."
    python ./example/create_sample_data.py
fi
echo "3. Test the evaluation with (make sure virtual environment is activated):"
echo "python run_ndcg_evaluation.py --pipeline ./example/atlas_search_pipeline.json"
echo ""