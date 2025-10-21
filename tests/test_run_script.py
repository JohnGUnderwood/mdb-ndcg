#!/usr/bin/env python3
"""
Test script for run_ndcg_evaluation.py

This script tests the run script functionality without requiring MongoDB.
"""

import json
import tempfile
import os
from pathlib import Path
import sys

# Add the parent directory to Python path to import the run script
sys.path.append(str(Path(__file__).parent.parent))

from run_ndcg_evaluation import load_pipeline, inject_query_into_pipeline


def test_pipeline_loading():
    """Test pipeline loading functionality."""
    print("Testing pipeline loading...")
    
    # Create test pipeline
    test_pipeline = [
        {"$match": {"title": {"$regex": "{{QUERY}}", "$options": "i"}}},
        {"$sort": {"score": -1}},
        {"$limit": 10}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_pipeline, f)
        pipeline_file = f.name
    
    try:
        pipeline = load_pipeline(pipeline_file)
        
        assert len(pipeline) == 3
        assert pipeline[0]["$match"]["title"]["$regex"] == "{{QUERY}}"
        print("✓ Pipeline loaded successfully")
        
        # Test query injection
        injected = inject_query_into_pipeline(pipeline, "test query","test_index")
        assert injected[0]["$match"]["title"]["$regex"] == "test query"
        print("✓ Query injection working correctly")
        
    finally:
        os.unlink(pipeline_file)


def test_wrapped_pipeline_loading():
    """Test loading wrapped pipeline format."""
    print("Testing wrapped pipeline loading...")
    
    wrapped_pipeline = {
        "pipeline": [
            {"$match": {"$text": {"$search": "{{QUERY}}"}}},
            {"$sort": {"score": -1}}
        ],
        "metadata": {"description": "Test pipeline"}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(wrapped_pipeline, f)
        pipeline_file = f.name
    
    try:
        pipeline = load_pipeline(pipeline_file)
        
        assert len(pipeline) == 2
        assert pipeline[0]["$match"]["$text"]["$search"] == "{{QUERY}}"
        print("✓ Wrapped pipeline format loaded successfully")
        
    finally:
        os.unlink(pipeline_file)


def test_query_injection():
    """Test query injection functionality."""
    print("Testing query injection...")
    
    pipeline = [
        {"$match": {"title": {"$regex": "{{QUERY}}", "$options": "i"}}},
        {"$addFields": {"search_query": "{{QUERY}}"}}
    ]
    
    injected = inject_query_into_pipeline(pipeline, "machine learning","test_index")
    
    assert injected[0]["$match"]["title"]["$regex"] == "machine learning"
    assert injected[1]["$addFields"]["search_query"] == "machine learning"
    
    print("✓ Query injection working correctly")


def main():
    """Run all tests."""
    print("Running tests for NDCG Evaluation Runner")
    print("=" * 50)
    
    try:
        test_pipeline_loading()
        test_wrapped_pipeline_loading()
        test_query_injection()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("The run script core functions are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()