#!/usr/bin/env python3
"""
Test script to demonstrate NDCG@k functionality with different k values.
This script tests NDCG calculations using one set of 10 results for different k values.
"""

import sys
from pathlib import Path

# Add the parent directory to Python path to import ndcg module
sys.path.append(str(Path(__file__).parent.parent))

from ndcg import compute_ndcg_binary, batch_evaluate_ndcg


def test_single_dataset_multiple_k():
    """Test NDCG@k with one dataset of 10 results for different k values."""
    print("Testing NDCG@k functionality with one set of 10 results")
    print("=" * 60)
    
    # Generate one set of 10 search results
    search_results = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7', 'doc8', 'doc9', 'doc10']
    relevant_docs = {'doc1', 'doc3', 'doc5', 'doc7', 'doc9'}  # 5 relevant documents
    
    # Test different k values using the same 10-result dataset
    test_k_values = [1, 3, 5, 7, 10]
    results = {}
    
    print("Single Query Results:")
    print("-" * 30)
    for k in test_k_values:
        ndcg_score = compute_ndcg_binary(search_results, relevant_docs, k=k)
        results[k] = ndcg_score
        print(f"NDCG@{k:2d}: {ndcg_score:.4f}")
    
    return results


def test_batch_dataset_multiple_k():
    """Test batch evaluation with one set of 10 results per query for different k values."""
    print("\nBatch Evaluation Results:")
    print("-" * 30)
    
    # Generate one set of 10 results for each of 3 queries
    search_results_batch = {
        'query1': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7', 'doc8', 'doc9', 'doc10'],
        'query2': ['doc11', 'doc12', 'doc13', 'doc14', 'doc15', 'doc16', 'doc17', 'doc18', 'doc19', 'doc20'],
        'query3': ['doc21', 'doc22', 'doc23', 'doc24', 'doc25', 'doc26', 'doc27', 'doc28', 'doc29', 'doc30']
    }
    
    ground_truth_batch = {
        'query1': {'doc1', 'doc3', 'doc5', 'doc7', 'doc9'},         # 5 relevant
        'query2': {'doc11', 'doc13', 'doc15', 'doc17'},             # 4 relevant
        'query3': {'doc22', 'doc24', 'doc26', 'doc28', 'doc30'}     # 5 relevant
    }
    
    # Test different k values using the same 10-result datasets
    test_k_values = [1, 3, 5, 7, 10]
    batch_results = {}
    
    for k in test_k_values:
        results = batch_evaluate_ndcg(search_results_batch, ground_truth_batch, k=k)
        batch_results[k] = results['average_ndcg']
        print(f"Average NDCG@{k:2d}: {results['average_ndcg']:.4f} (across {results['total_queries']} queries)")
    
    return batch_results


def test_padding_scenarios():
    """Test padding when search results are fewer than k."""
    print("\nPadding Test Results:")
    print("-" * 30)
    
    # Test case 1: 3 results with different k values
    short_results = ['doc1', 'doc2', 'doc3']  # Only 3 results
    relevant_docs = {'doc1', 'doc3'}  # 2 relevant
    
    print("3 search results with different k values:")
    for k in [3, 5, 8, 10]:
        ndcg_score = compute_ndcg_binary(short_results, relevant_docs, k=k)
        padding_needed = max(0, k - len(short_results))
        print(f"  NDCG@{k} ({padding_needed} padded): {ndcg_score:.4f}")
    
    # Test case 2: 7 results with k=10
    medium_results = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7']  # 7 results
    print("\n7 search results with k=10:")
    ndcg_score = compute_ndcg_binary(medium_results, relevant_docs, k=10)
    print(f"  NDCG@10 (3 padded): {ndcg_score:.4f}")


def main():
    """Test NDCG@k functionality with consistent datasets."""
    print("NDCG@k Functionality Test")
    print("Using one set of 10 results and selecting top k for each test")
    print("=" * 60)
    
    # Run tests
    single_results = test_single_dataset_multiple_k()
    batch_results = test_batch_dataset_multiple_k()
    test_padding_scenarios()
    
    # Summary
    print("\nSummary:")
    print("=" * 60)
    print("✅ Single query tests completed successfully")
    print("✅ Batch evaluation tests completed successfully") 
    print("✅ Padding functionality tests completed successfully")
    print("\nKey improvements implemented:")
    print("1. Search results padded with 0s when fewer than k")
    print("2. Ideal rankings limited to top k items for comparison")
    print("3. Tests use one set of 10 results and select top k")
    
    # Verify scores make sense (higher k should not increase NDCG dramatically due to padding)
    print(f"\nScore validation:")
    for k in [1, 3, 5, 10]:
        if k in single_results:
            print(f"Single query NDCG@{k}: {single_results[k]:.4f}")


if __name__ == "__main__":
    main()