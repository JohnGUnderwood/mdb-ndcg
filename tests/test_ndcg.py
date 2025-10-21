#!/usr/bin/env python3
"""
Test script for NDCG implementation with example usage scenarios.
"""

import sys
from pathlib import Path

# Add the parent directory to Python path to import ndcg module
sys.path.append(str(Path(__file__).parent.parent))

from ndcg import compute_ndcg_binary, batch_evaluate_ndcg, compute_ndcg_with_clicks


def test_single_query_binary_evaluation():
    """Test single query binary evaluation with ground truth."""
    print("Example 1: Single query binary evaluation with different k values")
    
    # Generate one set of 10 results and select top k for each test
    search_results_full = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7', 'doc8', 'doc9', 'doc10']
    relevant_docs = {'doc1', 'doc3', 'doc5', 'doc7', 'doc9'}  # Ground truth
    
    # Test with different k values using the same data set
    for k in [3, 5, 7, 10]:
        ndcg_score = compute_ndcg_binary(search_results_full, relevant_docs, k=k)
        print(f"NDCG@{k} (binary): {ndcg_score:.4f}")
    print()


def test_batch_evaluation():
    """Test batch evaluation for automated testing."""
    print("Example 2: Batch evaluation with one set of 10 results per query")
    
    # Generate one set of 10 results for each query
    search_results_batch = {
        'query1': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7', 'doc8', 'doc9', 'doc10'],
        'query2': ['doc11', 'doc12', 'doc13', 'doc14', 'doc15', 'doc16', 'doc17', 'doc18', 'doc19', 'doc20'],
        'query3': ['doc21', 'doc22', 'doc23', 'doc24', 'doc25', 'doc26', 'doc27', 'doc28', 'doc29', 'doc30']
    }
    
    ground_truth_batch = {
        'query1': {'doc1', 'doc3', 'doc5', 'doc7'},
        'query2': {'doc11', 'doc15', 'doc19'},
        'query3': {'doc22', 'doc25', 'doc28', 'doc30'}
    }
    
    # Test with different k values using the same 10-result dataset
    for k in [3, 5, 8]:
        results = batch_evaluate_ndcg(search_results_batch, ground_truth_batch, k=k)
        print(f"Batch Evaluation Results for k={k}:")
        print(f"  Average NDCG@{k}: {results['average_ndcg']:.4f}")
        print(f"  Total queries evaluated: {results['total_queries']}")
        
        for query_id, score in results['individual_scores'].items():
            print(f"    {query_id}: {score:.4f}")
        print()


def test_click_based_evaluation():
    """Test click-based evaluation using click-through data."""
    print("Example 3: Click-based evaluation with 10 results and different k values")
    
    # Generate one set of 10 results
    search_results = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7', 'doc8', 'doc9', 'doc10']
    click_data = {
        'doc1': {'clicks': 10},
        'doc2': {'clicks': 2}, 
        'doc3': {'clicks': 8},
        'doc4': {'clicks': 0},
        'doc5': {'clicks': 15},
        'doc6': {'clicks': 1},
        'doc7': {'clicks': 7},
        'doc8': {'clicks': 3},
        'doc9': {'clicks': 12},
        'doc10': {'clicks': 0}
    }
    
    # Test with different k values and thresholds
    for k in [3, 5, 8, 10]:
        for threshold in [5, 7]:
            ndcg_clicks = compute_ndcg_with_clicks(search_results, click_data, click_threshold=threshold, k=k)
            print(f"NDCG@{k} (click-based, threshold={threshold}): {ndcg_clicks:.4f}")
    print()


def test_ranking_quality_comparison():
    """Demonstrate perfect vs imperfect ranking comparison with padding."""
    print("Example 4: Comparison of ranking quality with padding when results < k")
    
    # Test case 1: Full 10 results
    relevant_docs = {'doc1', 'doc3', 'doc5', 'doc7', 'doc9'}
    perfect_results = ['doc1', 'doc3', 'doc5', 'doc7', 'doc9', 'doc2', 'doc4', 'doc6', 'doc8', 'doc10']  # Relevant docs first
    imperfect_results = ['doc2', 'doc1', 'doc4', 'doc3', 'doc6', 'doc5', 'doc8', 'doc7', 'doc10', 'doc9']  # Mixed order
    
    perfect_ndcg = compute_ndcg_binary(perfect_results, relevant_docs, k=10)
    imperfect_ndcg = compute_ndcg_binary(imperfect_results, relevant_docs, k=10)
    
    print(f"Comparison with 10 results:")
    print(f"  Perfect ranking NDCG@10: {perfect_ndcg:.4f}")
    print(f"  Imperfect ranking NDCG@10: {imperfect_ndcg:.4f}")
    if imperfect_ndcg > 0:
        print(f"  Improvement: {((perfect_ndcg - imperfect_ndcg) / imperfect_ndcg * 100):.1f}%")
    
    # Test case 2: Only 3 results with k=10 (demonstrates padding)
    short_perfect = ['doc1', 'doc3', 'doc5']  # Only 3 results but k=10
    short_imperfect = ['doc2', 'doc1', 'doc4']  # Only 3 results but k=10
    
    short_perfect_ndcg = compute_ndcg_binary(short_perfect, relevant_docs, k=10)
    short_imperfect_ndcg = compute_ndcg_binary(short_imperfect, relevant_docs, k=10)
    
    print(f"Comparison with 3 results but k=10 (with padding):")
    print(f"  Perfect ranking NDCG@10: {short_perfect_ndcg:.4f}")
    print(f"  Imperfect ranking NDCG@10: {short_imperfect_ndcg:.4f}")
    if short_imperfect_ndcg > 0:
        print(f"  Improvement: {((short_perfect_ndcg - short_imperfect_ndcg) / short_imperfect_ndcg * 100):.1f}%")
    print()


def test_padding_functionality():
    """Test padding with zeros when search results are fewer than k."""
    print("Example 5: Testing padding functionality when results < k")
    
    # Test case: 3 search results but k=8 (should pad with 5 zeros)
    search_results_short = ['doc1', 'doc2', 'doc3']  # Only 3 results
    relevant_docs = {'doc1', 'doc3'}  # 2 relevant out of 3
    
    # Test with different k values
    for k in [3, 5, 8, 10]:
        ndcg_score = compute_ndcg_binary(search_results_short, relevant_docs, k=k)
        actual_results = len(search_results_short)
        padding_needed = max(0, k - actual_results)
        print(f"NDCG@{k} with {actual_results} results ({padding_needed} padded zeros): {ndcg_score:.4f}")
    
    print()


def main():
    """Run all test examples."""
    print("NDCG Implementation Test Examples")
    print("=" * 40)
    print()
    
    test_single_query_binary_evaluation()
    test_batch_evaluation()
    test_click_based_evaluation()
    test_ranking_quality_comparison()
    test_padding_functionality()


if __name__ == "__main__":
    main()