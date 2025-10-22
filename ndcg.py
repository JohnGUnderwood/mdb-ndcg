from math import log2
from typing import Dict, List, Any

# Compute NDCG given a list of ideal scores, a search result list and a cutoff k
def compute_ndcg(ideal_scores, search_results, k, debug=False) -> float:
    """
    Compute NDCG given ideal scores and search results.
    Uses the formula:
        NDCG@k = DCG@k / IDCG@k
    Where
        DCG@k = Σ(i=1 to k) [rel_i / log2(i + 1)]
        IDCG@k = Σ(i=1 to k) [rel_ideal_i / log2(i + 1)]
    Args:
        ideal_scores: Dict mapping document IDs to their relevance scores
        search_results: List of document IDs ranked by the search system
        k: Number of top results to evaluate
        debug: If True, print detailed NDCG calculation steps
    Returns:
        NDCG@k score between 0 and 1
    """
    if debug:
        print(f"\n📊 NDCG@{k} Calculation Details")
        print("=" * 60)
        print("Formula: NDCG@k = DCG@k / IDCG@k")
        print("Where:")
        print("  DCG@k  = Σ(i=1 to k) [relevance_i / log2(i + 1)]")
        print("  IDCG@k = Σ(i=1 to k) [ideal_relevance_i / log2(i + 1)]")
        print()
    
    # Calculate DCG for search results
    dcg = 0.0
    dcg_components = []
    for i, doc_id in enumerate(search_results[:k], start=1):
        relevance = ideal_scores.get(doc_id, 0)
        discount = log2(i + 1)
        gain = relevance / discount
        dcg += gain
        dcg_components.append({
            'position': i,
            'doc_id': str(doc_id),
            'relevance': relevance,
            'discount': discount,
            'gain': gain
        })
    
    # Calculate IDCG using the ideal scores (sorted in descending order)
    idcg = 0.0
    idcg_components = []
    sorted_scores = sorted(ideal_scores.values(), reverse=True)
    for i, score in enumerate(sorted_scores[:k], start=1):
        discount = log2(i + 1)
        gain = score / discount
        idcg += gain
        idcg_components.append({
            'position': i,
            'relevance': score,
            'discount': discount,
            'gain': gain
        })
    
    ndcg = dcg / idcg if idcg > 0 else 0.0

    if debug:
        print("🔍 DCG Calculation (Actual Search Results):")
        print("Pos | Doc ID              | Relevance | log2(pos+1) | Gain      | Running DCG")
        print("-" * 80)
        running_dcg = 0.0
        for comp in dcg_components:
            running_dcg += comp['gain']
            print(f"{comp['position']:3d} | {comp['doc_id']:19s} | {comp['relevance']:9.3f} | "
                  f"{comp['discount']:11.3f} | {comp['gain']:9.3f} | {running_dcg:11.4f}")
        print(f"                                                     Final DCG@{k}: {dcg:.4f}")
        print()
        
        print("⭐ IDCG Calculation (Ideal Ranking - Best Possible):")
        print("Pos | Ideal Relevance     | log2(pos+1) | Gain      | Running IDCG")
        print("-" * 70)
        running_idcg = 0.0
        for comp in idcg_components:
            running_idcg += comp['gain']
            print(f"{comp['position']:3d} | {comp['relevance']:19.3f} | {comp['discount']:11.3f} | "
                  f"{comp['gain']:9.3f} | {running_idcg:12.4f}")
        print(f"                                           Final IDCG@{k}: {idcg:.4f}")
        print()
        
        print("🎯 Final NDCG Calculation:")
        print(f"NDCG@{k} = DCG@{k} / IDCG@{k}")
        print(f"NDCG@{k} = {dcg:.4f} / {idcg:.4f}")
        print(f"NDCG@{k} = {ndcg:.4f}")

    return ndcg

# Compute score for docs in ideal ranking
def compute_scores(ideal_ranking: List[Any], k: int, method: str = ['binary','inverse_rank','decay','score'], debug=False) -> Dict[str, float]:
    """
    Compute relevance scores for documents in the ideal ranking based on the specified method.
    There are four methods supported, 1 explicit and 3 implicit:
    - [implicit] 'binary' : Binary relevance (1 for relevant, 0 for not relevant) all docs in ideal ranking are implicitly relevant
    - [implicit] 'inverse_rank': Relevance based on inverse of rank position. Higher ranking implies higher relevance.
    - [implicit] 'decay': Exponential decay relevance based on position. Higher ranking implies higher relevance.
    - [explicit] 'score': Fixed scores assigned externally (not computed here). Expertly provided scores, e.g. 0-5 scale.

    INVERSE_RANK METHOD DETAILS:
    Documents are assigned relevance scores which are the inverse of their rank in the ideal ranking:
    - Position 1: relevance = 1/1 = 1.0
    - Position 2: relevance = 1/2 = 0.5
    - Position 3: relevance = 1/3 = 0.333
    - Position i: relevance = 1/i
    - Documents not in ideal ranking: relevance = 0

    This inverse ranking ensures evaluator opinions on higher positions carry much more weight
    than lower positions, which reflects real user behavior with search results.

    DECAY METHOD DETAILS:
    This approach adapts the position 1 score based on 
    the size of the ideal ranking, ensuring that longer ideal rankings (more relevant documents)
    have proportionally higher maximum relevance scores.
    
    The adaptive base relevance is calculated as:
    base_relevance = max(1, scaling_factor * log2(len(ideal_ranking)) + 1)
    
    Each document gets a score of 2^(base_relevance - position), ensuring:
    - ALL documents in the ideal ranking receive non-zero relevance scores
    - Exponential decay provides strong preference for higher-ranked documents
    - The decay continues smoothly beyond the base relevance threshold
    
    Examples:
    - Short ideal ranking (3 docs): base ≈ 2.6 → scores [6.06, 3.03, 1.52]
    - Medium ideal ranking (8 docs): base ≈ 4.0 → scores [16, 8, 4, 2, 1, 0.5, 0.25, 0.125]  
    - Long ideal ranking (20 docs): base ≈ 5.3 → scores [20, 10, 5, 2.5, 1.25, 0.625, 0.313, ...]
    
    Args:
        ideal_ranking: List of document IDs in ideal order (most relevant first)
        k: Number of top results to evaluate for NDCG@K
        method: Scoring method to use ('binary','inverse_rank','decay','score')
        debug: If True, print debug information
    Returns:
        Dictionary mapping document IDs to their computed relevance scores
    """
    scores = {}
    for i, doc_id in enumerate(ideal_ranking, start=1):
        score = 0
        if method == 'binary':
            score = 1  # relevant
        elif method == 'inverse_rank':
            score = 1 / i
        elif method == 'decay':
            # Calculate adaptive base relevance based on ideal ranking length
            ideal_length = len(ideal_ranking)
            scaling_factor = 1.0 # Not currently configurable
            base_relevance = max(1, scaling_factor * log2(ideal_length) + 1)
            exponent = base_relevance - i
            score = 2 ** exponent  # Perfect ranking relevance with adaptive exponential decay
        elif method == 'score':
            try:
                if isinstance(doc_id, dict):
                    score = int(doc_id['score'])
                    doc_id = str(doc_id['doc_id'])
                else:
                    raise ValueError("For 'score' method, ideal_ranking must be a list of dicts with 'doc_id' and 'score' keys")
            except Exception as e:
                raise ValueError("Invalid ideal ranking format for use with 'score' method. Must be a list of dicts: [{'doc_id':<doc_id>,'score':<int:score>}]") from e
        else:
            raise ValueError("Unknown scoring method")

        # Convert doc_id to string for consistent key usage
        scores[str(doc_id)] = score
    return scores

def batch_evaluate_ndcg(search_results_dict, ground_truth_dict, k, debug=False, scoring='binary'):
    """
    Evaluate NDCG for multiple queries in batch for automated testing.
    
    Args:
        search_results_dict: Dict mapping query_id to list of ranked document IDs
        ground_truth_dict: Dict mapping query_id to set/list of relevant document IDs or ideal rankings
        k: Number of top results to evaluate
        debug: If True, print detailed debug information for each query
        scoring: NDCG rank scoring algorithm to use (default: binary)
            'binary' - use binary relevance (0/1) for relevant document sets
            'inverse_rank' - use inverse rank graded relevance
            'decay' - use exponential decay graded relevance
        
    Returns:
        Dict with individual NDCG scores and average NDCG across all queries
        
    Example:
        # For graded relevance with ideal rankings:
        search_results = {
            'query1': ['doc1', 'doc2', 'doc3'],
            'query2': ['doc4', 'doc5', 'doc6']
        }
        ideal_rankings = {
            'query1': ['doc3', 'doc1', 'doc2'],  # doc3 is most relevant
            'query2': ['doc4', 'doc6', 'doc5']   # doc4 is most relevant
        }
        results = batch_evaluate_ndcg(search_results, ideal_rankings, k=3,'inverse_rank')

        # Or for exponential decay graded relevance:
        results = batch_evaluate_ndcg(search_results, ideal_rankings, k=3,'decay')

        # For binary relevance with relevant document sets:
        relevant_docs = {
            'query1': {'doc1', 'doc3'},
            'query2': {'doc4'}
        }
        results = batch_evaluate_ndcg(search_results, relevant_docs, k=3,'binary')
    """
    if not search_results_dict or not ground_truth_dict:
        return {'individual_scores': {}, 'average_ndcg': 0.0, 'total_queries': 0}
    
    individual_scores = {}
    total_ndcg = 0.0
    evaluated_queries = 0
    
    for query_id in search_results_dict:
        if query_id in ground_truth_dict:
            search_results = search_results_dict[query_id]
            ground_truth = ground_truth_dict[query_id]
            
            if debug:
                print(f"\n=== Debug Info for Query: {query_id} ===")
                if scoring in ['inverse_rank','decay'] and isinstance(ground_truth, list):
                    print(f"Ideal ranking: {ground_truth}")
                else:
                    print(f"Relevant documents: {ground_truth}")
                print(f"\nSearch results: {search_results}")
                
                # Show side-by-side comparison
                print(f"\nSide-by-side comparison (top {k}):")
                if scoring in ['inverse_rank','decay'] and isinstance(ground_truth, list):
                    print("Position | Search Result | Ideal Ranking | Match?")
                    print("-" * 55)
                    for i in range(k):
                        search_doc = str(search_results[i]) if i < len(search_results) else "<empty>"
                        ideal_doc = str(ground_truth[i]) if i < len(ground_truth) else "<empty>"
                        match = "✓" if search_doc == ideal_doc else "✗"
                        print(f"{i+1:8d} | {search_doc:25s} | {ideal_doc:25s} | {match:6s}")
                else:
                    print("Position | Search Result | Ideal Ranking | Relevant?")
                    print("-" * 75)
                    relevant_set = set(ground_truth) if isinstance(ground_truth, list) else ground_truth
                    # For binary relevance, show the ideal ranking in order of relevance (all relevant docs first)
                    ideal_list = list(relevant_set) if isinstance(relevant_set, set) else ground_truth
                    for i in range(k):
                        search_doc = str(search_results[i]) if i < len(search_results) else "<empty>"
                        ideal_doc = str(ideal_list[i]) if i < len(ideal_list) else "<empty>"
                        is_relevant = "✓" if (i < len(search_results) and search_results[i] in relevant_set) else "✗"
                        print(f"{i+1:8d} | {search_doc:25s} | {ideal_doc:25s} | {is_relevant:9s}")

            ndcg_score = compute_ndcg(compute_scores(ground_truth, k, method=scoring, debug=debug), search_results, k, debug=debug)

            individual_scores[query_id] = ndcg_score
            total_ndcg += ndcg_score
            evaluated_queries += 1
    
    average_ndcg = total_ndcg / evaluated_queries if evaluated_queries > 0 else 0.0

    return {
        'individual_scores': individual_scores,
        'average_ndcg': average_ndcg,
        'total_queries': evaluated_queries
    }
