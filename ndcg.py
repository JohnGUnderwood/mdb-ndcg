from math import log2

# This script implements NDCG evaluation for search systems.
# Discounted Cumulative Gain
# DCG@k = Σ(i=1 to k) [rel_i / log2(i + 1)]

# Ideal DCG (perfect ranking)
# IDCG@k = Σ(i=1 to k) [rel_ideal_i / log2(i + 1)]

# Normalized DCG
# NDCG@k = DCG@k / IDCG@k

# Multiple implementations are provided:
# 1. Manual scoring with graded relevance scores
# 2. Automated binary scoring with ground truth sets
# 3. MongoDB aggregation implementation for production use

# The python implementation
def compute_ndcg(ranking_list, k):
    dcg = sum(rel / log2(i + 1) for i, rel in enumerate(ranking_list[:k], start=1))
    idcg = sum(rel / log2(i + 1) for i, rel in enumerate(sorted(ranking_list, reverse=True)[:k], start=1))
    return dcg / idcg if idcg > 0 else 0

# Automated binary evaluation implementations
def compute_ndcg_binary(search_results, relevant_docs, k, debug=False):
    """
    Compute NDCG using binary relevance (0/1) for automated evaluation.
    
    Args:
        search_results: List of document IDs ranked by the search system
        relevant_docs: Set or list of document IDs that are considered relevant
        k: Number of top results to evaluate
        debug: If True, print detailed debug information about the calculation
        
    Returns:
        NDCG@k score between 0 and 1
        
    Example:
        search_results = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant_docs = {'doc1', 'doc3', 'doc5'}
        ndcg = compute_ndcg_binary(search_results, relevant_docs, k=5)
    """
    if not search_results or not relevant_docs:
        return 0.0
    
    relevant_set = set(relevant_docs)
    # Also add string representations to handle ObjectId/string mismatches
    relevant_set_expanded = set()
    for doc in relevant_set:
        relevant_set_expanded.add(doc)
        relevant_set_expanded.add(str(doc))
    
    # Create binary relevance scores for search results
    ranking_list = []
    for doc_id in search_results[:k]:
        if doc_id in relevant_set_expanded:
            ranking_list.append(1)  # relevant
        else:
            ranking_list.append(0)  # not relevant
    
    # Pad with zeros if search results are fewer than k
    while len(ranking_list) < k:
        ranking_list.append(0)
    
    if debug:
        print(f"\nBinary relevance vector (top {k}): {ranking_list}")
        print(f"Relevant set: {relevant_set}")
        
        # Show relevance calculation step by step
        print("\nRelevance calculation:")
        for i, doc_id in enumerate(search_results[:k]):
            rel = 1 if doc_id in relevant_set_expanded else 0
            print(f"  Position {i+1}: {str(doc_id)} -> {'RELEVANT (1)' if rel else 'NOT RELEVANT (0)'}")
        
        # Show DCG calculation step by step
        print(f"\nDCG@{k} calculation:")
        dcg = 0
        for i, rel in enumerate(ranking_list[:k], start=1):
            gain = rel / log2(i + 1)
            dcg += gain
            print(f"  Position {i}: rel={rel}, log2({i}+1)={log2(i + 1):.3f}, gain={gain:.3f}, DCG so far={dcg:.3f}")
    else:
        # Calculate DCG normally when not debugging
        dcg = sum(rel / log2(i + 1) for i, rel in enumerate(ranking_list[:k], start=1))
    
    # Calculate IDCG - this should be based on the ideal case of having 
    # min(k, len(relevant_docs)) relevant documents in the top k positions
    num_relevant = min(k, len(relevant_set))
    
    if debug:
        print(f"\nIDCG@{k} calculation (with {num_relevant} relevant docs in top {k}):")
        idcg = 0
        for i in range(num_relevant):
            gain = 1 / log2(i + 2)  # i+2 because positions start at 1  
            idcg += gain
            print(f"  Position {i+1}: rel=1, log2({i+1}+1)={log2(i + 2):.3f}, gain={gain:.3f}, IDCG so far={idcg:.3f}")
    else:
        idcg = sum(1 / log2(i + 2) for i in range(num_relevant))  # i+2 because positions start at 1
    
    ndcg = dcg / idcg if idcg > 0 else 0
    
    if debug:
        print(f"Final: DCG={dcg:.3f}, IDCG={idcg:.3f}, NDCG@{k}={ndcg:.3f}")
    
    return ndcg

def compute_ndcg_with_clicks(search_results, click_data, click_threshold, k):
    """
    Compute NDCG using click-through data as relevance indicator.
    
    Args:
        search_results: List of document IDs ranked by the search system
        click_data: Dict mapping doc_id to click metrics {'clicks': int, 'impressions': int}
        click_threshold: Minimum clicks to consider a document relevant
        k: Number of top results to evaluate
        
    Returns:
        NDCG@k score between 0 and 1
        
    Example:
        search_results = ['doc1', 'doc2', 'doc3']
        click_data = {'doc1': {'clicks': 10}, 'doc2': {'clicks': 2}, 'doc3': {'clicks': 0}}
        ndcg = compute_ndcg_with_clicks(search_results, click_data, click_threshold=5, k=3)
    """
    if not search_results or not click_data:
        return 0.0
    
    # Create binary relevance scores based on click threshold
    ranking_list = []
    for doc_id in search_results[:k]:
        clicks = click_data.get(doc_id, {}).get('clicks', 0)
        if clicks >= click_threshold:
            ranking_list.append(1)  # clicked enough = relevant
        else:
            ranking_list.append(0)  # not clicked enough = not relevant
    
    # Pad with zeros if search results are fewer than k
    while len(ranking_list) < k:
        ranking_list.append(0)
    
    return compute_ndcg(ranking_list, k)

def compute_ndcg_inverse_rank(search_results, ideal_ranking, k, debug=False):
    """
    Compute NDCG using graded relevance based on ideal ranking positions.
    
    Documents are assigned relevance scores which are the inverse of their rank in the ideal ranking:
    - Position 1: relevance = 1/1 = 1.0
    - Position 2: relevance = 1/2 = 0.5
    - Position 3: relevance = 1/3 = 0.333
    - Position i: relevance = 1/i
    - Documents not in ideal ranking: relevance = 0

    This inverse ranking ensures evaluator opinions on higher positions carry much more weight
    than lower positions, which reflects real user behavior with search results.
    
    Args:
        search_results: List of document IDs ranked by the search system
        ideal_ranking: List of document IDs in ideal order (most relevant first)
        k: Number of top results to evaluate
        debug: If True, print detailed debug information about the calculation
        base_relevance: Base exponent for relevance calculation (default: 4, giving 2^4=16 for position 1)
        
    Returns:
        NDCG@k score between 0 and 1
    """
    if not search_results or not ideal_ranking:
        return 0.0
    
    # Create relevance mapping based on ideal ranking positions with exponential decay
    relevance_map = {}
    for i, doc_id in enumerate(ideal_ranking):
        # Use inverse ranking: 1/1, 1/2, 1/3, 1/4, 1/5, 0, 0...
        relevance_score = 1 / (i + 1)
        # Store both string and ObjectId representations to handle type mismatches
        doc_id_str = str(doc_id)
        relevance_map[doc_id] = relevance_score
        relevance_map[doc_id_str] = relevance_score
    
    # Calculate DCG for search results
    dcg = 0.0
    ranking_scores = []
    
    if debug:
        print(f"\nRelevance scores based on ideal ranking (inverse ranking):")
        for i, doc_id in enumerate(ideal_ranking[:10]):  # Show first 10
            print(f"  Position {i+1}: {str(doc_id)} -> relevance = 1/{i+1} = {relevance_map[doc_id]}")
        if len(ideal_ranking) > 10:
            print(f"  ... and {len(ideal_ranking) - 10} more documents")
    
    for i, doc_id in enumerate(search_results[:k], start=1):
        relevance = relevance_map.get(doc_id, 0)
        ranking_scores.append(relevance)
        gain = relevance / log2(i + 1)
        dcg += gain
        
        if debug:
            print(f"\nDCG calculation for position {i}:")
            print(f"  Document: {str(doc_id)}")
            print(f"  Relevance: {relevance}")
            print(f"  Discount: 1/log2({i}+1) = 1/{log2(i + 1):.3f} = {1/log2(i + 1):.3f}")
            print(f"  Gain: {relevance}/{log2(i + 1):.3f} = {gain:.3f}")
            print(f"  DCG so far: {dcg:.3f}")
    
    # Calculate IDCG using the ideal ranking with inverse ranking
    idcg = 0.0
    for i in range(min(k, len(ideal_ranking))):
        relevance = 1 / (i + 1)  # Perfect ranking relevance with inverse ranking
        gain = relevance / log2(i + 2)  # i+2 because positions start at 1
        idcg += gain
        
        if debug and i == 0:
            print(f"\nIDCG calculation (ideal ranking with inverse ranking):")
        if debug:
            print(f"  Position {i+1}: relevance=1/{i+1}={relevance}, gain={gain:.3f}, IDCG so far={idcg:.3f}")

    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    if debug:
        print(f"\nFinal calculation:")
        print(f"  DCG@{k}: {dcg:.3f}")
        print(f"  IDCG@{k}: {idcg:.3f}")
        print(f"  NDCG@{k}: {ndcg:.3f}")
    if ndcg == 0:
        print("Warning: NDCG is 0, possibly due to no relevant documents in search results or ideal ranking.")
    return ndcg

def compute_ndcg_decay(search_results, ideal_ranking, k, debug=False, scaling_factor=1.0):
    """
    Compute NDCG using graded relevance with adaptive exponential decay that scales 
    based on the total length of the ideal ranking.
    
    Unlike the fixed exponential decay, this approach adapts the position 1 score based on 
    the size of the ideal ranking, ensuring that longer ideal rankings (more relevant documents)
    have proportionally higher maximum relevance scores.
    
    The adaptive base relevance is calculated as:
    base_relevance = max(1, scaling_factor * log2(len(ideal_ranking)) + 1)
    
    This means:
    - Short ideal ranking (3 docs): base ≈ 1 + 1.58 * scaling_factor = ~2.6 (giving scores 6, 3, 1)
    - Medium ideal ranking (8 docs): base ≈ 1 + 3.0 * scaling_factor = ~4.0 (giving scores 16, 8, 4, 2)  
    - Long ideal ranking (32 docs): base ≈ 1 + 5.0 * scaling_factor = ~6.0 (giving scores 64, 32, 16, 8, 4, 2, 1)
    
    Args:
        search_results: List of document IDs ranked by the search system
        ideal_ranking: List of document IDs in ideal order (most relevant first)
        k: Number of top results to evaluate
        debug: If True, print detailed debug information about the calculation
        scaling_factor: Controls how aggressively the base relevance scales with ranking length (default: 1.0)
        
    Returns:
        NDCG@k score between 0 and 1
    """
    if not search_results or not ideal_ranking:
        return 0.0
    
    # Calculate adaptive base relevance based on ideal ranking length
    ideal_length = len(ideal_ranking)
    base_relevance = max(1, scaling_factor * log2(ideal_length) + 1)
    
    if debug:
        print(f"Adaptive exponential decay parameters:")
        print(f"  Ideal ranking length: {ideal_length}")
        print(f"  Scaling factor: {scaling_factor}")
        print(f"  Calculated base relevance: {base_relevance:.2f}")
    
    # Create relevance mapping based on ideal ranking positions with adaptive exponential decay
    relevance_map = {}
    for i, doc_id in enumerate(ideal_ranking):
        # Use adaptive exponential decay
        exponent = max(0, base_relevance - i)
        relevance_score = 2 ** exponent if exponent > 0 else 0
        # Store both string and ObjectId representations to handle type mismatches
        doc_id_str = str(doc_id)
        relevance_map[doc_id] = relevance_score
        relevance_map[doc_id_str] = relevance_score
    
    # Calculate DCG for search results
    dcg = 0.0
    ranking_scores = []
    
    if debug:
        print(f"\nAdaptive relevance scores (base={base_relevance:.2f}):")
        for i, doc_id in enumerate(ideal_ranking[:10]):  # Show first 10
            exponent = max(0, base_relevance - i)
            print(f"  Position {i+1}: {str(doc_id)} -> relevance = 2^{exponent:.2f} = {relevance_map[doc_id]:.2f}")
        if len(ideal_ranking) > 10:
            print(f"  ... and {len(ideal_ranking) - 10} more documents")
    
    for i, doc_id in enumerate(search_results[:k], start=1):
        relevance = relevance_map.get(doc_id, 0)
        ranking_scores.append(relevance)
        gain = relevance / log2(i + 1)
        dcg += gain
        
        if debug:
            print(f"\nDCG calculation for position {i}:")
            print(f"  Document: {str(doc_id)}")
            print(f"  Relevance: {relevance:.2f}")
            print(f"  Discount: 1/log2({i}+1) = 1/{log2(i + 1):.3f} = {1/log2(i + 1):.3f}")
            print(f"  Gain: {relevance:.2f}/{log2(i + 1):.3f} = {gain:.3f}")
            print(f"  DCG so far: {dcg:.3f}")
    
    # Calculate IDCG using the ideal ranking with adaptive exponential decay
    idcg = 0.0
    for i in range(min(k, len(ideal_ranking))):
        exponent = max(0, base_relevance - i)
        relevance = 2 ** exponent if exponent > 0 else 0  # Perfect ranking relevance with adaptive exponential decay
        gain = relevance / log2(i + 2)  # i+2 because positions start at 1
        idcg += gain
        
        if debug and i == 0:
            print(f"\nIDCG calculation (adaptive exponential decay, base={base_relevance:.2f}):")
        if debug:
            print(f"  Position {i+1}: relevance=2^{exponent:.2f}={relevance:.2f}, gain={gain:.3f}, IDCG so far={idcg:.3f}")
    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    if debug:
        print(f"\nFinal calculation:")
        print(f"  DCG@{k}: {dcg:.3f}")
        print(f"  IDCG@{k}: {idcg:.3f}")
        print(f"  NDCG@{k}: {ndcg:.3f}")
    
    return ndcg

def batch_evaluate_ndcg(search_results_dict, ground_truth_dict, k, debug=False, algorithm='binary'):
    """
    Evaluate NDCG for multiple queries in batch for automated testing.
    
    Args:
        search_results_dict: Dict mapping query_id to list of ranked document IDs
        ground_truth_dict: Dict mapping query_id to set/list of relevant document IDs or ideal rankings
        k: Number of top results to evaluate
        debug: If True, print detailed debug information for each query
        algorithm: NDCG rank scoring algorithm to use (default: binary)
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
                if algorithm in ['inverse_rank','decay'] and isinstance(ground_truth, list):
                    print(f"Ideal ranking: {ground_truth}")
                else:
                    print(f"Relevant documents: {ground_truth}")
                print(f"Search results: {search_results}")
                
                # Show side-by-side comparison
                print(f"\nSide-by-side comparison (top {k}):")
                if algorithm in ['inverse_rank','decay'] and isinstance(ground_truth, list):
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
            
            # Choose the appropriate NDCG calculation method
            if algorithm == 'decay' and isinstance(ground_truth, list):
                # Use graded relevance based on ideal ranking positions with exponential decay
                ndcg_score = compute_ndcg_decay(search_results, ground_truth, k, debug)
            elif algorithm == 'inverse_rank' and isinstance(ground_truth, list):
                # Use graded relevance based on ideal ranking positions with inverse ranking
                ndcg_score = compute_ndcg_inverse_rank(search_results, ground_truth, k, debug)
            elif algorithm == 'binary':
                # Use binary relevance for relevant document sets
                ndcg_score = compute_ndcg_binary(search_results, ground_truth, k, debug)
            
            individual_scores[query_id] = ndcg_score
            total_ndcg += ndcg_score
            evaluated_queries += 1
    
    average_ndcg = total_ndcg / evaluated_queries if evaluated_queries > 0 else 0.0
    
    return {
        'individual_scores': individual_scores,
        'average_ndcg': average_ndcg,
        'total_queries': evaluated_queries
    }
