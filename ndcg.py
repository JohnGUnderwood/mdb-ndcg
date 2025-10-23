from math import log2
from typing import Dict, List, Any

# Compute NDCG given a list of ideal scores, a search result list and a cutoff k
def compute_ndcg(ideal_ranking, search_results, k, method='binary', debug=False) -> float:
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

    ideal_scores = compute_scores(ideal_ranking, method=method)

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
        # If using binary score get 1 if doc in ideal ranking @ k else 0
        if method == 'binary':
            relevance = 1 if doc_id in list(ideal_scores.keys())[:k] else 0
        # If using explicit or implicit scores get score
        else:
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
    
    # Calculate IDCG using the ideal scores 
    idcg = 0.0
    idcg_components = []
    if method != 'binary':
        scores = list(ideal_scores.values())
    else:
        # For binary relevance the scores are 1 for all relevant docs
        scores = [1] * min(len(ideal_ranking), k)
    for i, score in enumerate(scores[:k], start=1):
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
def compute_scores(ideal_ranking: List[Any], method: str = ['binary','inverse_rank','decay','score']) -> Dict[str, float]:
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
    for i, doc in enumerate(ideal_ranking, start=1):
        # Check if we are iterating through dicts with scores or just IDs
        if isinstance(doc, dict):
            if 'doc_id' not in doc or 'score' not in doc:
                raise ValueError("For dict entries in ideal_ranking, 'doc_id' and 'score' keys are required")
            doc_id = doc.get('doc_id')
        else:
            doc_id = doc

        score = 0
        if method == 'binary':
            score = 1  # All docs in ideal ranking are relevant
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
                if isinstance(doc, dict):
                    score = float(doc['score'])
                    doc_id = str(doc['doc_id'])
                else:
                    raise ValueError("For 'score' method, ideal_ranking must be a list of dicts with 'doc_id' and 'score' keys")
            except Exception as e:
                raise ValueError("Invalid ideal ranking format for use with 'score' method. Must be a list of dicts: [{'doc_id':<doc_id>,'score':<score>}]") from e
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
                # Ground truth might be a list of Ids or Dicts with scores
                # For debugging we just need list of Ids
                if isinstance(ground_truth, list) and all(isinstance(item, dict) for item in ground_truth):
                    gt_ids = [str(item['doc_id']) for item in ground_truth]
                else:
                    gt_ids = ground_truth

                print(f"\n=== Debug Info for Query: {query_id} ===")
                if scoring in ['inverse_rank','decay'] and isinstance(gt_ids, list):
                    print(f"Ideal ranking: {gt_ids}")
                else:
                    print(f"Relevant documents: {gt_ids}")
                print(f"\nSearch results: {search_results}")
                
                # Show side-by-side comparison
                print(f"\nSide-by-side comparison (top {k}):")
                if scoring == 'binary' and isinstance(gt_ids, list):
                    print("Position | Search Result | Relevant? | Ideal Ranking")
                    print("-" * 75)
                    for i in range(k):
                        search_doc = str(search_results[i]) if i < len(search_results) else "<empty>"
                        ideal_doc = str(gt_ids[i]) if i < len(gt_ids) else "<empty>"
                        is_relevant = "✓" if (i < len(search_results) and search_results[i] in gt_ids[:k]) else "✗"
                        print(f"{i+1:8d} | {search_doc:25s} | {is_relevant:9s} | {ideal_doc:25s} ")
                else:
                    print("Position | Search Result | Ideal Ranking | Match?")
                    print("-" * 55)
                    for i in range(k):
                        search_doc = str(search_results[i]) if i < len(search_results) else "<empty>"
                        ideal_doc = str(gt_ids[i]) if i < len(gt_ids) else "<empty>"
                        match = "✓" if search_doc == ideal_doc else "✗"
                        print(f"{i+1:8d} | {search_doc:25s} | {ideal_doc:25s} | {match:6s}")

            ndcg_score = compute_ndcg(ground_truth, search_results, k, method=scoring, debug=debug)

            individual_scores[query_id] = ndcg_score
            total_ndcg += ndcg_score
            evaluated_queries += 1
    
    average_ndcg = total_ndcg / evaluated_queries if evaluated_queries > 0 else 0.0

    return {
        'individual_scores': individual_scores,
        'average_ndcg': average_ndcg,
        'total_queries': evaluated_queries
    }

"""
Simplified NDCG Evaluation Runner

These functions evaluate NDCG using MongoDB search pipelines and ideal ranking lists.
"""

import argparse
import json
import sys
from typing import Dict, List, Any, Union
from pathlib import Path
from bson import ObjectId
from bson.binary import Binary 

def load_pipeline(pipeline_file: str) -> List[Dict[str, Any]]:
    """Load MongoDB aggregation pipeline from JSON file."""
    pipeline_path = Path(pipeline_file)
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")
    
    with open(pipeline_path, 'r') as f:
        pipeline_data = json.load(f)
    
    # Support both direct pipeline array and wrapped format
    if isinstance(pipeline_data, list):
        return pipeline_data
    elif 'pipeline' in pipeline_data:
        return pipeline_data['pipeline']
    else:
        raise ValueError("Invalid pipeline file format. Expected array or object with 'pipeline' key.")


def inject_query_into_pipeline(pipeline: List[Dict[str, Any]], query: Union[str,Binary,List[float],List[int]], index_name: str) -> List[Dict[str, Any]]:
    """Inject query into pipeline by replacing {{QUERY}} and {{INDEX_NAME}} placeholders."""
    def recursive_replace(obj):
        if isinstance(obj, dict):
            return {k: recursive_replace(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_replace(item) for item in obj]
        elif obj == '{{QUERY}}':
            # Direct replacement for {{QUERY}} placeholder - supports any query type
            return query
        elif isinstance(obj, str):
            # Replace {{INDEX_NAME}} placeholder in string values
            result = obj.replace('{{INDEX_NAME}}', index_name)
            # Replace {{QUERY}} placeholder in string values only if query is a string
            if isinstance(query, str):
                result = result.replace('{{QUERY}}', query)
            return result
        else:
            return obj
    
    return recursive_replace(pipeline)


def get_queries_from_ideal_rankings(db, filter: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve queries and their ideal rankings from MongoDB.
    
    Expected collection schema:
    {
        "query_id": "unique_identifier",
        "query": "search query text",
        "collection": "documents", 
        "ideal_ranking": ["doc1", "doc2", "doc3", ...]
    }
    """
    collection = db.ideal_rankings
    
    queries = {}
    cursor = collection.find(filter, {"query_id": 1, "query": 1, "ideal_ranking": 1})
    
    for doc in cursor:
        query_id = doc.get('query_id')
        query = doc.get('query')
        ideal_ranking = doc.get('ideal_ranking', [])
        
        if query_id and query:
            queries[query_id] = {
                'query': query,
                'ideal_ranking': ideal_ranking
            }

    return queries

def execute_search_pipeline(search_collection, search_index, pipeline: List[Dict[str, Any]], query: Union[str,Binary,List[float],List[int]]) -> List[str]:
    """Execute the search pipeline with injected query and return ranked document IDs."""
    
    # Inject query into pipeline
    injected_pipeline = inject_query_into_pipeline(pipeline, query, search_index)
    
    # Execute pipeline on documents collection
    results = list(search_collection.aggregate(injected_pipeline))
    
    # Extract document IDs from results  
    document_ids = []
    for doc in results:
        doc_id = doc.get('_id')
        if doc_id:
            document_ids.append(str(doc_id))
    
    return document_ids

def run(args):
    """
    Run NDCG evaluation using MongoDB search pipelines
    
    Args:
        args: Command-line arguments parsed by argparse. Dictionary with keys:
            - pipeline: Path to the MongoDB aggregation pipeline JSON file
            - uri: MongoDB connection URI
            - eval_database: Database name for ideal rankings
            - search_database: Database name for search documents
            - search_collection: Collection name for search documents
            - search_index: Search index name
            - query_filter: Filter to select specific queries from ideal rankings
            - k: Number of top results to evaluate for NDCG@K
            - debug: If True, print detailed debug information
            - scoring: NDCG rank scoring algorithm to use ('binary','inverse_rank','decay','score')
            - print: Whether to print results to console or return them

    Returns:
        NDCG evaluation results dictionary if 'print' is not True, otherwise prints to console.

    """
    # Default arguments
    default_args = {
        'pipeline': None,  # This should be provided
        'k': 10,
        'scoring': 'binary',
        'eval_database': 'search_evaluation',
        'search_database': 'search_evaluation',
        'search_collection': 'documents',
        'search_index': 'text_search_index',
        'query_filter': {'type': 'text'},
        'uri': 'mongodb://admin:admin@localhost:27017/?directConnection=true&authSource=admin',
        'debug': False,
        'print': False
    }
    
    # Handle different input types
    if args is None:
        # No args provided, use defaults
        final_args = default_args.copy()
    elif hasattr(args, '__dict__'):
        # argparse Namespace object - convert to dict and merge
        args_dict = vars(args)
        final_args = default_args.copy()
        final_args.update(args_dict)
    elif isinstance(args, dict):
        # Dictionary provided - merge with defaults
        final_args = default_args.copy()
        final_args.update(args)
    else:
        raise ValueError("args must be None, argparse.Namespace, or dict")
    
    # Convert to namespace-like object for dot notation access
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(**final_args)
    
    # Validate required arguments
    if args.pipeline is None:
        raise ValueError("pipeline argument is required")

    try:
        from pymongo import MongoClient
        MONGO_AVAILABLE = True
    except ImportError:
        MONGO_AVAILABLE = False
        print("Error: pymongo is required. Install with: pip install pymongo")
        sys.exit(1)
        
    try:
        if not MONGO_AVAILABLE:
            print("Error: pymongo is required for MongoDB functionality.")
            print("Install it with: pip install pymongo")
            sys.exit(1)
        
        # Turn off debugging if output is not print
        if args.print is False:
            args.debug = False

        # Connect to MongoDB
        if args.print: print(f"Connecting to MongoDB: {args.uri}")
        client = MongoClient(args.uri)
        db = client[args.eval_database]  # Database for ideal rankings
        search_db = client[args.search_database]  # Database for search documents
        search_collection = search_db[args.search_collection]  # Collection for search documents
        search_index = args.search_index  # Index name

        # Test connection
        db.command('ping')
        if args.print: print("Connected to MongoDB successfully")
        
        # Load pipeline
        if args.print: print(f"Loading pipeline from: {args.pipeline}")
        pipeline = load_pipeline(args.pipeline)
        if args.print: print(f"Loaded pipeline with {len(pipeline)} stages")
        
        # Run evaluation

        # Get queries and ideal rankings
        queries_data = get_queries_from_ideal_rankings(db,args.query_filter)
        if args.print: print(f"Retrieved {len(queries_data)} queries from ideal_rankings collection")

        # Execute search pipeline for each query
        search_results = {}
        ideal_rankings = {}
        
        for query_id, data in queries_data.items():
            query = data['query']
            ideal_ranking = data['ideal_ranking']
            
            try:
                ranked_docs = execute_search_pipeline(search_collection, search_index, pipeline, query)
                search_results[query_id] = ranked_docs
                # Keep the full ideal ranking for relevance determination
                # The k parameter will be used in the NDCG calculation itself
                ideal_rankings[query_id] = ideal_ranking
            except Exception as e:
                print(f"Warning: Failed to execute pipeline for query {query_id}: {e}")

        # Evaluate NDCG using graded relevance (since we have ideal rankings)
        results = batch_evaluate_ndcg(search_results, ideal_rankings, args.k, args.debug, args.scoring)
        
        if args.print:
            # Output results
            print("\nNDCG Evaluation Results")
            print("=" * 50)
            if args.debug:
                if results['total_queries'] > 0:
                    print("Individual Query Scores:")
                    for query_id, score in results['individual_scores'].items():
                        print(f"  {query_id}: {score:.4f} {'🟢' if score >= 0.8 else '🟡' if score >= 0.6 else '🟠' if score >= 0.4 else '🔴'}")
                print()
            
            # Group scores to get number of excellent, good, fair, poor queries
            excellent_count = sum(1 for score in results['individual_scores'].values() if score >= 0.8)
            good_count = sum(1 for score in results['individual_scores'].values() if 0.6 <= score < 0.8)
            fair_count = sum(1 for score in results['individual_scores'].values() if 0.4 <= score < 0.6)
            poor_count = sum(1 for score in results['individual_scores'].values() if score < 0.4)

            print(f"🟢 Excellent:\t{excellent_count/len(results['individual_scores']):.1%} ({excellent_count})")
            print(f"🟡 Good:\t{good_count/len(results['individual_scores']):.1%} ({good_count})")
            print(f"🟠 Fair:\t{fair_count/len(results['individual_scores']):.1%} ({fair_count})")
            print(f"🔴 Poor:\t{poor_count/len(results['individual_scores']):.1%} ({poor_count})")
            print()

            print(f"Average NDCG@{args.k}: {results['average_ndcg']:.4f}")
            print(f"Total queries evaluated: {results['total_queries']}")
            print()

            # Add performance interpretation
            if results['average_ndcg'] >= 0.8:
                performance = "🟢 Excellent"
            elif results['average_ndcg'] >= 0.6:
                performance = "🟡 Good"
            elif results['average_ndcg'] >= 0.4:
                performance = "🟠 Fair"
            else:
                performance = "🔴 Poor"

            print(f"\n📈 Performance Assessment: {performance} ({results['average_ndcg']:.1%})")
            print("=" * 60)
        else:
            return results
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    """Main entry point for the NDCG evaluation runner."""
    parser = argparse.ArgumentParser(
        description='Run NDCG evaluation using MongoDB search pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Basic usage with pipeline file
    python ndcg.py --pipeline ./example/search-text-pipeline.json
    
    # Custom k value and MongoDB URI
    python ndcg.py --pipeline ./example/search-text-pipeline.json --k 5 --uri mongodb://localhost:27017

    # Custom search database/collection/index
    python ndcg.py --pipeline ./example/search-text-pipeline.json --search-database mydb --search-collection mycol --search-index myindex

    # Enable debug output for detailed NDCG calculations
    python ndcg.py --pipeline ./example/search-text-pipeline.json --debug
            """
    )
    
    parser.add_argument('--pipeline', required=True,
                       help='Path to JSON file containing MongoDB aggregation pipeline')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of top results to evaluate (default: 10)')
    parser.add_argument('--scoring', choices=['inverse_rank','binary','decay','score'], default='binary',
                       help='NDCG relevance scoring algorithm to use for ideal rankings (default: binary)')
    parser.add_argument('--eval-database', '-d', default='search_evaluation',
                       help='MongoDB database name for storing ideal rankings (default: search_evaluation)')
    parser.add_argument('--search-database', '--sdb', default='search_evaluation',
                       help='MongoDB database name for search documents (default: search_evaluation)')
    parser.add_argument('--search-collection', '--scol', default='documents',
                       help='MongoDB collection name for search documents (default: documents)')
    parser.add_argument('--search-index', '-i', default='text_search_index',
                       help='MongoDB Atlas Search index name (default: text_search_index)')
    parser.add_argument('--query-filter', '-qf', default={'type': 'text'},
                       type=json.loads,
                       help='MongoDB query filter to retrieve query rankings as JSON string (default: {"type":"text"})')
    parser.add_argument('--uri', 
                       default='mongodb://admin:admin@localhost:27017/?directConnection=true&authSource=admin',
                       help='MongoDB connection string (default: mongodb://admin:admin@localhost:27017/?directConnection=true&authSource=admin)')
    parser.add_argument('--debug', default=False, action='store_true',
                       help='Enable debug output (default: False)')
    parser.add_argument('--print', '-p', default=True, action='store_false',
                       help='Whether to print results to console (default: True)')
    args = parser.parse_args()

    run(args)
