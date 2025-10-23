#!/usr/bin/env python3
"""
Simplified NDCG Evaluation Runner

This script evaluates NDCG using MongoDB search pipelines and ideal ranking lists.
"""

import argparse
import json
import sys
from typing import Dict, List, Any, Union
from pathlib import Path
from bson import ObjectId
from bson.binary import Binary 
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    print("Error: pymongo is required. Install with: pip install pymongo")
    sys.exit(1)

from ndcg import batch_evaluate_ndcg


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
    
    print(f"Retrieved {len(queries)} queries from ideal_rankings collection")
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

def main():
    """Main entry point for the NDCG evaluation runner."""
    parser = argparse.ArgumentParser(
        description='Run NDCG evaluation using MongoDB search pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with pipeline file
  python run_ndcg_evaluation.py --pipeline ./example/search-text-pipeline.json
  
  # Custom k value and MongoDB URI
  python run_ndcg_evaluation.py --pipeline ./example/search-text-pipeline.json --k 5 --uri mongodb://localhost:27017

  # Custom search database/collection/index
  python run_ndcg_evaluation.py --pipeline ./example/search-text-pipeline.json --search-database mydb --search-collection mycol --search-index myindex

  # Enable debug output for detailed NDCG calculations
  python run_ndcg_evaluation.py --pipeline ./example/search-text-pipeline.json --debug
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
    args = parser.parse_args()
    
    try:
        if not MONGO_AVAILABLE:
            print("Error: pymongo is required for MongoDB functionality.")
            print("Install it with: pip install pymongo")
            sys.exit(1)
        
        # Connect to MongoDB
        print(f"Connecting to MongoDB: {args.uri}")
        client = MongoClient(args.uri)
        db = client[args.eval_database]  # Database for ideal rankings
        search_db = client[args.search_database]  # Database for search documents
        search_collection = search_db[args.search_collection]  # Collection for search documents
        search_index = args.search_index  # Index name

        # Test connection
        db.command('ping')
        print("Connected to MongoDB successfully")
        
        # Load pipeline
        print(f"Loading pipeline from: {args.pipeline}")
        pipeline = load_pipeline(args.pipeline)
        print(f"Loaded pipeline with {len(pipeline)} stages")
        
        # Run evaluation

        # Get queries and ideal rankings
        queries_data = get_queries_from_ideal_rankings(db,args.query_filter)
        
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

        # Group scores to get number of excellent, good, fair, poor queries
        excellent_count = sum(1 for score in results['individual_scores'].values() if score >= 0.8)
        good_count = sum(1 for score in results['individual_scores'].values() if 0.6 <= score < 0.8)
        fair_count = sum(1 for score in results['individual_scores'].values() if 0.4 <= score < 0.6)
        poor_count = sum(1 for score in results['individual_scores'].values() if score < 0.4)
        
        # Output results
        print("\nNDCG Evaluation Results")
        print("=" * 50)
        if args.debug:
            if results['total_queries'] > 0:
                print("Individual Query Scores:")
                for query_id, score in results['individual_scores'].items():
                    print(f"  {query_id}: {score:.4f} {'🟢' if score >= 0.8 else '🟡' if score >= 0.6 else '🟠' if score >= 0.4 else '🔴'}")
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
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()