# MongoDB NDCG Evaluation

Evaluate search systems using NDCG (Normalized Discounted Cumulative Gain) with MongoDB aggregation pipelines.

## Key Features

- Execute MongoDB search pipelines with dynamic query injection
- Configurable NDCG@k evaluation (k=1 to k=10)
- Batch processing with ideal rankings stored in MongoDB
- Binary relevance and graded relevance support

## Files

- `ndcg.py` - Core NDCG calculation functions
- `run_ndcg_evaluation.py` - Main evaluation runner
- `setup.sh` - Environment setup script
- `example/create_sample_data.py` - Creates sample test data
- `example/atlas_search_pipeline.json` - Example search pipeline
- `tests/` - Test suite

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies and create sample data  
./setup.sh

# Activate virtual environment (required for each session)
source .venv/bin/activate
```

### 2. Basic Usage

```bash
# Evaluate using the example pipeline (default k=5)
python run_ndcg_evaluation.py --pipeline ./example/atlas_search_pipeline.json

# Evaluate NDCG@3 (top 3 results only)
python run_ndcg_evaluation.py --pipeline ./example/atlas_search_pipeline.json --k 3

# Use MongoDB aggregation for NDCG calculation
python run_ndcg_evaluation.py --pipeline ./example/atlas_search_pipeline.json --aggregation
```

## How It Works

1. **Load Pipeline**: JSON files containing MongoDB aggregation pipelines
2. **Query Injection**: Automatically inject query strings using `{{QUERY}}` placeholders
3. **Execute Search**: Run pipelines against your document collection
4. **Compare Results**: Evaluate against ideal rankings stored in MongoDB
5. **Calculate NDCG**: Use Python or MongoDB aggregation for final scores

### Pipeline Format
```json
[
  {
    "$search": {
      "index":"{{INDEX_NAME}}",
      "text":{
        "path":"content",
        "query":"{{QUERY}}"
      }
    }
  },
  {
    "$limit": 100
  }
]
```

## Command Line Options

```bash
python run_ndcg_evaluation.py [OPTIONS]

options:
  -h, --help            show this help message and exit
  --pipeline PIPELINE   Path to JSON file containing MongoDB aggregation pipeline
  --k K                 Number of top results to evaluate (default: 10)
  --algorithm {inverse_rank,binary,decay}
                        NDCG relevance scoring algorithm to use for ideal rankings (default: binary)
  --eval-database, -d EVAL_DATABASE
                        MongoDB database name for storing ideal rankings (default: search_evaluation)
  --search-database, --sdb SEARCH_DATABASE
                        MongoDB database name for search documents (default: search_evaluation)
  --search-collection, --scol SEARCH_COLLECTION
                        MongoDB collection name for search documents (default: documents)
  --search-index, -i SEARCH_INDEX
                        MongoDB Atlas Search index name (default: text_search_index)
  --uri URI             MongoDB connection string (default: mongodb://admin:admin@localhost:27017/?directConnection=true&authSource=admin)
  --debug               Enable debug output (default: False)
```

## Python API

### Basic NDCG Evaluation

```python
from ndcg import compute_ndcg_binary, batch_evaluate_ndcg

# Binary evaluation
search_results = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
relevant_docs = {'doc1', 'doc3', 'doc5'}
ndcg_score = compute_ndcg_binary(search_results, relevant_docs, k=5)

# Batch evaluation
search_batch = {'query1': ['doc1', 'doc2'], 'query2': ['doc3', 'doc4']}
ground_truth = {'query1': ['doc1'], 'query2': ['doc3']}
results = batch_evaluate_ndcg(search_batch, ground_truth, k=10)
```

### Advanced NDCG Scoring Algorithms

The NDCG evaluation supports three different scoring algorithms, each designed for different evaluation scenarios:

```python
from ndcg import batch_evaluate_ndcg

search_results = {'query1': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']}
ideal_ranking = {'query1': ['doc3', 'doc1', 'doc2', 'doc4', 'doc5']}  # Most relevant first

# Binary relevance scoring (for relevant document sets)
relevant_docs = {'query1': {'doc1', 'doc3', 'doc5'}}
ndcg_binary = batch_evaluate_ndcg(search_results, relevant_docs, k=5, algorithm='binary')

# Inverse rank graded relevance (position-based scoring)
ndcg_inverse = batch_evaluate_ndcg(search_results, ideal_ranking, k=5, algorithm='inverse_rank')

# Exponential decay graded relevance (adaptive scoring)
ndcg_decay = batch_evaluate_ndcg(search_results, ideal_ranking, k=5, algorithm='decay')
```

#### Algorithm Comparison

**Binary Relevance** (`algorithm='binary'`):
- Uses simple 0/1 relevance scores
- Position 1-k: 1 if relevant, 0 if not relevant
- Best for: Simple relevant/not-relevant judgments, A/B testing
- Input: Set or list of relevant document IDs

**Inverse Rank Graded Relevance** (`algorithm='inverse_rank'`):
- Uses position-based relevance: 1/1, 1/2, 1/3, 1/4, ...
- Position 1: relevance = 1.0, Position 2: relevance = 0.5, Position 3: relevance = 0.33
- Best for: When you have clear ranking preferences but want moderate scoring differences
- Input: Ideal ranking list (most relevant first)

**Exponential Decay Graded Relevance** (`algorithm='decay'`):
- Uses adaptive exponential scoring that scales with ranking length
- Base relevance = `scaling_factor * log2(ideal_ranking_length) + 1`
- Short ideal ranking (3 docs): Position 1 ≈ 2^2.6 ≈ 6.0
- Medium ideal ranking (8 docs): Position 1 ≈ 2^4.0 = 16.0  
- Long ideal ranking (16 docs): Position 1 ≈ 2^5.0 = 32.0
- Best for: When you have queries with varying numbers of relevant documents and want to emphasize top positions more strongly for complex queries
- Input: Ideal ranking list (most relevant first)

```python
# Example: Choose algorithm based on your evaluation needs

# Binary: Simple A/B testing with relevant/not-relevant judgments
search_results = {'query1': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']}
relevant_docs = {'query1': {'doc1', 'doc3', 'doc5'}}  # Just mark as relevant
ndcg_binary = batch_evaluate_ndcg(search_results, relevant_docs, k=5, algorithm='binary')

# Inverse Rank: You have rankings but want moderate score differences  
ideal_rankings = {'query1': ['doc3', 'doc1', 'doc5', 'doc2', 'doc4']}  # Ranked by preference
ndcg_inverse = batch_evaluate_ndcg(search_results, ideal_rankings, k=5, algorithm='inverse_rank')

# Exponential Decay: Mixed query complexity with adaptive top-position emphasis
short_query = {'query_simple': ['doc1', 'doc2', 'doc3']}
short_ideal = {'query_simple': ['doc3', 'doc1', 'doc2']}  # 3 relevant docs

complex_query = {'query_complex': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']}  
complex_ideal = {'query_complex': ['doc3', 'doc1', 'doc2'] + [f'doc{i}' for i in range(6, 16)]}  # 15 relevant docs

# Decay algorithm adapts: simple query position 1 ≈ 6, complex query position 1 ≈ 32
ndcg_decay_simple = batch_evaluate_ndcg(short_query, short_ideal, k=5, algorithm='decay')
ndcg_decay_complex = batch_evaluate_ndcg(complex_query, complex_ideal, k=5, algorithm='decay')
```

## Requirements

- Python 3.7+
- pymongo (for MongoDB functionality)
- MongoDB server (local or remote)

## License

Open source - feel free to use and modify for your search evaluation needs.