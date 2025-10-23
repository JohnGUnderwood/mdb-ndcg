# MongoDB NDCG Evaluation

Evaluate search systems using NDCG (Normalized Discounted Cumulative Gain) with MongoDB aggregation pipelines.

## Key Features

- Execute MongoDB search pipelines with dynamic query injection
- Configurable NDCG@k evaluation (k=1 to k=10)
- Batch processing with ideal rankings stored in MongoDB
- Binary relevance and graded relevance support
- Sample data with vector embeddings to test behaviour

## Files

- `ndcg.py` - NDCG calculation functions and main evaluation runner
- `setup.sh` - Environment setup script
- `example/create_sample_data.sh` - Creates sample test data
- `example/atlas_text_pipeline.json` - Example search pipeline
- `tests/` - Test suite

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies and create sample data  
./setup.sh

# Activate virtual environment (required for each session)
source .venv/bin/activate
```

### 2. Usage with sample data

```bash
# Evaluate using the text example pipeline (default k=5)
python ndcg.py --pipeline ./example/search-text-pipeline.json

# Evaluate using vector queries
python ndcg.py --pipeline ./example/search-voyage3large-pipeline.json --query-filter '{"type":"voyage-3-large"}' --search-index vector_search_index

# Evaluate NDCG@3 (top 3 results only)
python ndcg.py --pipeline ./example/search-text-pipeline.json --k 3

# Evaluate NDCG@10 using inverse rank for relevance scores
python ndcg.py --pipeline ./example/search-text-pipeline.json --k 10 --scoring 'score'
```

## How It Works

1. **Load Pipeline**: JSON files containing MongoDB aggregation pipelines
2. **Query Injection**: Automatically inject query strings using `{{QUERY}}` placeholders
3. **Execute Search**: Run pipelines against your document collection
4. **Compare Results**: Evaluate against ideal rankings stored in MongoDB
5. **Calculate NDCG**: Calculate NCDG scores for each query saved in MongoDB

### Rankings Format
```json
{
  "query_id":"query1",
  "query":"machine learning",
  "type":"text",
  "ideal_ranking":[
    "doc1",
    "doc3",
    "doc5",
    "doc6",
  ]
}
```
Or with explicit scores and vector embedding:
```json
{
  "query_id":"query1",
  "query":"BinaryBinary.fromFloat32Array(new Float32Array([...])"
  "type":"text",
  "ideal_ranking":[
    {"doc_id":"doc1","score":5}
    {"doc_id":"doc3","score":4.5},
    {"doc_id":"doc5","score":3},
    {"doc_id":"doc6","score":1}  
  ]
}
```
Rankings can have other metadata which can be used with a filter when running the eval using `--query-filter` parameter. You must pass a valid json string with single quotes and double quotes around keys: `'{"type":"voyage-3-large"}'`.

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
python ndcg.py [OPTIONS]

options:
  -h, --help            show this help message and exit
  --pipeline PIPELINE   Path to JSON file containing MongoDB aggregation pipeline
  --k K                 Number of top results to evaluate (default: 10)
  --scoring {inverse_rank,binary,decay,score}
                        NDCG relevance scoring algorithm to use for ideal rankings (default: binary)
  --eval-database, -d EVAL_DATABASE
                        MongoDB database name for storing ideal rankings (default: search_evaluation)
  --search-database, --sdb SEARCH_DATABASE
                        MongoDB database name for search documents (default: search_evaluation)
  --search-collection, --scol SEARCH_COLLECTION
                        MongoDB collection name for search documents (default: documents)
  --search-index, -i SEARCH_INDEX
                        MongoDB Atlas Search index name (default: text_search_index)
  --query-filter, -qf QUERY_FILTER
                        MongoDB query filter to retrieve query rankings as JSON string (default: {"type":"text"})
  --uri URI             MongoDB connection string (default: mongodb://admin:admin@localhost:27017/?directConnection=true&authSource=admin)
  --debug               Enable debug output (default: False)
  --print, -p           Whether to print results to console (default: True)

    Examples:
    # Basic usage with pipeline file
    python ndcg.py --pipeline ./example/search-text-pipeline.json
    
    # Custom k value and MongoDB URI
    python ndcg.py --pipeline ./example/search-text-pipeline.json --k 5 --uri mongodb://localhost:27017

    # Custom search database/collection/index
    python ndcg.py --pipeline ./example/search-text-pipeline.json --search-database mydb --search-collection mycol --search-index myindex

    # Enable debug output for detailed NDCG calculations
    python ndcg.py --pipeline ./example/search-text-pipeline.json --debug
```

## Python API

### Executing and returning for programmatic acces
```python
from ndcg import run
args = {
  "k":5,
  "pipeline":"./example/search-text-pipeline.json",
  "scoring":"inverse_rank",
  "search_index":"text_search_index",
  "query_filter":{"type":"text"}
}
run(args)
```

### Basic NDCG Evaluation

```python
from ndcg import compute_ndcg,batch_evaluate_ndcg

# Binary evaluation using compute_ndcg with compute_scores
search_results = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
relevant_docs = ['doc1', 'doc3', 'doc5']  # List of relevant documents (binary relevance)
ndcg_score = compute_ndcg(relevant_docs, search_results, k=5, method='binary')

# Batch evaluation (automatically handles scoring internally)
search_batch = {'query1': ['doc1', 'doc2'], 'query2': ['doc3', 'doc4']}
ground_truth = {'query1': ['doc1'], 'query2': ['doc3']}
results = batch_evaluate_ndcg(search_batch, ground_truth, k=10, scoring='binary')
```

### Relevance Scoring with compute_scores Function

The `compute_scores` function generates relevance scores for documents based on their position in ideal rankings. It supports four different scoring methods:

```python
from ndcg import compute_scores

# Method 1: Binary relevance (implicit) - all documents in ranking are relevant
relevant_docs = ['doc3', 'doc1', 'doc5']
binary_scores = compute_scores(relevant_docs, method='binary')
# Returns: {'doc3': 1, 'doc1': 1, 'doc5': 1}

# Method 2: Inverse rank graded relevance (implicit) - position-based scoring
ideal_ranking = ['doc3', 'doc1', 'doc5', 'doc2', 'doc4']  # Most relevant first
inverse_scores = compute_scores(ideal_ranking, method='inverse_rank')
# Returns: {'doc3': 1.0, 'doc1': 0.5, 'doc5': 0.333, 'doc2': 0.25, 'doc4': 0.2}

# Method 3: Exponential decay graded relevance (implicit) - adaptive scoring
decay_scores = compute_scores(ideal_ranking, method='decay')
# Returns exponentially decaying scores based on ranking length

# Method 4: Explicit scores (explicit) - externally provided relevance scores
scored_docs = [
    {'doc_id': 'doc1', 'score': 5},
    {'doc_id': 'doc2', 'score': 3},
    {'doc_id': 'doc3', 'score': 4}
]
explicit_scores = compute_scores(scored_docs, method='score')
# Returns: {'doc1': 5, 'doc2': 3, 'doc3': 4}
```

### Advanced NDCG Scoring Algorithms

The NDCG evaluation supports four different scoring algorithms, each designed for different evaluation scenarios:

```python
from ndcg import batch_evaluate_ndcg

search_results = {'query1': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']}
ideal_ranking = {'query1': ['doc3', 'doc1', 'doc2', 'doc4', 'doc5']}  # Most relevant first

# Explicit scoring
ndcg_score = batch_evaluate_ndcg(search_results, ideal_ranking, k=5, scoring='score')

# Binary relevance scoring (for relevant document sets)
relevant_docs = {'query1': {'doc1', 'doc3', 'doc5'}}
ndcg_binary = batch_evaluate_ndcg(search_results, relevant_docs, k=5, scoring='binary')

# Inverse rank graded relevance (position-based scoring)
ndcg_inverse = batch_evaluate_ndcg(search_results, ideal_ranking, k=5, scoring='inverse_rank')

# Exponential decay graded relevance (adaptive scoring)
ndcg_decay = batch_evaluate_ndcg(search_results, ideal_ranking, k=5, scoring='decay')
```

#### Algorithm Comparison

**Binary Relevance** (`scoring='binary'`):
- Uses simple 0/1 relevance scores
- Position 1-k: 1 if relevant, 0 if not relevant
- Best for: Simple relevant/not-relevant judgments, A/B testing
- Input: Set or list of relevant document IDs

**Inverse Rank Graded Relevance** (`scoring='inverse_rank'`):
- Uses position-based relevance: 1/1, 1/2, 1/3, 1/4, ...
- Position 1: relevance = 1.0, Position 2: relevance = 0.5, Position 3: relevance = 0.33
- Best for: When you have clear ranking preferences but want moderate scoring differences
- Input: Ideal ranking list (most relevant first)

**Exponential Decay Graded Relevance** (`scoring='decay'`):
- Uses adaptive exponential scoring that scales with ranking length
- Base relevance = `scaling_factor * log2(ideal_ranking_length) + 1`
- Short ideal ranking (3 docs): Position 1 ≈ 2^2.6 ≈ 6.0
- Medium ideal ranking (8 docs): Position 1 ≈ 2^4.0 = 16.0  
- Long ideal ranking (16 docs): Position 1 ≈ 2^5.0 = 32.0
- Best for: When you have queries with varying numbers of relevant documents and want to emphasize top positions more strongly for complex queries
- Input: Ideal ranking list (most relevant first)

**Explicit Score Graded Relevance** (`scoring='score'`):
- Uses externally provided relevance scores (e.g., 0-5 scale, expert judgments)
- Documents are scored individually with explicit numeric relevance values
- Best for: When you have expert-curated relevance scores, user ratings, or other numeric quality measures
- Input: List of dictionaries with 'doc_id' and 'score' keys: `[{'doc_id': 'doc1', 'score': 5}, {'doc_id': 'doc2', 'score': 3}]`

```python
# Example: Choose algorithm based on your evaluation needs

# Binary: Simple A/B testing with relevant/not-relevant judgments
search_results = {'query1': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']}
relevant_docs = {'query1': {'doc1', 'doc3', 'doc5'}}  # Just mark as relevant
ndcg_binary = batch_evaluate_ndcg(search_results, relevant_docs, k=5, scoring='binary')

# Inverse Rank: You have rankings but want moderate score differences  
ideal_rankings = {'query1': ['doc3', 'doc1', 'doc5', 'doc2', 'doc4']}  # Ranked by preference
ndcg_inverse = batch_evaluate_ndcg(search_results, ideal_rankings, k=5, scoring='inverse_rank')

# Exponential Decay: Mixed query complexity with adaptive top-position emphasis
short_query = {'query_simple': ['doc1', 'doc2', 'doc3']}
short_ideal = {'query_simple': ['doc3', 'doc1', 'doc2']}  # 3 relevant docs

complex_query = {'query_complex': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']}  
complex_ideal = {'query_complex': ['doc3', 'doc1', 'doc2'] + [f'doc{i}' for i in range(6, 16)]}  # 15 relevant docs

# Decay algorithm adapts: simple query position 1 ≈ 6, complex query position 1 ≈ 32
ndcg_decay_simple = batch_evaluate_ndcg(short_query, short_ideal, k=5, scoring='decay')
ndcg_decay_complex = batch_evaluate_ndcg(complex_query, complex_ideal, k=5, scoring='decay')

# Explicit Score: Using expert-provided relevance scores
from ndcg import compute_ndcg
scored_results = {'query1': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']}
expert_scores = {'query1': [{'doc_id': 'doc1', 'score': 5}, {'doc_id': 'doc3', 'score': 4}, {'doc_id': 'doc5', 'score': 2}]}
# Note: For 'score' algorithm, use compute_scores and compute_ndcg directly
ndcg_expert = compute_ndcg(expert_scores['query1'], scored_results['query1'], k=5, scoring='score')
```

## Requirements

- Python 3.7+
- pymongo (for MongoDB functionality)
- MongoDB server (local or remote)

## License

Open source - feel free to use and modify for your search evaluation needs.