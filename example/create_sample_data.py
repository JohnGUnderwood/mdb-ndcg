#!/usr/bin/env python3
"""
Create sample data for NDCG evaluation testing.
"""
from pymongo import MongoClient

def create_sample_data():
    """Create sample documents and ideal rankings for testing."""
    client = MongoClient('mongodb://admin:admin@localhost:27017/?directConnection=true&authSource=admin')
    db = client['search_evaluation']
    
    # Sample documents
    documents = [
        {"_id": "doc1", "title": "Introduction to Machine Learning", "content": "Machine learning basics and algorithms", "tags": ["ml", "ai", "algorithms"]},
        {"_id": "doc2", "title": "Deep Learning Guide", "content": "Neural networks and deep learning techniques", "tags": ["dl", "neural", "ai"]},
        {"_id": "doc3", "title": "Data Science Fundamentals", "content": "Statistics and data analysis methods", "tags": ["data", "statistics", "analysis"]},
        {"_id": "doc4", "title": "Python Programming", "content": "Python programming language tutorial", "tags": ["python", "programming", "tutorial"]},
        {"_id": "doc5", "title": "Machine Learning with Python", "content": "Using Python for machine learning projects", "tags": ["python", "ml", "projects"]},
        {"_id": "doc6", "title": "Statistics for Data Science", "content": "Statistical methods used in data science", "tags": ["statistics", "data", "methods"]},
        {"_id": "doc7", "title": "Neural Network Architecture", "content": "Different types of neural network architectures", "tags": ["neural", "architecture", "dl"]},
        {"_id": "doc8", "title": "AI Ethics and Bias", "content": "Ethical considerations in artificial intelligence", "tags": ["ai", "ethics", "bias"]},
        {"_id": "doc9", "title": "Data Visualization", "content": "Creating effective data visualizations", "tags": ["data", "visualization", "charts"]},
        {"_id": "doc10", "title": "Algorithm Analysis", "content": "Time and space complexity of algorithms", "tags": ["algorithms", "complexity", "analysis"]}
    ]
    
    # Sample ideal rankings for queries (10 results each to support NDCG@k for k=1-10)
    ideal_rankings = [
        {
            "query_id": "query1",
            "query": "machine learning",
            "collection": "documents",
            "ideal_ranking": ["doc1", "doc5", "doc2", "doc8", "doc7", "doc3", "doc10", "doc4", "doc6", "doc9"]
        },
        {
            "query_id": "query2", 
            "query": "python programming",
            "collection": "documents",
            "ideal_ranking": ["doc4", "doc5", "doc1", "doc2", "doc3", "doc7", "doc8", "doc6", "doc9", "doc10"]
        },
        {
            "query_id": "query3",
            "query": "data science",
            "collection": "documents",
            "ideal_ranking": ["doc3", "doc6", "doc9", "doc1", "doc5", "doc2", "doc8", "doc4", "doc7", "doc10"]
        },
        {
            "query_id": "query4",
            "query": "neural networks",
            "collection": "documents",
            "ideal_ranking": ["doc2", "doc7", "doc8", "doc1", "doc5", "doc3", "doc4", "doc6", "doc9", "doc10"]
        },
        {
            "query_id": "query5",
            "query": "statistics",
            "collection": "documents",
            "ideal_ranking": ["doc6", "doc3", "doc10", "doc1", "doc9", "doc2", "doc4", "doc5", "doc7", "doc8"]
        }
    ]
    
    # Insert data
    print("Inserting sample documents...")
    db.documents.delete_many({})  # Clear existing
    db.documents.insert_many(documents)
    
    print("Inserting ideal rankings...")
    db.ideal_rankings.delete_many({})  # Clear existing
    db.ideal_rankings.insert_many(ideal_rankings)
    
    # Create text index for search
    print("Creating text index...")
    try:
        db.documents.create_search_index(
            {
                "name": "text_search_index",
                "type": "search",
                "definition": {
                    "mappings": {
                    "dynamic": False,
                        "fields": {
                            "title": {
                                "type": "string",
                                "analyzer": "lucene.standard",
                                "searchAnalyzer": "lucene.standard"
                            },
                            "content": {
                                "type": "string",
                                "analyzer": "lucene.standard",
                                "searchAnalyzer": "lucene.standard"
                            },
                            "tags": {
                                "type": "string",
                                "analyzer": "lucene.keyword",
                                "searchAnalyzer": "lucene.keyword"
                            }
                        }
                    }
                }
            }
        )
        print("Text index created successfully")
    except Exception as e:
        print(f"Note: {e}")
    
    print(f"Sample data created successfully!")
    print(f"- {len(documents)} documents inserted")
    print(f"- {len(ideal_rankings)} ideal rankings inserted")
    print("Database: search_evaluation")
    print("Collections: documents, ideal_rankings")

if __name__ == "__main__":
    create_sample_data()