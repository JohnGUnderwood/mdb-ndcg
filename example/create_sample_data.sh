#!/bin/bash
# Create sample data for NDCG evaluation testing using mongorestore

set -e  # Exit on any error

read -p "Paste MongoDB URI connection string or use default (mongodb://admin:admin@localhost:27017/?directConnection=true&authSource=admin): " uri

# Use default URI if none provided
if [ -z "$uri" ]; then
    MONGO_URI="mongodb://admin:admin@localhost:27017/?directConnection=true&authSource=admin"
else
    # Strip DB name if provided in URI
    MONGO_URI=$(echo "$uri" | sed 's|/[^/?]*\?|/?|')
fi

echo "Using MongoDB URI: $MONGO_URI"
echo "Loading sample data using mongorestore..."

DB_NAME="search_evaluation"

# Restore data from dump directory
# Assumes data files are in example/data/search_evaluation/
DATA_DIR="./example/data/search_evaluation"

if [ -d "$DATA_DIR" ]; then
    mongorestore --uri "$MONGO_URI" \
                 --nsInclude="$DB_NAME.*" \
                 "$DATA_DIR"
    echo "Data restored successfully!"
else
    echo "Error: $DATA_DIR directory not found. Please ensure your data files are in the correct location."
    echo "Expected structure:"
    echo "  example/data/search_evaluation/"
    echo "    documents.bson (or .json)" 
    echo "    ideal_rankings.bson (or .json)"
    exit 1
fi

echo "Creating search indexes..."
mongosh "$MONGO_URI" --file "./example/create_indexes.js"

echo "Sample data setup completed successfully!"
echo "Database: search_evaluation"
echo "Collections: documents, ideal_rankings"
echo "Indexes: text_search_index, vector_search_index"