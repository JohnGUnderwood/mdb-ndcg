// Create search indexes for the search_evaluation database
use('search_evaluation');

// Create text search index
try {
    db.documents.createSearchIndex(
        'text_search_index',
        'search',
        {
            mappings: {
                dynamic: false,
                fields: {
                    page: {
                        type: 'string',
                        analyzer: 'lucene.standard',
                        searchAnalyzer: 'lucene.standard'
                    },
                    text: {
                        type: 'string',
                        analyzer: 'lucene.standard',
                        searchAnalyzer: 'lucene.standard'
                    },
                    hierarchy: {
                        type: 'string',
                        analyzer: 'lucene.keyword',
                        searchAnalyzer: 'lucene.keyword'
                    }
                }
            }
        }
    );
    print('Text search index created successfully');
} catch(e) {
    print('Note: ' + e);
}

// Create vector search index
try {
    db.documents.createSearchIndex(
        'vector_search_index',
        'vectorSearch',
        {
            fields: [
                {
                    type: 'vector',
                    path: 'embedding.voyage-3-large',
                    numDimensions: 1024,
                    similarity: 'dotProduct'
                },
                {
                    type: 'vector',
                    path: 'embedding.voyage-context-3',
                    numDimensions: 1024,
                    similarity: 'dotProduct'
                }
            ]
        }
    );
    print('Vector search index created successfully');
} catch(e) {
    print('Note: ' + e);
}