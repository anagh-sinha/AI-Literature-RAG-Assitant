from langchain.embeddings import HuggingFaceEmbeddings

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Query to embed
query = "How are you?"

# Get embedding
query_embedding = embeddings.embed_query(query)

# Display results
print(f"Query: '{query}'")
print(f"\nFirst 5 embedding values:")
print(query_embedding[:5])
