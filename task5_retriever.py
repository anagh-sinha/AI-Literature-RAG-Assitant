from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing Chroma database (from Task 4)
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# Perform retrieval
query = "Email policy"
retrieved_docs = retriever.get_relevant_documents(query)

# Display results
print(f"Query: '{query}'")
print(f"\nTop 2 retrieved results:")
print("=" * 80)
for i, doc in enumerate(retrieved_docs):
    print(f"\nResult {i + 1}:")
    print(doc.page_content)
    print("-" * 50)
