from langchain.document_loaders import PyPDFLoader, WebBaseLoader

# # Option 1: Load from URL (replace with actual URL)
# pdf_url = ""  # REPLACE WITH ACTUAL URL
# loader = WebBaseLoader(pdf_url)

# Option 2: Load from local file
loader = PyPDFLoader(r"C:\Users\anagh\Documents\Code\AI RAG Assistant Using LangChain\A-Comprehensive-Review-of-Low-Rank-Adaptation-in-Large-Language-Models-for-Efficient-Parameter-Tuning-1.pdf")
    
# Load the document
documents = loader.load()

# Display first 1000 characters
print("First 1000 characters of content:")
print("=" * 80)
print(documents[0].page_content[:1000])
print("=" * 80)
