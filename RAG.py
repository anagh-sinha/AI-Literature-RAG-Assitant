# RAG Application for Quest Analytics
# Complete code for all 6 tasks

# First, install required packages:
# pip install langchain langchain-community chromadb pypdf gradio sentence-transformers ibm-watsonx-ai

import os
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import LatexTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import WatsonxLLM
from ibm_watsonx_ai.credentials import Credentials
from ibm_watsonx_ai.foundation_models import Model
import gradio as gr

# Configure watsonx.ai credentials
# Replace with your actual credentials
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key="YOUR_API_KEY"  # Replace with your API key
)

# Configure watsonx LLM
model_id = "mistralai/mixtral-8x7b-instruct-v01"
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 500,
    "temperature": 0.7
}

# Task 1: Load document using LangChain
print("=== Task 1: Loading PDF Document ===")

# Replace with the actual paper URL from the task
pdf_url = "https://example.com/paper.pdf"  # Replace with actual URL
loader = WebBaseLoader(pdf_url)
documents = loader.load()

# Display first 1000 characters
print("First 1000 characters of content:")
print(documents[0].page_content[:1000])

# Alternative if you have the PDF locally:
# loader = PyPDFLoader("path/to/paper.pdf")
# documents = loader.load()

# Task 2: Apply text splitting techniques
print("\n=== Task 2: Text Splitting ===")

latex_text = """
    \\documentclass{article}
    \\begin{document}
    \\maketitle
    \\section{Introduction}
    Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in various natural language processing tasks, including language translation, text generation, and sentiment analysis.
    \\subsection{History of LLMs}
The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.
\\subsection{Applications of LLMs}
LLMs have many applications in the industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.
\\end{document}
"""

# Create LaTeX splitter
latex_splitter = LatexTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

# Split the LaTeX text
latex_chunks = latex_splitter.split_text(latex_text)

print("LaTeX Text Splitting Results:")
for i, chunk in enumerate(latex_chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk)
    print("-" * 50)

# Task 3: Embed documents
print("\n=== Task 3: Embedding ===")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Embed the query
query = "How are you?"
query_embedding = embeddings.embed_query(query)

print(f"Query: {query}")
print(f"First 5 embedding values: {query_embedding[:5]}")

# Task 4: Create vector database
print("\n=== Task 4: Vector Database ===")

# First, create the new-Policies.txt file if it doesn't exist
policies_content = """
Company Policies

Smoking Policy:
Smoking is strictly prohibited in all office buildings and company vehicles. 
Designated smoking areas are available outside the building at least 50 feet from entrances.
Violations will result in disciplinary action.

Email Policy:
All company emails should be professional and courteous.
Personal use of company email should be limited.
Confidential information must not be shared via unsecured email.
All emails are subject to monitoring and review.

Vacation Policy:
Employees accrue 15 days of vacation per year.
Vacation requests must be submitted at least 2 weeks in advance.
Unused vacation days can be carried over to the next year (maximum 5 days).

Remote Work Policy:
Employees may work remotely up to 2 days per week with manager approval.
Remote workers must be available during core business hours (9 AM - 3 PM).
Company equipment must be secured when working remotely.
"""

# Save the content to file
with open("new-Policies.txt", "w") as f:
    f.write(policies_content)

# Load the document
from langchain.document_loaders import TextLoader
loader = TextLoader("new-Policies.txt")
documents = loader.load()

# Split the documents
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Create Chroma vector database
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Conduct similarity search
query = "Smoking policy"
results = vectordb.similarity_search(query, k=5)

print(f"Query: {query}")
print(f"Top 5 results:")
for i, doc in enumerate(results):
    print(f"\nResult {i + 1}:")
    print(doc.page_content)
    print("-" * 50)

# Task 5: Develop a retriever
print("\n=== Task 5: Retriever ===")

# Use ChromaDB as a retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# Conduct similarity search with retriever
query = "Email policy"
retrieved_docs = retriever.get_relevant_documents(query)

print(f"Query: {query}")
print(f"Top 2 retrieved results:")
for i, doc in enumerate(retrieved_docs):
    print(f"\nResult {i + 1}:")
    print(doc.page_content)
    print("-" * 50)

# Task 6: Construct QA Bot
print("\n=== Task 6: QA Bot ===")

# Initialize watsonx LLM
watsonx_model = Model(
    model_id=model_id,
    credentials=credentials,
    params=parameters,
    project_id="YOUR_PROJECT_ID"  # Replace with your project ID
)

# Create LangChain wrapper for watsonx
llm = WatsonxLLM(model=watsonx_model)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Create Gradio interface
def process_pdf_and_query(pdf_file, query):
    if pdf_file is not None:
        # Load the uploaded PDF
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        
        # Split documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever()
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get answer
        result = qa_chain({"query": query})
        
        answer = result['result']
        sources = "\n\nSources:\n"
        for doc in result['source_documents']:
            sources += f"- {doc.page_content[:200]}...\n"
        
        return answer + sources
    else:
        return "Please upload a PDF file first."

# Create Gradio interface
iface = gr.Interface(
    fn=process_pdf_and_query,
    inputs=[
        gr.File(label="Upload PDF", file_types=[".pdf"]),
        gr.Textbox(label="Enter your question", placeholder="What this paper is talking about?")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Quest Analytics RAG Assistant",
    description="Upload a PDF document and ask questions about it!"
)

# Launch the interface
if __name__ == "__main__":
    # For running individual tasks, comment out the line below
    # iface.launch(share=True)
    
    # For Task 6 screenshot, uncomment the line above to launch Gradio
    print("\nTo launch the QA Bot interface for Task 6, uncomment the iface.launch() line")