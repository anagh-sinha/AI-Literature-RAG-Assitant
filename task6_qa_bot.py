import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import WatsonxLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from ibm_watsonx_ai.credentials import Credentials
from ibm_watsonx_ai.foundation_models import Model

# Configure watsonx.ai
credentials = Credentials(
    url="https://eu-de.ml.cloud.ibm.com",
    api_key=""  # REPLACE WITH YOUR API KEY
)

model_id = "mistralai/mixtral-8x7b-instruct-v01"
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 500,
    "temperature": 0.7
}

# Create LLM wrapper
llm = WatsonxLLM(
    model_id="mistralai/mixtral-8x7b-instruct-v01",
    apikey="",
    url="https://eu-de.ml.cloud.ibm.com",
    project_id="",
    params={
        "decoding_method": "greedy",
        "max_new_tokens": 500,
        "temperature": 0.7
    }
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Process PDF and answer questions
def process_pdf_and_query(pdf_file, query):
    if pdf_file is not None:
        # Load PDF
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
        sources = "\n\nRelevant Sources:\n"
        for i, doc in enumerate(result['source_documents'][:3]):
            sources += f"\n{i+1}. {doc.page_content[:200]}...\n"
        
        return answer + sources
    else:
        return "Please upload a PDF file first."

# Create Gradio interface
iface = gr.Interface(
    fn=process_pdf_and_query,
    inputs=[
        gr.File(label="Upload PDF", file_types=[".pdf"]),
        gr.Textbox(
            label="Enter your question", 
            placeholder="What this paper is talking about?",
            value="What this paper is talking about?"  # Pre-fill the query
        )
    ],
    outputs=gr.Textbox(label="Answer", lines=10),
    title="Quest Analytics RAG Assistant",
    description="Upload a research paper PDF and ask questions about it!",
    examples=[
        [None, "What this paper is talking about?"],
        [None, "What are the main findings?"],
        [None, "What methodology was used?"]
    ]
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)