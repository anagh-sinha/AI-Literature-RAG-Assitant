Setup and Execution Instructions for RAG Project
Prerequisites and Setup
1. Install Required Packages
bashpip install langchain langchain-community chromadb pypdf gradio sentence-transformers ibm-watsonx-ai
2. Set Up IBM watsonx.ai Credentials

Sign up for IBM watsonx.ai if you haven't already
Get your API key from the IBM Cloud console
Get your project ID from watsonx.ai
Update the credentials in the code:

Replace YOUR_API_KEY with your actual API key
Replace YOUR_PROJECT_ID with your actual project ID



Running Each Task and Capturing Screenshots
Task 1: PDF Loader (pdf_loader.png)
python# Run only the Task 1 section of the code
# Make sure to replace the PDF URL with the actual URL from the assignment
# The code will display the first 1000 characters
# Take a screenshot showing both the code and output
Steps:

Update the pdf_url variable with the actual paper URL
Run the Task 1 section
Screenshot should include:

The loader code
The print statement showing first 1000 characters



Task 2: Code Splitter (code_splitter.png)
python# Run only the Task 2 section
# The LaTeX text is already provided in the code
# Screenshot the splitting code and all chunks
Steps:

Run the Task 2 section
Screenshot should include:

The LaTeX splitter initialization code
The splitting code
All resulting chunks displayed



Task 3: Embedding (embedding.png)
python# Run only the Task 3 section
# The query "How are you?" is already in the code
# Screenshot the embedding code and first 5 values
Steps:

Run the Task 3 section
Screenshot should include:

The embedding initialization code
The embed_query code
The output showing first 5 embedding values



Task 4: Vector Database (vectordb.png)
python# Run the Task 4 section
# This creates new-Policies.txt and performs similarity search
# Query: "Smoking policy"
Steps:

Run the Task 4 section
Screenshot should include:

The Chroma database creation code
The similarity search code
All 5 search results



Task 5: Retriever (retriever.png)
python# Run the Task 5 section
# Uses the vector database from Task 4
# Query: "Email policy"
Steps:

Ensure Task 4 has been run first (to create the vector database)
Run the Task 5 section
Screenshot should include:

The retriever creation code
The retrieval code
The top 2 results



Task 6: QA Bot (QA_bot.png)
python# Uncomment the iface.launch() line at the end
# Run the entire script
# This will launch the Gradio interface
Steps:

Uncomment iface.launch(share=True) at the bottom of the code
Run the entire script
The Gradio interface will open in your browser
Upload the PDF from the provided link
Enter the query: "What this paper is talking about?"
Click Submit
Screenshot should include:

The Gradio interface
The uploaded PDF file name
The query in the text box
The answer from the bot



Important Notes

File Paths: Make sure all file paths are correct and the files exist
API Keys: Never share screenshots with visible API keys
Dependencies: Ensure all packages are installed correctly
Gradio: For Task 6, the interface will open in your default browser
Screenshots: Save each screenshot with the exact filename specified:

pdf_loader.png
code_splitter.png
embedding.png
vectordb.png
retriever.png
QA_bot.png



Troubleshooting
Common Issues:

Import Errors: Install missing packages with pip
API Errors: Check your watsonx credentials
File Not Found: Ensure new-Policies.txt is created (Task 4)
Gradio Not Opening: Check if port 7860 is available

For PDF Loading Issues:

If the web loader fails, download the PDF locally and use PyPDFLoader
Example: loader = PyPDFLoader("local_paper.pdf")

Project Submission Checklist

 Task 1: pdf_loader.png showing code and first 1000 characters
 Task 2: code_splitter.png showing LaTeX splitting code and results
 Task 3: embedding.png showing embedding code and first 5 values
 Task 4: vectordb.png showing Chroma creation and similarity search
 Task 5: retriever.png showing retriever code and top 2 results
 Task 6: QA_bot.png showing Gradio interface with PDF and query

Final Tips

Run each task separately first to ensure it works
Clear and readable screenshots are crucial for grading
Include all required elements in each screenshot
Test the complete flow before final submission
Make sure the watsonx LLM is properly configured for Task 6
