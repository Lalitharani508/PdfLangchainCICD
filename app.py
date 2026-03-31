import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("PDF Chunk Viewer with LangChain")

# Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with open("./sample.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("PDF uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader("./sample.pdf")
    documents = loader.load()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    st.write(f"Total chunks: {len(chunks)}")

    # Show first chunk
    st.subheader("First Chunk")
    st.write(chunks[0].page_content)
    st.write(chunks[0].metadata)