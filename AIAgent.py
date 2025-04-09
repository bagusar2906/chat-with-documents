import os
import openai
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from langchain.text_splitter import CharacterTextSplitter
from supabase import create_client, Client

# --- Config ---
openai.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Helper Functions ---

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages)
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return ""

def process_text(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def store_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    return SupabaseVectorStore.from_texts(chunks, embeddings, client=supabase, table_name="documents")

def chat_with_doc(db, query):
    results = db.similarity_search(query, k=3)
    context = "\n".join([r.page_content for r in results])
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# --- Streamlit UI ---

st.title("📄 Chat with Your Document")

uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

if uploaded_file:
    text = extract_text(uploaded_file)
    st.success("Document uploaded and text extracted.")

    with st.spinner("Processing and embedding..."):
        chunks = process_text(text)
        db = store_embeddings(chunks)
        st.success("Text embedded and stored!")

    query = st.text_input("Ask something about the document:")
    if query:
        with st.spinner("Thinking..."):
            answer = chat_with_doc(db, query)
            st.markdown(f"**Answer:** {answer}")
