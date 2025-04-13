import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from supabase import create_client, Client
from openai import OpenAI
import numpy as np


class SupabaseUUIDVectorStore(SupabaseVectorStore):
    def __init__(self, client, embedding, table_name):
        self.client = client
        self.embedding = embedding
        self._embedding = embedding 
        self.table_name = table_name

    @classmethod
    def from_texts(
        cls,
        texts,
        embedding,
        client,
        table_name,
        metadata=None,
    ):
        vectors = embedding.embed_documents(texts)

        rows = []
        for i, text in enumerate(texts):
            row = {
                "content": text,
                "embedding": vectors[i],
            }
            if metadata:
                row["metadata"] = metadata[i]
            rows.append(row)

        insert_resp = client.table(table_name).insert(rows).execute()

        if insert_resp.data is None:
            raise Exception(f"Failed to insert documents")

        return cls(client=client, embedding=embedding, table_name=table_name)

    def similarity_search_by_vector_with_relevance_scores(
        self, embedding, k=4, filter=None, **kwargs
    ):
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        try:
            response = self.client.rpc(
                "match_documents",
                {
                    "query_embedding": embedding,
                    "match_count": k,
                    "filter": filter or {},
                },
            ).execute()
        except Exception as e:
            raise ValueError(f"Error querying Supabase: {e}")

        if not response.data:
            return []

        docs = []
        scores = []
        for match in response.data:
            metadata = match.get("metadata", {})
            text = match.get("content", "")
            score = match.get("similarity", 0)
            docs.append(Document(page_content=text, metadata=metadata))
            scores.append(score)

        return list(zip(docs, scores))

# --- Config ---

# Removed redundant line as openai.api_key is already set


client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")
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
    splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
    return splitter.split_text(text)

def store_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    return SupabaseUUIDVectorStore.from_texts(chunks, embeddings, client=supabase, table_name="documents")


def chat_with_doc(db, query):
    results = db.similarity_search(query, k=3)
    context = "\n".join([r.page_content for r in results])
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    )
    return  response.choices[0].message.content


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
