# Full Streamlit App with YouTube & Document Chat Agent

import os
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from supabase import create_client, Client
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from urllib.parse import urlparse, parse_qs
import numpy as np
import datetime

# --- Supabase Config ---
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
        doc = DocxDocument(file)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

def process_text(text):
    splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
    return splitter.split_text(text)

def generate_summary(text):
    prompt = f"Summarize the following document in 3-4 concise sentences:\n\n{text[:3000]}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def store_embeddings(chunks, doc_name, extra_metadata=None):
    embeddings = OpenAIEmbeddings()
    summary = generate_summary(" ".join(chunks))
    base_metadata = {
        "source": doc_name,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "summary": summary,
    }
    if extra_metadata:
        base_metadata.update(extra_metadata)
    metadata = [base_metadata] * len(chunks)
    SupabaseVectorStore.from_texts(chunks, embeddings, client=supabase, table_name="documents", metadata=metadata)

def chat_with_doc(db, query, filter=None):
    results = db.similarity_search_by_vector_with_relevance_scores(
        embedding=db.embedding.embed_query(query),
        k=3,
        filter=filter or {}
    )
    summary_texts = [doc.metadata.get("summary", "") for doc, _ in results]
    summaries = "\n".join(set(summary_texts))
    context = "\n".join([doc.page_content for doc, score in results])
    prompt = f"""Answer the question based on the following document summaries and context:

Summaries:
{summaries}

Context:
{context}

Question: {query}"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def get_documents():
    res = supabase.table("documents").select("metadata").execute()
    files = {}
    if res.data:
        for row in res.data:
            meta = row.get("metadata", {})
            if isinstance(meta, dict):
                source = meta.get("source", "Unknown")
                if source not in files:
                    files[source] = {
                        "count": 0,
                        "timestamp": meta.get("timestamp", ""),
                        "summary": meta.get("summary", ""),
                        "title": meta.get("title", ""),
                        "author": meta.get("author", ""),
                        "video_url": meta.get("video_url", ""),
                        "thumbnail_url": meta.get("thumbnail_url", "")
                    }
                files[source]["count"] += 1
    return files

def delete_document(source):
    supabase.table("documents").delete().match({"metadata->>source": source}).execute()

def extract_youtube_transcript(url):
    try:
        video_id = extract_video_id(url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript])
    except Exception as e:
        st.error(f"❌ Failed to extract transcript: {e}")
        return ""

def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        return parse_qs(parsed_url.query)['v'][0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip('/')
    raise ValueError("Invalid YouTube URL")

def get_youtube_metadata(url):
    try:
        yt = YouTube(url)
        return {
            "title": yt.title,
            "author": yt.author,
            "thumbnail_url": yt.thumbnail_url,
            "video_url": yt.watch_url,
            "video_id": yt.video_id
        }
    except Exception as e:
        st.error(f"⚠️ Could not fetch YouTube metadata: {e}")
        return {}

# --- Streamlit UI ---
st.set_page_config(page_title="AI Chat with Files & YouTube", layout="wide")
st.title("📄🤖 Chat with Your Files or YouTube Videos")
mode = st.radio("Select mode:", ["📤 Upload Document", "🔍 Query Existing Data", "▶️ Embed YouTube Video"])
embeddings = OpenAIEmbeddings()
db = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name="documents")

# Upload PDF/DOCX
if mode == "📤 Upload Document":
    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])
    if uploaded_file:
        text = extract_text(uploaded_file)
        st.success("✅ Document uploaded and text extracted.")
        with st.expander("🔍 Preview Document Text"):
            st.text_area("Document Content", value=text[:3000], height=300)
        with st.spinner("Processing and embedding..."):
            chunks = process_text(text)
            store_embeddings(chunks, uploaded_file.name)
            st.success("📌 Text embedded and stored!")

# Embed YouTube
elif mode == "▶️ Embed YouTube Video":
    yt_url = st.text_input("Paste a YouTube video URL:")
    if yt_url:
        with st.spinner("Fetching transcript and metadata..."):
            transcript = extract_youtube_transcript(yt_url)
            meta = get_youtube_metadata(yt_url)
        if transcript and meta:
            st.markdown(f"### 🎬 [{meta['title']}]({meta['video_url']}) by *{meta['author']}*")
            st.image(meta["thumbnail_url"], width=300)
            with st.expander("📜 Transcript Preview"):
                st.text_area("Transcript Content", value=transcript[:3000], height=300)
            if st.button("Embed YouTube Transcript"):
                with st.spinner("Generating summary and embedding..."):
                    chunks = process_text(transcript)
                    store_embeddings(chunks, f"YouTube: {meta['video_id']}", {
                        "title": meta["title"],
                        "author": meta["author"],
                        "video_url": meta["video_url"],
                        "thumbnail_url": meta["thumbnail_url"]
                    })
                    st.success("🎉 Transcript embedded successfully!")

# Query Interface
elif mode == "🔍 Query Existing Data":
    docs = get_documents()
    sources = list(docs.keys())
    selected_source = st.selectbox("📁 Filter by document (optional):", ["All"] + sources)
    metadata_filter = {} if selected_source == "All" else {"source": selected_source}
    query = st.text_input("Ask something:")
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if query:
        with st.spinner("Thinking..."):
            answer = chat_with_doc(db, query, filter=metadata_filter)
            st.session_state.query_history.append((query, answer))
            st.markdown(f"**Answer:** {answer}")
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("### 🕘 Query History")
        for q, a in reversed(st.session_state.query_history):
            with st.expander(f"Q: {q}"):
                st.markdown(f"**A:** {a}")

# Manage Documents
st.markdown("---")
st.markdown("## 📚 Uploaded Documents")
with st.expander("📁 Manage Documents"):
    docs = get_documents()
    if not docs:
        st.info("No documents found.")
    else:
        for doc_name, info in docs.items():
            col1, col2, col3 = st.columns([4, 2, 1])
            col1.markdown(f"**{info.get('title') or doc_name}**")
            if info.get("video_url"):
                col1.markdown(f"[▶️ Watch Video]({info['video_url']})")
            col2.markdown(f"{info['count']} chunks")
            if col3.button("❌ Delete", key=f"del_{doc_name}"):
                delete_document(doc_name)
                st.success(f"Deleted '{doc_name}'")
                st.experimental_rerun()
            if info.get("thumbnail_url"):
                st.image(info["thumbnail_url"], width=320)
            if info.get("summary"):
                with st.expander("📝 Summary"):
                    st.markdown(info["summary"])
