import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from supabase import create_client, Client
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import yt_dlp
from urllib.parse import urlparse, parse_qs
from openai import OpenAI
import numpy as np
import datetime
from datetime import datetime

# --- Custom Vector Store Class ---

class SupabaseUUIDVectorStore(SupabaseVectorStore):
    def __init__(self, client, embedding, table_name):
        self.client = client
        self.embedding = embedding
        self._embedding = embedding 
        self.table_name = table_name

    @classmethod
    def from_texts(cls, texts, embedding, client, table_name, metadata=None):
        vectors = embedding.embed_documents(texts)
        rows = []

        for i, text in enumerate(texts):
            row = {
                "content": text,
                "embedding": vectors[i],
                "metadata": metadata[i] if metadata else {},
            }
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
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Helper Functions ---


import streamlit as st


def delete_document(source):
    try:
        response = supabase.table("documents") \
            .delete() \
            .filter("metadata->>source", "eq", source) \
            .execute()

        if response.error:
            st.error(f"Failed to delete document: {response.error.message}")
        else:
            st.success("Document deleted successfully.")
            st.json(response.data)  # Optional: Show deleted rows

    except Exception as e:
        st.exception(f"Error occurred while deleting document: {e}")



def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages)
    elif file.name.endswith(".docx"):
        from docx import Document as DocxDocument
        doc = DocxDocument(file)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return ""

def process_text(text):
    splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
    return splitter.split_text(text)


def get_youtube_metadata(url):
    try:
        ydl_opts = {
            'quiet': True,  # suppress CLI output
            'skip_download': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get('title'),
                "author": info.get('uploader'),
                "thumbnail_url": info.get('thumbnail'),
                "video_url": info.get('webpage_url'),
                "video_id": info.get('id')
            }
    except Exception as e:
        st.error(f"⚠️ Could not fetch YouTube metadata: {e}")
        return {}
    
def get_youtube_transcript(url, preferred_languages=['id']):
    # Extract video ID from URL
    def extract_video_id(youtube_url):
        parsed = urlparse(youtube_url)
        if parsed.hostname in ['www.youtube.com', 'youtube.com']:
            return parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.hostname == 'youtu.be':
            return parsed.path[1:]
        return None

    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    # Get transcript
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=preferred_languages)
        return " ".join([t['text'] for t in transcript])
    except NoTranscriptFound:
        st.error("⚠️ Transcript not available in the preferred languages.")
    except Exception as e:
        st.error(f"❌ Failed to extract transcript: {e}")
        return None

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
    res = supabase.table("documents").select("id, metadata").execute()
    files = []
    seen = set()  # For deduplication

    if res.data:
        for row in res.data:
            doc_id = row.get("id")
            meta_list = row.get("metadata")
            
            if isinstance(meta_list, list):
                for meta in meta_list:
                    if isinstance(meta, dict):
                        item = {
                            "document_id": doc_id,
                            "source": meta.get("source", "Unknown"),
                            "title": meta.get("title", ""),
                            "summary": meta.get("summary", ""),
                            "timestamp": meta.get("timestamp", ""),
                            "author": meta.get("author", ""),
                            "video_url": meta.get("video_url", ""),
                            "thumbnail_url": meta.get("thumbnail_url", "")
                        }

                        # Create a unique key for deduplication
                        unique_key = (
                            item["document_id"],
                            item["title"],
                            item["timestamp"]
                        )

                        if unique_key not in seen:
                            seen.add(unique_key)
                            files.append(item)

    # Optional: sort by timestamp if it's a parseable date
    def parse_time(item):
        try:
            return datetime.fromisoformat(item["timestamp"])
        except Exception:
            return datetime.min

    files.sort(key=parse_time, reverse=True)

    return files

# --- Streamlit UI ---

st.set_page_config(page_title="AI Chat with Docs & YouTube", layout="wide")
st.title("📄🤖 Chat with Your Docs or YouTube Videos")
mode = st.radio("Select mode:", ["📤 Upload Document", "🔍 Query Existing Data", "▶️ Embed YouTube Video"])

embeddings = OpenAIEmbeddings()
db = SupabaseUUIDVectorStore(client=supabase, embedding=embeddings, table_name="documents")

# 📤 Upload Mode
if mode == "📤 Upload Document":
    st.markdown("---")
    st.markdown("## 📚 Uploaded Documents")
    
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
          # transcript = extract_youtube_transcript(yt_url)
            transcript = get_youtube_transcript(yt_url)
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

# 🔍 Query Mode
elif mode == "🔍 Query Existing Data":
    #sources = list(get_documents().keys())
    sources = get_documents()
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
        for info in docs:
            col1, col2, col3 = st.columns([4, 2, 1])
            col1.markdown(f"**{info['title']}**")
            if info.get("video_url"):
                col1.markdown(f"[▶️ Watch Video]({info['video_url']})")
           # col2.markdown(f"{info['count']} chunks")
            if col3.button("❌ Delete", key=f"del_{info['title']}"):
                delete_document(info["source"])
                st.success(f"Deleted '{info['title']}'")
                st.rerun()
            if info.get("thumbnail_url"):
                st.image(info["thumbnail_url"], width=320)
            if info.get("summary"):
                #with st.expander("📝 Summary"):
                st.markdown(info["summary"])