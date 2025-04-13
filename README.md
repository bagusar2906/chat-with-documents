# 🤖 Chat with Google Drive Documents

Interact with your Google Drive documents (PDF, DOCX) using GPT-4!  
This app uses LangChain, Supabase, and OpenAI to let you semantically search and chat with your files — all inside a Streamlit web interface.

![demo](docs/demo.gif)

---

## 🚀 Features

- 🧠 GPT-4-based Q&A over your uploaded documents
- 📁 Google Drive integration for file listing and upload
- 🔎 Supabase as a vector store for document embeddings
- 🧱 LangChain for RAG pipeline
- 🖥️ Streamlit-powered UI for local or cloud deployment

---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/chat-with-documents.git
cd chat-with-documents
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# Activate it:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
---

## 🛠️ Setting Up Supabase for Vector Search
To enable semantic search capabilities in your application, you'll need to set up a table in Supabase to store document embeddings.

1. Enable the vector Extension
Before creating the table, ensure the vector extension is enabled in your Supabase database. This extension allows you to store and query vector embeddings.
Using Supabase SQL Editor:
   1. Navigate to your Supabase Project.
   2. Select your project.
   3. In the left sidebar, click on SQL Editor.
   4. Run the following SQL command:
  
  ```sql
  create extension if not exists vector;
  ```
2. Create the documents Table
After enabling the vector extension, create a table to store your documents along with their embeddings. Using Supabase SQL Editor:
    1. In the SQL Editor, run the following SQL command:
  
  ```sql
  create table documents (
    id uuid primary key default gen_random_uuid(),
    content text,
    embedding vector(1536) -- Adjust the dimension to match your embedding size
  );
  ```
  Note: Ensure that the embedding dimension matches the output size of your embedding model (e.g., 1536 for OpenAI's text-embedding-ada-002).

3. Create the match_documents Function
This SQL function performs a similarity search on the documents table using the embedding vector. It returns the top matches with an optional filter on metadata.
    1. In the SQL Editor, run the following SQL command:

```sql
create or replace function match_documents(
    query_embedding vector,
    match_count int,
    filter jsonb default '{}'
)
returns table (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
)
language sql
as $$
  select
    id,
    content,
    metadata,
    1 - (embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by embedding <=> query_embedding
  limit match_count;
$$;
```
✅ Tip: This function uses the <=> operator to compute cosine similarity, and supports filtering results based on metadata fields.

---

## 🔐 Environment Variables
Create a .env file in the root directory and add the following:
```ini
OPENAI_API_KEY=your-openai-api-key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-service-role-key
SUPABASE_TABLE=documents
GOOGLE_DRIVE_FOLDER_ID=your-shared-folder-id
```
---

## 🧪 Usage
Run the app:
```bash
streamlit run AIAgent.py
```

Then go to <http://localhost:8501> in your browser.

---
## 📁 Project Structure
```bash
chat-with-documents/
├── AIAgent.py           # Main Streamlit app
├── requirements.txt     # Project dependencies
├── .env                 # API credentials (not committed)
├── utils/
│   └── gdrive.py        # Google Drive API utility
├── docs/
│   └── demo.gif         # Screenshots / video demos
└── README.md            # You're reading it!
```
---
## 📦 Tech Stack
[OpenAI GPT-4](https://platform.openai.com/)
[LangChain](https://github.com/langchain-ai/langchain)
[Supabase](https://supabase.com/)
[Streamlit](https://streamlit.io/)
[Google Drive API]

