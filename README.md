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

