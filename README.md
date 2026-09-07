# CareGuide — AI Hospital Assistant
## Deployement Link - https://medical-rag-chatbot-pna0.onrender.com
CareGuide is a Flask-based Retrieval-Augmented Generation (RAG) chatbot that answers medical, medicine, and hospital-related questions using a curated knowledge base, and can analyze uploaded images (e.g. medicine packaging, forms) using a vision-capable LLM. It maintains per-session conversational memory so it can resolve follow-up questions and recall information the user has shared earlier in the chat.

This project was built as a Summer Training Project for academic submission.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Environment Variables](#environment-variables)
- [Running the App](#running-the-app)
- [API Endpoints](#api-endpoints)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Known Limitations](#known-limitations)
- [References](#references)

---

## Features

- **Medical Q&A over a custom knowledge base** — retrieves and answers from hospital info, medicine data, WHO guideline PDFs, and the MedQuAD medical QnA dataset.
- **Image analysis** — upload a photo (e.g. a medicine strip or hospital form) and ask questions about it using a vision LLM.
- **Multi-turn conversational memory** — remembers what the user said earlier in the session (name, age, preferences) and can recall it on request.
- **Pronoun / coreference resolution** — follow-up questions like *"what drugs are used for its treatment?"* are automatically resolved against the current topic being discussed.
- **Personal-info gate** — statements like *"my name is X and I am 20 years old"* are acknowledged conversationally without accidentally triggering irrelevant medical retrieval.
- **Spelling-correction retry and memory fallback** — if the first retrieval attempt finds nothing, the query is spell-corrected and retried before falling back to conversation memory.
- **Session-based chat history** with search/filter via a REST endpoint.
- **Modern chat UI** — voice input (Web Speech API), dark/light theme, ECG-style loading animation, drag-and-drop image upload, per-message copy/text-to-speech actions.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | Flask |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) |
| Vector store | FAISS (flat index) |
| Conversation memory | LangChain `ConversationBufferMemory` |
| Text LLM | Groq — `openai/gpt-oss-20b` |
| Vision LLM | Groq — `qwen/qwen3.6-27b` |
| Data sources | Hospital info CSV, medicine details CSV, hospital manual PDF, WHO guideline PDFs, MedQuAD dataset |
| Frontend | HTML, CSS, vanilla JS (Web Speech API for voice input) |

---

## Architecture Overview

CareGuide follows a layered architecture:

- **Client (browser)** — chat UI, voice input, drag-and-drop image upload.
- **Flask backend (`app.py`)** — orchestrates the query pipeline, session/memory store, retrieval layer, and the image-analysis endpoint.
- **Retrieval layer** — SentenceTransformer embedder + FAISS index + document store, built offline from the CSV/PDF/MedQuAD sources.
- **External services** — Groq-hosted text and vision LLMs.

A user request flows through: intent gating (greeting / personal info / memory recall) → query rewriting (coreference resolution) → embedding + FAISS retrieval → LLM answer generation → spelling-retry / memory-fallback if nothing is found → response returned with sources and a medical disclaimer. Image requests go through a separate path straight to the vision LLM, with the result stored so follow-up text questions can reference it.

*(See the project report for the full set of use case, data flow, system architecture, and activity diagrams.)*

---

## Project Structure

```
medical-chatbot-flask/
├── app.py                  # Flask app: routes, RAG pipeline, session/memory logic
├── data/
│   ├── faiss_index.bin     # Pre-built FAISS vector index
│   └── doc_store.pkl       # Pickled document store aligned with the FAISS index
├── templates/
│   ├── index.html          # Main chat UI
│   └── history.html        # Chat history viewer
├── .env                     # Environment variables (not committed)
└── requirements.txt
```

---

## Setup and Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd medical-chatbot-flask
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install flask python-dotenv faiss-cpu sentence-transformers groq langchain-groq langchain-classic
   ```
   *(Or `pip install -r requirements.txt` if a requirements file is maintained.)*

4. **Add the knowledge base files**
   Place `faiss_index.bin` and `doc_store.pkl` inside a `data/` folder at the project root. These are built offline from the CSVs, PDFs, and MedQuAD dataset (see the project report for the ingestion process).

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
FLASK_SECRET_KEY=some_random_secret_string
```

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Authenticates requests to Groq's text and vision LLMs |
| `FLASK_SECRET_KEY` | Signs the Flask session cookie (defaults to a dev value if unset — set a real value in production) |

---

## Running the App

```bash
python app.py
```

The app starts a Flask development server at `http://127.0.0.1:5000`.

> ⚠️ This is a development server (`debug=True`). Do not use it as-is in production — run behind a proper WSGI server (e.g. Gunicorn) instead.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the main chat UI |
| GET | `/history.html` | Serves the chat history page |
| POST | `/ask` | Submit a text query; returns `{ answer, sources }` |
| POST | `/analyze-image` | Submit an image (+ optional question); returns `{ answer }` |
| POST | `/reset` | Clears the current session's memory, topic state, and history |
| GET | `/api/prefs` | Returns UI preference flags (e.g. disclaimer toggle) |
| GET | `/api/history` | Returns session chat history, filterable by `kind` and `q` (search) |
| DELETE | `/api/history` | Clears session chat history |
| GET | `/health` | Health check; returns indexed document count |

---

## How the Pipeline Works

1. **Intent gates** — the query is checked against greeting, personal-info, and memory-recall patterns before anything else runs. A match short-circuits straight to a canned/acknowledgment/memory-based reply with no retrieval.
2. **Query rewriting** — if the query contains pronouns or comparative language, it's rewritten into a standalone query using the session's tracked `current_topic`, so follow-ups like *"what drugs are used for its treatment?"* resolve correctly.
3. **Preprocessing** — advice-style phrasing is normalized and acronyms are expanded before searching.
4. **Retrieval** — the (rewritten) query is embedded and matched against the FAISS index to pull the top-k relevant chunks.
5. **Generation** — the retrieved context, conversation history, and (if relevant) previously analyzed image context are passed to the text LLM to produce an answer.
6. **Fallback chain** — if no relevant answer is found, the query is spell-corrected and retried; if it still fails, the system falls back to answering from conversation memory alone.
7. **Response** — the answer is returned with source citations and a medical disclaimer, and the turn is saved to both the LangChain memory buffer and the session's chat history.

---

## Known Limitations

- Session state (`sessions = {}`, `ConversationBufferMemory`) is stored **in-process memory only** — restarting the Flask app clears all active sessions.
- Not tested for concurrent multi-worker deployment (in-memory session dict won't be shared across worker processes).
- A message combining personal info and a real medical question in one sentence without a `?` may be swallowed entirely by the personal-info gate rather than being answered — flagged as a known edge case.
- No authentication/user accounts — sessions are anonymous and cookie-based.
- This is a decision-support/informational tool, not a diagnostic system; every response includes a disclaimer directing users to consult a medical professional.

---

## References

1. Lewis, P. et al.; Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks; Advances in Neural Information Processing Systems (NeurIPS)., Vancouver, Canada, 2020, vol.33, pp.9459-9474.
2. Reimers, N. and Gurevych, I.; Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks; Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)., Hong Kong, China, 2019, pp.3982-3982.
3. Ben Abacha, A. and Demner-Fushman, D.; A Question-Entailment Approach to Question Answering; BMC Bioinformatics., ISSN: 1471-2105; 2019, vol.20, no.1, pp.511:1-511:23.
4. Johnson, J., Douze, M. and Jégou, H.; Billion-Scale Similarity Search with GPUs; IEEE Transactions on Big Data., ISSN: 2332-7790; 2019, vol.7, no.3, pp.535-547.
5. Xiao, S., Liu, Z., Zhang, P. and Muennighoff, N.; C-Pack: Packaged Resources to Advance General Chinese Embedding; arXiv:2309.07597, 2023. Available: https://arxiv.org/abs/2309.07597
6. Groq API Documentation. Available: https://console.groq.com/docs
7. LangChain Documentation. Available: https://docs.langchain.com/
8. Flask Documentation. Available: https://flask.palletsprojects.com/
9. World Health Organization, Health Topics. Available: https://www.who.int/health-topics

*(See the full project report for the complete alphabetized reference list.)*
