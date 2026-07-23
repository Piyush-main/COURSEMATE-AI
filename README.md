# RAG Book Assistant — Retrieval-Augmented Generation with LangChain + Chroma

A simple RAG (Retrieval-Augmented Generation) system that lets you upload a PDF,
build a local vector database from it, and ask questions about its content.

This version is configured to run **entirely on free tiers**:

| Component        | Tool used                                  | Cost |
|-------------------|---------------------------------------------|------|
| Embeddings        | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, runs **locally**) | Free, no API key |
| LLM (chat model)  | Mistral AI (`mistral-small`)                | Free API key (generous free tier) |
| Vector store      | ChromaDB (local, on-disk)                   | Free |

No OpenAI key is required anywhere in this project.

---

## 1. Project structure

```
GenAI-part2-Rag-implementation-main/
│
├── app.py                    # Streamlit UI — upload a PDF & ask questions
├── main.py                   # CLI chatbot — ask questions from the terminal
├── create_database.py        # One-off script to build chroma_db from a sample PDF
├── requirements.txt          # Python dependencies
├── .env.example               # Template for your API key (copy to .env)
├── .gitignore
│
├── chroma_db/                 # Persisted vector database (auto-created)
│
├── document loaders/          # Example scripts for loading PDFs / text / web pages
│   ├── pdf.py
│   ├── page.py
│   ├── test.py
│   ├── notes.txt
│   ├── GRU.pdf
│   └── deeplearning.pdf
│
├── vector store/
│   └── DB.py                  # Example: build a small Chroma DB and query it
│
└── retrievers/
    ├── mmr.py                 # Similarity vs MMR retrieval demo
    ├── multiquery.py          # Multi-query retriever demo
    └── arixv.py                # ArXiv paper retriever demo
```

---

## 2. Prerequisites

- Python 3.10+ installed
- `pip` available
- Internet connection (to download the embedding model the first time, and to call the Mistral API)

---

## 3. Setup — step by step

### Step 1: Get the code

If you've already downloaded/cloned the repo, `cd` into the project folder:

```bash
cd GenAI-part2-Rag-implementation-main
```

### Step 2: Create and activate a virtual environment

```bash
# Create
python3 -m venv venv

# Activate
# On macOS/Linux:
source venv/bin/activate
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

This installs LangChain, ChromaDB, `sentence-transformers` (for free local
embeddings), and `langchain-mistralai` (for the free Mistral LLM).

> The first time you run the app, `sentence-transformers` will download the
> `all-MiniLM-L6-v2` embedding model (~90 MB) automatically and cache it
> locally. This only happens once.

### Step 4: Get a FREE Mistral API key

1. Go to **https://console.mistral.ai/**
2. Sign up (free) with your email or Google/GitHub account.
3. Once logged in, go to **API Keys** → **Create new key**
   (direct link: https://console.mistral.ai/api-keys/)
4. Copy the generated key — you'll only see it once, so save it somewhere safe.

Mistral's free tier is generously scoped for personal projects and experimentation.

### Step 5: Create your `.env` file

In the project root, copy the example file:

```bash
# macOS/Linux
cp .env.example .env

# Windows
copy .env.example .env
```

Now open `.env` in any text editor and paste your key:

```
MISTRAL_API_KEY=your_actual_mistral_api_key_here
```

Save the file. `.env` is already listed in `.gitignore`, so it will
never be committed to GitHub — keep it that way.

### Step 6: Build the vector database

Option A — use the sample script (uses `document loaders/deeplearning.pdf`):

```bash
python create_database.py
```

This will:
1. Load the sample PDF
2. Split it into chunks
3. Generate embeddings locally with HuggingFace's MiniLM model
4. Store everything in the local `chroma_db/` folder

Option B — use the Streamlit app to upload your **own** PDF (see Step 7 below);
the app builds the database for you when you click "Create Vector Database".

### Step 7: Run the app

**Option A: Streamlit web app (recommended, supports PDF upload)**

```bash
streamlit run app.py
```

This opens a browser window where you can:
- Upload any PDF
- Click "Create Vector Database"
- Ask questions about the document in a text box

**Option B: Command-line chatbot**

```bash
python main.py
```

This uses whatever is already stored in `chroma_db/` (run `create_database.py`
first if you haven't already). Type your questions, and type `0` to exit.

---

## 4. How it works (high level)

1. **Load** — a PDF is loaded and split into overlapping text chunks
   (`RecursiveCharacterTextSplitter`).
2. **Embed** — each chunk is converted into a vector using the free, local
   `sentence-transformers/all-MiniLM-L6-v2` model (no API call, runs on your CPU).
3. **Store** — vectors are saved in a local Chroma vector database (`chroma_db/`).
4. **Retrieve** — when you ask a question, it's embedded the same way, and the
   most relevant chunks are pulled from Chroma using MMR (Maximal Marginal
   Relevance) search.
5. **Generate** — the retrieved chunks + your question are sent to Mistral's
   chat model (`mistral-small`), which answers using only the retrieved context.

---

## 5. Switching to a different free LLM (optional)

The LLM is isolated to one line in `main.py` / `app.py`:

```python
llm = ChatMistralAI(model="mistral-small-2506")
```

Other free-tier-friendly options you could swap in instead (each needs its own
LangChain integration package and API key env variable):

- **Google Gemini** — `langchain-google-genai`, free tier via Google AI Studio
- **Groq** — `langchain-groq`, very fast free inference for Llama/Mixtral models
- **Cohere** — `langchain-cohere`, free trial key

If you switch providers, remember to:
1. `pip install` the relevant `langchain-<provider>` package
2. Add the new API key variable to `.env` (e.g. `GROQ_API_KEY=...`)
3. Replace the `llm = ...` line with the new provider's chat class

---

## 6. Troubleshooting

- **`ModuleNotFoundError`** → make sure your virtual environment is activated
  and you ran `pip install -r requirements.txt` inside it.
- **`401 Unauthorized` / auth error from Mistral** → double-check your
  `MISTRAL_API_KEY` in `.env` has no extra spaces or quotes around it.
- **First run is slow** → the embedding model download (~90 MB) only happens
  once; subsequent runs are fast since it's cached locally.
- **`.env` not being picked up** → make sure the file is literally named
  `.env` (not `.env.txt`) and sits in the same folder as `main.py`/`app.py`.

---

## 7. Notes

- `chroma_db/` is your local vector database — delete this folder and re-run
  `create_database.py` (or re-upload in the Streamlit app) if you want to
  start fresh with new documents.
- Everything in `document loaders/`, `vector store/`, and `retrievers/` are
  standalone learning/demo scripts showing different LangChain building blocks
  (PDF loading, web loading, text splitting, MMR vs similarity search,
  multi-query retrieval, ArXiv retrieval) — they aren't required to run the
  main app but are useful references.
