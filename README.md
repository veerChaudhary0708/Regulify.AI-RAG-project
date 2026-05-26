# Regulify.AI — RAG Assistant

A Streamlit-powered Retrieval-Augmented Generation (RAG) application designed to parse, reason over, and synthesize complex mathematical documents. Built to handle the structural and notational demands of content like linear algebra texts, theorem proofs, and matrix-heavy PDFs.

## Features

- **Intelligent Chunking** — Uses `RecursiveCharacterTextSplitter` with a wide chunk window and high overlap, preventing multi-line equations, matrices, and theorems from being split across context boundaries.
- **Math-Aware Prompting** — The LLM is instructed to act as a dedicated mathematical assistant, preserving notation and formatting output cleanly with Markdown and LaTeX.
- **Conceptual Synthesis** — Goes beyond raw retrieval; the system prompt requires the model to explain the *purpose* and *implications* of the math it surfaces, not just quote it back.
- **In-Memory Vector Store** — FAISS provides fast, lightweight semantic retrieval with no external database dependencies.
- **Google Gemini Integration** — Powered by `gemini-3.5-flash` for high-accuracy reasoning on algebra, calculus, and linear transformations.(other OpenAI and opensource models from huggingface can also be used)
- **Clean Chat UI** — Streamlit delivers a conversational interface with dynamic API key input directly in the sidebar.

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| RAG Orchestration | LangChain |
| Vector Store | FAISS |
| LLM & Embeddings | Google Gemini API |

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/veerChaudhary0708/Regulify.AI-RAG-project.git
cd Regulify.AI-RAG-project
```

**2. Set up a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

## Usage

1. Grab a free API key from [Google AI Studio](https://aistudio.google.com).(or use this key for testing purpose only "AIzaSyBxqKdwLkquHQ6_SUO0Ni6VQVRV99ICzuM")
2. Paste it into the secure sidebar panel when the app opens and hit Enter.
3. Upload a mathematical/normal PDF (e.g., `lintransf.pdf`).
4. Start querying — for example:
   > *"According to Theorem 4.22, how can we represent a linear transformation of rank r using an m × n matrix A?"*
