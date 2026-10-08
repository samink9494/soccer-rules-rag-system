# Soccer Rules RAG System: Can AI Referee a Match?

This project is a **Retrieval-Augmented Generation (RAG)** system built in Python that answers questions about **soccer rules** using a custom document collection.

The system retrieves relevant sections from official-style soccer rule documents and uses a Large Language Model (LLM) to generate accurate, grounded answers.

---

## Project Overview

- Topic: Soccer Rules
- Type: Retrieval-Augmented Generation (RAG)
- Language: Python
- Focus: Document retrieval, chunking, and answer generation
- Course Project: Programming for Business Analytics
- Institution: California State University, Long Beach

---
## How It Works

1. Soccer rule documents are loaded from text files.
2. Documents are split into overlapping chunks.
3. Chunks are embedded and stored in a vector database.
4. User questions are embedded and matched with relevant chunks.
5. An LLM generates answers **only using retrieved context**.

---

## Dataset

The knowledge base consists of **10 text documents** covering different aspects of soccer rules, including:

- Match duration and structure  
- Offside rule  
- Fouls and misconduct  
- Free kicks and penalties  
- Goalkeeper rules  
- Cards and substitutions  

All documents are written in clear, human-readable language and stored locally.

## Technologies Used

- Python
- Jupyter Notebook
- Vector embeddings
- Local LLM (via Ollama)
- Retrieval-Augmented Generation (RAG)

---

## Setup and How to Run

### Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download) for local answer generation

The 10 soccer rule text files are included in the repository root. The notebook loads them from there; no `my_docs/` folder or file moves are needed.

### Install Python dependencies

From the repository root, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

### Set up Ollama

Install Ollama, then download the model used by `rag_helpers.py`:

```bash
ollama pull mistral
```

Keep Ollama running locally at `http://127.0.0.1:11434`. If it is not already running, start it in a separate terminal with `ollama serve`. The notebook's LLM client uses the `mistral` model by default.

### Verify setup

Run the lightweight local checks from the repository root:

```bash
python verify_setup.py
```

This checks the document files and Python dependencies without connecting to Ollama. To also test the local Ollama endpoint and confirm that `mistral` is available, run:

```bash
python verify_setup.py --ollama
```

The Ollama check requires Ollama to be running locally and the `mistral` model to be pulled.

### Run the notebook

Open `student_rag_project.ipynb` in Jupyter and run the cells in order. The first run of the embedding model may download model files from Hugging Face.

Ask questions such as:
   - *What is the offside rule?*
   - *When is a penalty kick awarded?*
   - *How many substitutions are allowed?*

---

## Example Questions

- What happens if a goalkeeper handles the ball outside the penalty area?
- Does a soccer match always end after 90 minutes?
- In what situations is a red card given?

---

## Learning Outcomes

- Implemented a full RAG pipeline
- Practiced text chunking and retrieval
- Evaluated system performance with different configurations
- Gained hands-on experience with AI-powered question answering

---

## Evaluation Summary

The system was evaluated using a mix of **factual, inferential, vague, and out-of-scope questions**.

**Key observations:**
- Factual questions were answered accurately and consistently  
- Inferential questions showed strong performance when relevant context was retrieved  
- Vague or humorous questions highlighted the limitations of retrieval-based systems  
- Chunk size and overlap significantly affected retrieval quality  

Overall, the RAG approach produced more reliable answers than a standalone language model.

---
