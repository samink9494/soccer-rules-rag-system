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

## How to Run

1. Make sure Python and required dependencies are installed.
2. Place soccer rule documents inside the `my_docs/` folder.
3. Open `student_rag_project.ipynb`.
4. Run the notebook cells in order.
5. Ask questions such as:
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
