
# 🧠 Class 8 Science Chatbot (RAG-Based Educational Assistant)

This project is a **Retrieval-Augmented Generation (RAG)**-based educational chatbot designed to help Class 8 students ask science-related questions. It retrieves relevant textbook content and images, and answers questions using pre-trained NLP models.

---

## 🚀 Features

- 🔐 **User Login System** (MySQL-based authentication)
- 🤖 **Question Answering** using `roberta-base-squad2` LLM
- 📚 **Text Retrieval** using SentenceTransformer + FAISS
- 🖼️ **Image Retrieval** using CLIP (text-to-image search)
- 📝 **Chapter Summaries** using `facebook/bart-large-cnn`
- 📖 **Chapter Listing** from textbook metadata
- 🧵 Built with **Streamlit** for an interactive UI

---

## 🧠 Technologies Used

| Component       | Tech/Library                         |
|------------------|--------------------------------------|
| Interface        | Streamlit                           |
| NLP Models       | Hugging Face Transformers           |
| Text Embedding   | SentenceTransformer (`all-MiniLM`)  |
| Image Embedding  | CLIP (`openai/clip-vit-base-patch32`) |
| Similarity Search| FAISS                               |
| Database         | MySQL                               |

---

## 📂 Project Structure

```
📁 project/
│
├── app.py                        # Main Streamlit app
├── db_config.json               # MySQL credentials
├── chapter_embeddings.index     # FAISS text vector index
├── text_embeddings.json         # Metadata for text chunks
├── image_index.faisss           # FAISS image vector index
├── image_index_metadata.json    # Metadata for image retrieval
```

---

## 🧾 Prerequisites

- Python 3.8+
- MySQL database running with a `users` and `chatbot_logs` table
- Precomputed FAISS indices and metadata files

Install dependencies:

```bash
pip install streamlit mysql-connector-python transformers sentence-transformers faiss-cpu torch Pillow
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Make sure:
- Your model files and indexes are correctly referenced in `app.py`
- MySQL is configured via `db_config.json`

---

## 🧠 How It Works (RAG Pipeline)

1. User logs in and asks a question.
2. System retrieves top-k relevant passages via FAISS.
3. Uses a QA model (`roberta-base-squad2`) to answer based on context.
4. Optionally shows diagrams via CLIP-based image search.
5. Logs all questions in a MySQL database.

---

## 📚 Example Queries

- "What is photosynthesis?"
- "Give me the summary of Chapter 2"
- "List all available chapters"
- "Explain friction with diagrams"


