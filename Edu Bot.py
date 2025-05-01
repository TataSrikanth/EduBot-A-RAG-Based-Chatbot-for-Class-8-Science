import streamlit as st
import mysql.connector
import json
import datetime
from transformers import pipeline, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from PIL import Image

# --------------------------
# CONFIGURATION & DB SETUP
# --------------------------

# Load DB config from JSON (ensure db_config.json exists in your project folder)
with open(r"C:/Users/vicky/Documents/pro4/db_config.json", "r") as f:
    db_config = json.load(f)

def connect_db():
    return mysql.connector.connect(
        host=db_config["host"],
        user=db_config["user"],
        password=db_config["password"],
        database=db_config["database"]
    )

# Function to check login credentials from MySQL
def check_login(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
    result = cursor.fetchone()
    conn.close()
    return result is not None

# Log question in the DB
def log_question(user, question):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chatbot_logs (username, question, timestamp) VALUES (%s, %s, %s)",
                   (user, question, datetime.datetime.now()))
    conn.commit()
    cursor.close()
    conn.close()

# --------------------------
# LLM & EMBEDDING FUNCTIONS
# --------------------------

# Initialize a Q&A model for answering questions
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def generate_answer(question, context):
    if not context.strip():
        return "I couldn't find relevant information in the book."
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Initialize text embedding model for retrieval
text_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_similar_text(query, index, metadata, k=3):
    query_vec = text_embed_model.encode(query).astype("float32")
    D, I = index.search(np.array([query_vec]), k)
    return [metadata[i] for i in I[0]]

# Initialize CLIP models for image retrieval
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_similar_image(query, index, metadata, k=2):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_feat = clip_model.get_text_features(**inputs).cpu().numpy().astype("float32")
    D, I = index.search(text_feat, k)
    return [metadata[i] for i in I[0]]

# Initialize summarization model (to generate chapter summaries on the fly)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --------------------------
# LOAD EMBEDDING INDEXES & METADATA
# --------------------------

# Load text embeddings FAISS index and metadata
text_index = faiss.read_index(r"C:\Users\vicky\Documents\pro4\chapter_embeddings.index")
with open(r"C:/Users/vicky/Documents/pro4/text_embeddings.json", "r", encoding="utf-8") as f:
    text_metadata = json.load(f)

# Load image embeddings FAISS index and metadata
image_index = faiss.read_index(r"C:/Users/vicky/Documents/pro4/image_index.faisss")
with open(r"C:/Users/vicky/Documents/pro4/image_index_metadata.json", "r", encoding="utf-8") as f:
    image_metadata = json.load(f)
# --------------------------
# STREAMLIT UI
# --------------------------

st.title("🧠 Class 8 Science Chatbot")

# --- Login Section ---
if "user" not in st.session_state:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.user = username
        else:
            st.error("❌ Invalid credentials")
    st.stop()

st.success(f"👋 Welcome {st.session_state.user}")

# --- Chat Input ---
user_input = st.text_input("Ask a question about Class 8 Science")

if user_input:
    # Log the question to the database
    log_question(st.session_state.user, user_input)

    # --- Check for commands: chapter list & chapter summary ---
    if "chapter list" in user_input.lower():
        # List unique chapters from the text metadata
        chapters = sorted(list(set([meta['chapter'] for meta in text_metadata if 'chapter' in meta])))
        st.markdown("📚 **Available Chapters:**")
        for ch in chapters:
            st.write(f"- {ch}")

    elif "summary" in user_input.lower():
        # Generate a chapter summary on the fly
        # Try to identify the chapter name from the query
        chapter_name = None
        for meta in text_metadata:
            if "chapter" in meta and meta["chapter"].lower() in user_input.lower():
                chapter_name = meta["chapter"]
                break
        if chapter_name:
            # Combine all text chunks for that chapter
            combined_text = " ".join([meta["text"] for meta in text_metadata if meta.get("chapter") == chapter_name])
            if combined_text:
                summary = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
                st.markdown(f"### Summary of {chapter_name}")
                st.write(summary)
            else:
                st.write("No text available for this chapter.")
        else:
            st.write("Please specify which chapter you want a summary for.")
    
    else:
        # --- Retrieval for Regular Question ---
        # Retrieve similar text chunks
        text_results = get_similar_text(user_input, text_index, text_metadata)
        context = "\n".join([item['text'] for item in text_results if 'text' in item])
        response = generate_answer(user_input, context)
        
        # Check if the retrieved context is empty (i.e., out-of-syllabus)
        if not context.strip():
            st.warning("⚠️ Sorry, this topic seems out of the NCERT syllabus.")
        else:
            st.markdown("💬 **Answer:**")
            st.write(response)
            st.markdown("---")
            st.markdown("🖼️ **Related Images:**")
            image_results = get_similar_image(user_input, image_index, image_metadata)
            for item in image_results:
                if "image_path" in item:
                    st.image(item["image_path"], caption=item["caption"], use_column_width=True)
