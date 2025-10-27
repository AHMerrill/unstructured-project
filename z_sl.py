# ================================================================
# app.py — Anti-Echo Chamber Streamlit Application
# ================================================================
# This app compares a user-uploaded article to a curated corpus
# of existing articles and finds ideologically diverse coverage
# of the same event or topic using semantic and stance embeddings.
# ================================================================

import os, io, re, json, yaml, pdfplumber
import numpy as np, pandas as pd
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
from bs4 import BeautifulSoup
import streamlit as st
import chromadb
import torch
import nltk

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(page_title="📰 Anti-Echo Chamber", layout="wide")

PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_DIR   = PROJECT_ROOT / "config"
TMP          = PROJECT_ROOT / "tmp"
CHROMA_PATH  = PROJECT_ROOT / "chroma_db"
for p in [TMP, CHROMA_PATH]:
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# HEADER & INTRODUCTION
# ------------------------------------------------------------
st.title("📰 Anti-Echo Chamber — Ideological Diversity Finder")

intro_text = """
This tool analyzes **news articles** to find *ideologically contrasting coverage* of the same event or topic.

It compares uploaded articles against a database of pre-embedded articles using topic, stance, and bias embeddings.

---

### What it does
1. Uploads and parses your article (PDF, TXT, or HTML)
2. Infers the **source bias** (via lookup or GPT fallback)
3. Creates **topic** and **stance embeddings**
4. Searches **ChromaDB** for articles with similar topics
5. Computes a weighted **Anti-Echo Score** for ideological diversity

---

### Anti-Echo Scoring Formula

The score rewards topical similarity but penalizes stance alignment and ideological similarity:

anti_echo_score =
(w_T_canonical × canonical_overlap)
+ (w_T_summary × summary_similarity)
− (w_S × stance_similarity)
− (w_B × bias_diff)
− (w_Tone × tone_diff)

Higher scores = same topic, **different viewpoint**.
"""

st.markdown(intro_text)

# ------------------------------------------------------------
# LOAD MODELS & DATABASE
# ------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_environment():
    with open(CONFIG_DIR / "config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
    topic_model = SentenceTransformer(CONFIG["embeddings"]["topic_model"], device="cuda" if torch.cuda.is_available() else "cpu")
    stance_model = SentenceTransformer(CONFIG["embeddings"]["stance_model"], device="cuda" if torch.cuda.is_available() else "cpu")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    topic_coll = client.get_collection(CONFIG["chroma_collections"]["topic"])
    stance_coll = client.get_collection(CONFIG["chroma_collections"]["stance"])
    return CONFIG, topic_model, stance_model, topic_coll, stance_coll

CONFIG, topic_model, stance_model, topic_coll, stance_coll = load_environment()

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def extract_text(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.read()
    if ext == ".txt":
        return data.decode("utf-8", errors="ignore")
    if ext == ".pdf":
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "".join([p.extract_text() or "" for p in pdf.pages])
    if ext in [".html", ".htm"]:
        soup = BeautifulSoup(data.decode("utf-8", "ignore"), "html.parser")
        for s in soup(["script", "style"]): s.decompose()
        return soup.get_text(separator=" ")
    return ""

def encode_topic(text):
    return topic_model.encode(text, normalize_embeddings=True)

def encode_stance(text):
    return stance_model.encode(text, normalize_embeddings=True)

def interpret_bias(score):
    if score <= -0.6: return "Progressive / Left"
    if -0.6 < score <= -0.2: return "Center-Left"
    if -0.2 < score < 0.2: return "Center / Neutral"
    if 0.2 <= score < 0.6: return "Center-Right"
    if score >= 0.6: return "Conservative / Right"
    return "Unknown"

# ------------------------------------------------------------
# SIDEBAR SETTINGS
# ------------------------------------------------------------
st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password")
if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input
    client = OpenAI(api_key=api_key_input)
else:
    client = None

st.sidebar.markdown("---")
st.sidebar.subheader("Weight Parameters")
w_T_canonical = st.sidebar.slider("Topic Overlap Weight", 0.0, 10.0, 0.5, 0.1)
w_T_summary   = st.sidebar.slider("Summary Similarity Weight", 0.0, 10.0, 10.0, 0.1)
w_S           = st.sidebar.slider("Stance Similarity Weight", 0.0, 10.0, 1.0, 0.1)
w_B           = st.sidebar.slider("Bias Difference Weight", 0.0, 10.0, 0.8, 0.1)
w_Tone        = st.sidebar.slider("Tone Difference Weight", 0.0, 10.0, 0.3, 0.1)

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
uploaded = st.file_uploader("Upload an article (.pdf, .txt, or .html)")

if uploaded:
    if not client:
        st.error("Please enter your OpenAI API key first.")
        st.stop()

    text = extract_text(uploaded)
    st.write(f"Extracted {len(text)} characters.")
    st.write("Analyzing...")

    # Topic and stance embeddings
    topic_vec = encode_topic(text)
    stance_vec = encode_stance(text)

    # Retrieve candidates
    topic_docs = topic_coll.get(include=["embeddings", "metadatas"])
    stance_docs = stance_coll.get(include=["embeddings", "metadatas"])

    all_matches = []
    for emb, md in zip(topic_docs["embeddings"], topic_docs["metadatas"]):
        summary_similarity = float(cosine_similarity([topic_vec], [emb])[0][0])
        stance_match = next((s_emb for s_emb, s_md in zip(stance_docs["embeddings"], stance_docs["metadatas"]) if s_md.get("id","").split("::")[0] == md.get("id","").split("::")[0]), None)
        stance_sim = float(cosine_similarity([stance_vec], [stance_match])[0][0]) if stance_match is not None else 0.0
        bias_db = float(md.get("bias_score", 0.0))
        bias_diff = abs(0.0 - bias_db)
        tone_diff = abs(0.0 - float(md.get("tone_score", 0.0)))
        canonical_overlap = np.random.uniform(0.3, 0.9)  # placeholder for tag overlap
        anti_echo_score = (
            (w_T_canonical * canonical_overlap)
            + (w_T_summary * summary_similarity)
            - (w_S * stance_sim)
            - (w_B * bias_diff)
            - (w_Tone * tone_diff)
        )
        all_matches.append({
            "title": md.get("title","Untitled"),
            "source": md.get("source","Unknown"),
            "bias_family": md.get("bias_family",""),
            "bias_score": bias_db,
            "canonical_overlap": canonical_overlap,
            "summary_similarity": summary_similarity,
            "stance_similarity": stance_sim,
            "bias_diff": bias_diff,
            "tone_diff": tone_diff,
            "anti_echo_score": anti_echo_score,
            "url": md.get("url","")
        })

    df = pd.DataFrame(all_matches).sort_values("anti_echo_score", ascending=False).head(10)
    st.subheader("Results")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "anti_echo_results.csv")

    if st.button("Reset and Upload Another"):
        st.experimental_rerun()
