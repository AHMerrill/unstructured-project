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
from collections import defaultdict

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

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
    
    # Try to get collections, create if they don't exist
    try:
        topic_coll = client.get_collection(CONFIG["chroma_collections"]["topic"])
    except Exception:
        topic_coll = client.create_collection(CONFIG["chroma_collections"]["topic"], metadata={"hnsw:space": "cosine"})
    
    try:
        stance_coll = client.get_collection(CONFIG["chroma_collections"]["stance"])
    except Exception:
        stance_coll = client.create_collection(CONFIG["chroma_collections"]["stance"], metadata={"hnsw:space": "cosine"})
    
    return CONFIG, topic_model, stance_model, topic_coll, stance_coll

CONFIG, topic_model, stance_model, topic_coll, stance_coll = load_environment()

# Check if database is empty
try:
    topic_count = topic_coll.count()
    stance_count = stance_coll.count()
    if topic_count == 0 or stance_count == 0:
        st.error("⚠️ **ChromaDB is empty** - The database needs to be initialized with article data. Please run the database rebuild script from the notebook or clone the pre-built database.")
        st.stop()
except Exception as e:
    st.error(f"Error checking database: {e}")
    st.stop()

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

def infer_bias_from_text(text, client):
    """Infer bias score from article text using GPT"""
    try:
        prompt = f"""Analyze this article and determine its political bias score.
        Return a number between -1.0 (far left) and 1.0 (far right).
        
        Article excerpt: {text[:1000]}
        
        Return only the numeric bias score (e.g., -0.3, 0.5, etc.):"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.3
        )
        score_str = response.choices[0].message.content.strip()
        # Extract first number
        match = re.search(r'-?\d+\.?\d*', score_str)
        if match:
            score = float(match.group())
            return max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
    except:
        pass
    return 0.0  # Neutral fallback

def topic_overlap_simple(upload_topics, candidate_topics):
    """Calculate Jaccard similarity between topic lists"""
    if not upload_topics or not candidate_topics:
        return 0.0
    upload_set = set(str(t).lower().strip() for t in upload_topics if t)
    candidate_set = set(str(t).lower().strip() for t in candidate_topics if t)
    if not upload_set or not candidate_set:
        return 0.0
    intersection = len(upload_set & candidate_set)
    union = len(upload_set | candidate_set)
    return intersection / union if union > 0 else 0.0

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

    # Infer bias for uploaded article
    with st.spinner("Inferring article bias..."):
        bias_uploaded = infer_bias_from_text(text, client)
    st.success(f"Detected bias: {interpret_bias(bias_uploaded)} ({bias_uploaded:.2f})")

    # Topic and stance embeddings
    with st.spinner("Generating embeddings..."):
        topic_vec = encode_topic(text)
        stance_vec = encode_stance(text)

    # Use semantic search instead of loading all docs
    with st.spinner("Searching database..."):
        # Get top candidates using similarity search
        search_results = topic_coll.query(
            query_embeddings=[topic_vec.tolist()],
            n_results=min(100, topic_coll.count()),
            include=["embeddings", "metadatas"]
        )
        
        if not search_results["ids"] or not search_results["ids"][0]:
            st.error("No matching articles found in database.")
            st.stop()
        
        candidate_indices = search_results["ids"][0]
        candidate_embeddings = search_results["embeddings"][0]
        candidate_metadatas = search_results["metadatas"][0]

    # Get all stance docs for matching
    stance_docs = stance_coll.get(include=["embeddings", "metadatas"])
    stance_dict = {s_md.get("id", "").split("::")[0]: (s_emb, s_md) 
                   for s_emb, s_md in zip(stance_docs["embeddings"], stance_docs["metadatas"])}

    all_matches = []
    for i, (emb, md) in enumerate(zip(candidate_embeddings, candidate_metadatas)):
        # Calculate similarities
        summary_similarity = float(cosine_similarity([topic_vec], [emb])[0][0])
        
        # Match stance by article ID
        article_id_base = md.get("id", "").split("::")[0]
        stance_data = stance_dict.get(article_id_base)
        stance_sim = 0.0
        tone_db = 0.0
        
        if stance_data is not None:
            stance_match_emb, _ = stance_data
            stance_sim = float(cosine_similarity([stance_vec], [stance_match_emb])[0][0])
        
        # Extract bias and tone from metadata
        bias_db = float(md.get("bias_score", 0.0))
        
        # Get tone from stance metadata if available
        if stance_data:
            _, s_md = stance_data
            tone_db = float(s_md.get("tone_score", bias_db))  # Fallback to bias_score
        
        # Calculate topic overlap (simplified - using summary similarity as proxy)
        # In full version, would extract and match canonical topics
        candidate_topics = md.get("topics_flat", [])
        canonical_overlap = summary_similarity  # Simplified proxy
        if isinstance(candidate_topics, list) and len(candidate_topics) > 0:
            # Could improve by using actual topic matching here
            pass
        
        bias_diff = abs(bias_uploaded - bias_db)
        tone_diff = abs(bias_uploaded - tone_db)  # Using bias as proxy for tone
        
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
    
    if df.empty:
        st.warning("No articles matched the search criteria.")
    else:
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "anti_echo_results.csv", mime="text/csv")

    if st.button("Reset and Upload Another"):
        st.rerun()
