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
from huggingface_hub import list_repo_files, hf_hub_download
from itertools import product

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
# SIDEBAR SETTINGS (NEEDED FIRST FOR OTHER PARTS)
# ------------------------------------------------------------
st.sidebar.header("🔑 Setup")

# OpenAI API Key
api_key_input = st.sidebar.text_input(
    "OpenAI API Key", 
    type="password",
    help="Required for bias inference and article analysis",
    value=os.environ.get("OPENAI_API_KEY", "")
)

if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input
    openai_client = OpenAI(api_key=api_key_input)
else:
    openai_client = None

st.sidebar.markdown("---")

# Tunable Parameters from Stage 6
st.sidebar.subheader("⚙️ Tunable Parameters")
st.sidebar.markdown("""
**Adjust these weights to control the anti-echo scoring formula:**
""")

w_T_canonical = st.sidebar.slider(
    "w_T_canonical (Topic Overlap Weight)", 
    0.0, 50.0, 0.5, 0.5,
    help="Weight for canonical topic overlap similarity"
)
w_T_summary = st.sidebar.slider(
    "w_T_summary (Summary Similarity Weight)", 
    0.0, 50.0, 10.0, 0.5,
    help="Weight for semantic summary similarity (most important)"
)
w_S = st.sidebar.slider(
    "w_S (Stance Similarity Penalty)", 
    0.0, 50.0, 1.0, 0.5,
    help="Penalty for similar stances (lower = more different stances)"
)
w_B = st.sidebar.slider(
    "w_B (Bias Difference Penalty)", 
    0.0, 50.0, 0.8, 0.5,
    help="Penalty for similar bias scores"
)
w_Tone = st.sidebar.slider(
    "w_Tone (Tone Difference Penalty)", 
    0.0, 50.0, 0.3, 0.5,
    help="Penalty for similar tones"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Formula:** `(topic_weight × overlap) + (summary_weight × similarity) - (stance_penalty × stance_sim) - (bias_penalty × bias_diff) - (tone_penalty × tone_diff)`")

# ------------------------------------------------------------
# HEADER & INTRODUCTION
# ------------------------------------------------------------
st.title("📰 Anti-Echo Chamber — Ideological Diversity Finder")

intro_text = """
### 🎯 What This Tool Does

This tool helps you find **ideologically diverse coverage** of news stories you're reading. 

**The Problem:** Many people only see news from sources aligned with their views, creating "echo chambers" that reinforce existing beliefs.

**The Solution:** Upload any news article, and this tool will find articles from different ideological perspectives that cover the **same topic or event**.

---

### 📊 How It Works

1. **Upload & Source Confirmation**
   - Extracts text using `pdfplumber`, `BeautifulSoup`, or direct text parsing
   - GPT-4o-mini infers the publication source, you confirm or edit it
   - GPT-4o-mini classifies political bias (-1.0 to +1.0) by matching against known outlets

2. **Generate Topic Summary**
   - GPT-4o-mini generates a one-sentence topic summary (max 20 words)
   - Example: "Trump's plan to use unspent Defense Department funds to pay military personnel during a government shutdown violates the Constitution"

3. **Generate Composite Topic Embedding**
   - Base topic embedding: Uses `intfloat/e5-base-v2` (768 dim) on article text
   - Summary embedding: Uses `intfloat/e5-base-v2` on the GPT topic summary
   - Composite: Concatenates base + summary = 1536-dim vector

4. **Generate Stance Embedding**
   - GPT-4o-mini classifies: political leaning, implied stance, argument summary
   - These texts are concatenated and embedded with `all-mpnet-base-v2` (768 dim)

5. **Semantic Search in ChromaDB**
   - Retrieves ALL topic documents from the database
   - For each candidate: compares canonical topics (Jaccard + semantic) and summary embeddings
   - Filters candidates with topic overlap ≥ 0.0 and summary similarity ≥ 0.8
   - Deduplicates to keep best match per unique article

6. **Anti-Echo Scoring & Results**
   - Calculates weighted score favoring high topic similarity, low stance similarity, large bias differences
   - Displays organized results by similarity type

---

### 🧮 Scoring Formula

**Higher scores = better matches** (same topic, different viewpoint):

```
anti_echo_score =
(w_T_canonical × canonical_overlap)
+ (w_T_summary × summary_similarity)
  - (w_S × stance_similarity)
  - (w_B × bias_diff)
  - (w_Tone × tone_diff)
```

**Variables:**
- `canonical_overlap`: Topic label overlap (Jaccard similarity)
- `summary_similarity`: Semantic similarity of article summaries
- `stance_similarity`: How similar the political stance is
- `bias_diff`: Difference in political bias scores (-1.0 to 1.0)
- `tone_diff`: Difference in tone/sentiment

Adjust the weights in the sidebar to tune the algorithm!
"""

st.markdown(intro_text)

# ------------------------------------------------------------
# LOAD MODELS & DATABASE
# ------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_environment():
    """Load models with caching"""
    with open(CONFIG_DIR / "config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
    topic_model = SentenceTransformer(CONFIG["embeddings"]["topic_model"], device="cuda" if torch.cuda.is_available() else "cpu")
    stance_model = SentenceTransformer(CONFIG["embeddings"]["stance_model"], device="cuda" if torch.cuda.is_available() else "cpu")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    
    # Load topic anchors for semantic topic matching
    try:
        anchors_path = CONFIG_DIR / "topic_anchors.npz"
        if anchors_path.exists():
            anchors_npz = np.load(anchors_path, allow_pickle=True)
            topic_anchors = {k: anchors_npz[k] for k in anchors_npz.files}
        else:
            topic_anchors = None
    except Exception as e:
        topic_anchors = None
    
    return CONFIG, topic_model, stance_model, client, topic_anchors

CONFIG, topic_model, stance_model, chroma_client, topic_anchors = load_environment()

# Get or create collections (not cached to allow for rebuilds)
def get_collections():
    topic_name = CONFIG["chroma_collections"]["topic"]
    stance_name = CONFIG["chroma_collections"]["stance"]
    
    try:
        topic_coll = chroma_client.get_collection(topic_name)
    except Exception:
        topic_coll = chroma_client.create_collection(topic_name, metadata={"hnsw:space": "cosine"})
    
    try:
        stance_coll = chroma_client.get_collection(stance_name)
    except Exception:
        stance_coll = chroma_client.create_collection(stance_name, metadata={"hnsw:space": "cosine"})
    
    return topic_coll, stance_coll

topic_coll, stance_coll = get_collections()

# Check if database is empty and rebuild if needed
def rebuild_chromadb_from_hf(config_dict):
    """Rebuild ChromaDB from Hugging Face dataset"""
    HF_REPO = config_dict.get("hf_dataset_id")
    if not HF_REPO:
        return False, "No HF dataset ID in config"
    
    st.info(f"📥 Attempting to rebuild from: {HF_REPO}")
    
    topic_name = config_dict["chroma_collections"]["topic"]
    stance_name = config_dict["chroma_collections"]["stance"]
    
    try:
        # Recreate collections
        from chromadb import PersistentClient
        client = PersistentClient(path=str(CHROMA_PATH))
        
        # Delete and recreate
        try:
            client.delete_collection(topic_name)
        except:
            pass
        try:
            client.delete_collection(stance_name)
        except:
            pass
            
        topic_coll_new = client.create_collection(topic_name, metadata={"hnsw:space": "cosine"})
        stance_coll_new = client.create_collection(stance_name, metadata={"hnsw:space": "cosine"})
        
        # Download and load data
        st.info(f"Fetching files from {HF_REPO}...")
        files = list(list_repo_files(HF_REPO, repo_type="dataset"))
        
        st.info(f"📁 Found {len(files)} files in repository")
        
        # Show first few files for debugging
        if files:
            st.write(f"Sample files: {files[:3]}")
        
        if not files:
            return False, f"No files found in {HF_REPO}"
        
        batches = sorted({"/".join(f.split("/")[:2]) for f in files if f.startswith("batches/")})
        
        st.info(f"📦 Found {len(batches)} batch directories")
        
        if not batches:
            # Show what files we actually found
            unique_dirs = sorted(set(f.split("/")[0] for f in files if "/" in f))
            return False, f"No batches found in {HF_REPO}. Found directories: {unique_dirs[:10]}"
        
        st.info(f"✅ Processing {len(batches)} batches...")
        
        topic_total = stance_total = 0
        
        def load_npz_safely(path):
            arr = np.load(path, allow_pickle=False)
            if isinstance(arr, np.lib.npyio.NpzFile):
                for key in arr.files:
                    if arr[key].ndim == 2:
                        return arr[key]
                raise ValueError(f"No 2D arrays found in {path}")
            return arr
        
        def load_jsonl(fp):
            records = []
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except:
                        continue
            return records
        
        seen_topic_ids, seen_stance_ids = set(), set()
        
        for idx, batch in enumerate(batches, 1):
            try:
                st.info(f"Processing batch {idx}/{len(batches)}: {batch.split('/')[-1]}...")
                
                topic_npz = hf_hub_download(HF_REPO, f"{batch}/embeddings_topic.npz", repo_type="dataset")
                stance_npz = hf_hub_download(HF_REPO, f"{batch}/embeddings_stance.npz", repo_type="dataset")
                meta_topic = hf_hub_download(HF_REPO, f"{batch}/metadata_topic.jsonl", repo_type="dataset")
                meta_stance = hf_hub_download(HF_REPO, f"{batch}/metadata_stance.jsonl", repo_type="dataset")
                
                t_embs, s_embs = load_npz_safely(topic_npz), load_npz_safely(stance_npz)
                t_meta, s_meta = load_jsonl(meta_topic), load_jsonl(meta_stance)
                
                # Upsert topic
                t_records = []
                for e, m in zip(t_embs, t_meta):
                    if not isinstance(m, dict):
                        continue
                    
                    rid = m.get("row_id") or f"{m.get('id','unknown')}::topic::0"
                    if rid in seen_topic_ids:
                        continue
                    seen_topic_ids.add(rid)
                    
                    # Ensure metadata is a dict - pass through as-is
                    t_records.append((rid, e, m))
                
                if t_records:
                    topic_coll_new.upsert(
                        ids=[r[0] for r in t_records],
                        embeddings=[r[1].tolist() for r in t_records],
                        metadatas=[r[2] for r in t_records]
                    )
                topic_total += len(t_records)
                
                # Upsert stance
                s_records = []
                for e, m in zip(s_embs, s_meta):
                    if not isinstance(m, dict):
                        continue
                    
                    rid = m.get("row_id") or f"{m.get('id','unknown')}::stance::0"
                    if rid in seen_stance_ids:
                        continue
                    seen_stance_ids.add(rid)
                    
                    # Ensure metadata is a dict - pass through as-is
                    s_records.append((rid, e, m))
                
                if s_records:
                    stance_coll_new.upsert(
                        ids=[r[0] for r in s_records],
                        embeddings=[r[1].tolist() for r in s_records],
                        metadatas=[r[2] for r in s_records]
                    )
                stance_total += len(s_records)
                
            except Exception as e:
                st.warning(f"⚠️ Batch {batch} failed: {type(e).__name__}: {e}")
                continue
        
        if topic_total == 0 and stance_total == 0:
            return False, f"Rebuild completed but no vectors were added. Check if {HF_REPO} has data."
        
        return True, f"Rebuilt: {topic_total} topic vectors, {stance_total} stance vectors from {len(batches)} batches"
        
    except Exception as e:
        import traceback
        return False, f"Rebuild failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"

# Check if database is empty
try:
    topic_count = topic_coll.count()
    stance_count = stance_coll.count()
except Exception as e:
    # Collections don't exist or can't be accessed, rebuild needed
    st.warning("⚠️ **ChromaDB is empty or needs initialization** - Rebuilding from Hugging Face...")
    with st.spinner("Downloading embeddings from Hugging Face. This may take a few minutes..."):
        success, message = rebuild_chromadb_from_hf(CONFIG)
        if success:
            st.success(f"✓ {message}")
            st.rerun()  # Reload with new data
        else:
            st.error(f"✗ {message}")
            st.stop()
    # This code won't execute if st.rerun() is called, but for safety:
    topic_count = 0
    stance_count = 0

if topic_count == 0 or stance_count == 0:
    st.warning("⚠️ **ChromaDB is empty** - Rebuilding from Hugging Face...")
    with st.spinner("Downloading embeddings from Hugging Face. This may take a few minutes..."):
        success, message = rebuild_chromadb_from_hf(CONFIG)
        if success:
            st.success(f"✓ {message}")
            st.rerun()  # Reload with new data
        else:
            st.error(f"✗ {message}")
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

def infer_source_name(text):
    """Extract source name from article text"""
    # Try to find URL or domain name
    url_match = re.search(r'https?://([^/\s]+)', text)
    if url_match:
        domain = url_match.group(1).lower()
        domain = domain.replace("www.", "")
        source = domain.split(".")[0]
        return source.capitalize()
    return "Unknown"

def infer_bias_from_text(text, source_name, client):
    """Infer bias score from article text using GPT and source"""
    try:
        prompt = f"""Analyze this article and determine its political bias score.
        Article is from: {source_name}
        
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

def infer_stance_classification(text, source_name, client):
    """Generate stance classification text for embedding. Returns (stance_text, stance_data)"""
    try:
        prompt = f"""You are a political analyst. Based on the article below, classify its overall political leaning (tone) and implied stance.

        Leaning options: progressive left, center-left, center, center-right, conservative right, libertarian right, unknown
        Stance examples: critical of government, supportive of business, anti-war, pro-immigration, fiscal conservative
        
        Return JSON with fields:
        - political_leaning (string)
        - implied_stance (string)
        - summary (one-sentence summary of the article's main argument)
        
        Article from: {source_name}
        Excerpt: {text[:2000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.4
        )
        
        raw = response.choices[0].message.content.strip()
        try:
            # Try to parse as JSON, handling code blocks
            cleaned = raw.strip('```json').strip('```').strip()
            stance_data = json.loads(cleaned)
            # Combine fields for embedding
            stance_text = f"{stance_data.get('political_leaning', 'unknown')}\n{stance_data.get('implied_stance', 'unknown')}\n{stance_data.get('summary', raw)}"
            return stance_text, stance_data
        except:
            import re
            # Fallback: extract with regex
            leaning = re.search(r'"political_leaning":\s*"(.+?)"', raw, re.I)
            stance = re.search(r'"implied_stance":\s*"(.+?)"', raw, re.I)
            summary = re.search(r'"summary":\s*"(.+?)"', raw, re.I)
            stance_data_fallback = {
                "political_leaning": (leaning.group(1) if leaning else "unknown"),
                "implied_stance": (stance.group(1) if stance else "unknown"),
                "summary": (summary.group(1) if summary else raw[:200])
            }
            stance_text = f"{stance_data_fallback['political_leaning']}\n{stance_data_fallback['implied_stance']}\n{stance_data_fallback['summary']}"
            return stance_text, stance_data_fallback
    except:
        fallback_data = {"political_leaning": "unknown", "implied_stance": "unknown", "summary": "No summary available"}
        return "unknown\nunknown\nNo summary available", fallback_data

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
# FILE UPLOAD & ANALYSIS
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📄 Upload Article")

uploaded = st.file_uploader(
    "Choose an article to analyze:", 
    type=['pdf', 'txt', 'html', 'htm'],
    help="Upload a PDF, text file, or HTML file containing a news article"
)

if uploaded:
    if not openai_client:
        st.error("Please enter your OpenAI API key first.")
        st.stop()

    # Extract text
    text = extract_text(uploaded)
    st.success(f"✓ Extracted {len(text):,} characters from {uploaded.name}")
    
    st.markdown("---")
    st.subheader("🔍 Analysis Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Infer source
    status_text.text("Step 1/5: Detecting source...")
    progress_bar.progress(10)
    
    inferred_source = infer_source_name(text)
    
    # Store in session state
    if 'source_confirmed' not in st.session_state:
        st.session_state.source_confirmed = None
    
    source_confirmed = st.text_input(
        "Confirm or edit the source:", 
        value=inferred_source,
        key="source_input",
        help="The publication or outlet name. Press Enter to confirm."
    )
    
    # Button to confirm
    if st.button("✓ Confirm Source", type="primary"):
        st.session_state.source_confirmed = source_confirmed
    
    if not st.session_state.source_confirmed:
        st.info("👆 Please confirm the source above to continue")
        st.stop()
    
    st.success(f"✓ Source: **{st.session_state.source_confirmed}**")
    
    # Step 2: Infer bias
    status_text.text("Step 2/5: Inferring article bias...")
    progress_bar.progress(30)
    
    with st.spinner("Inferring article bias..."):
        bias_uploaded = infer_bias_from_text(text, st.session_state.source_confirmed, openai_client)
    
    st.success(f"✓ Detected bias: **{interpret_bias(bias_uploaded)}** (score: {bias_uploaded:.2f})")
    
    # Step 3: Generate topic summary with GPT
    status_text.text("Step 3/5: Generating GPT topic summary...")
    progress_bar.progress(45)
    
    topic_summary_prompt = f"""Summarize the main subject matter of this article in one sentence (max 20 words).
    Be concrete and specific about what event, person, location, or issue is discussed.
    Focus on the WHO/WHAT/WHERE, not opinion or analysis.
    
    Article title: {uploaded.name}
    Text excerpt: {text[:2500]}
    
    Summary:"""
    
    with st.spinner("Generating GPT topic summary..."):
        summary_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": topic_summary_prompt}],
            max_tokens=40,
            temperature=0.3
        )
        topic_summary = summary_response.choices[0].message.content.strip().strip('"').strip("'")
    
    st.success(f"✓ Topic summary: {topic_summary}")
    
    # Step 3.5: Extract canonical topics (needed for matching)
    # For now, use topic_summary as a proxy - the full implementation would match against anchors
    # This is a simplified version - in the full notebook, it does hierarchical clustering and matches to anchors
    upload_topics_list = []  # Empty for now, will use summary similarity primarily
    
    # Step 4: Generate embeddings (composite topic + stance)
    status_text.text("Step 4/5: Generating embeddings...")
    progress_bar.progress(60)
    
    with st.spinner("Generating embeddings..."):
        # Base topic embedding
        topic_vec_base = encode_topic(text)
        # Embed the GPT summary
        topic_summary_vec = encode_topic(topic_summary)
        
        # Composite: [base | summary]
        if topic_vec_base.ndim == 2:
            topic_vec_base = topic_vec_base.flatten()
        if topic_summary_vec.ndim == 2:
            topic_summary_vec = topic_summary_vec.flatten()
        topic_vec = np.concatenate([topic_vec_base, topic_summary_vec])
        
        # Generate stance classification with GPT
        stance_text, stance_data = infer_stance_classification(text, st.session_state.source_confirmed, openai_client)
        
        # Display stance metadata (like notebook Stage 5b output)
        st.markdown("**Stance Classification:**")
        st.json(stance_data)
        
        # Embed the stance classification
        stance_vec = encode_stance(stance_text)
    
    st.success("✓ Generated composite topic embeddings and stance embeddings")

    # Step 5: Search database
    status_text.text("Step 5/6: Searching database for similar articles...")
    progress_bar.progress(75)
    
    with st.spinner("Retrieving all topic documents..."):
        # Like notebook line 2365: get ALL topic docs (not query)
        topic_docs = topic_coll.get(include=["embeddings", "metadatas"])
    
    candidate_embeddings = topic_docs["embeddings"]
    candidate_metadatas = topic_docs["metadatas"]

    # Get all stance docs for matching
    stance_docs = stance_coll.get(include=["embeddings", "metadatas"])
    stance_dict = {s_md.get("id", "").split("::")[0]: (s_emb, s_md) 
                   for s_emb, s_md in zip(stance_docs["embeddings"], stance_docs["metadatas"])}

    # Step 6: Calculate scores
    status_text.text("Step 6/6: Calculating anti-echo scores...")
    progress_bar.progress(85)

    # Use the topics list we have (currently empty, using summary similarity primarily)

    # Constants from notebook
    # NOTE: Set topic threshold to 0.0 since we don't extract canonical topics for uploaded article
    # The notebook extracts topics via hierarchical clustering + anchor matching in Stage 5a
    # For simplicity, we rely primarily on summary similarity (0.8 threshold)
    CANONICAL_TOPIC_THRESHOLD = 0.0  # Disabled: app doesn't extract canonical topics yet
    SUMMARY_SIMILARITY_THRESHOLD = 0.8

    all_matches = []
    # Extract uploaded summary vector
    uploaded_summary = topic_vec[768:] if len(topic_vec) == 1536 else topic_vec
    
    st.caption(f"Checking {len(candidate_embeddings)} topic vectors from database...")
    passed_summary = 0
    passed_topic = 0
    
    # Check if we need to encode old summaries
    old_format_count = 0
    for emb, md in zip(candidate_embeddings[:10], candidate_metadatas[:10]):  # Sample first 10
        if len(emb) == 768 and md.get("openai_topic_summary"):
            old_format_count += 1
    if old_format_count > 0:
        st.info(f"⚠️ Found {old_format_count}/10 sample vectors with old format (768-dim + text summary). These require encoding.")
    else:
        st.info("✓ All sample vectors appear to be in composite 1536-dim format (no encoding needed)")
    
    progress_bar_search = st.progress(0)
    total_candidates = len(candidate_embeddings)
    
    for i, (emb, md) in enumerate(zip(candidate_embeddings, candidate_metadatas)):
        # Update progress every 50 items
        if i % 50 == 0:
            progress_bar_search.progress(i / total_candidates)
        # NOTEBOOK ORDER: canonical overlap FIRST (line 2394)
        # Calculate canonical topic overlap using Jaccard similarity (same as notebook)
        def parse_topics(obj):
            if obj is None:
                return []
            if isinstance(obj, list):
                return [t.strip() for t in obj if t.strip()]
            if isinstance(obj, str):
                parts = [t.strip() for t in obj.split(";") if t.strip()]
                if len(parts) == 1 and parts[0].startswith("["):
                    try:
                        parsed = json.loads(parts[0])
                        if isinstance(parsed, list):
                            return [t.strip() for t in parsed if isinstance(t, str)]
                    except Exception:
                        pass
                return parts
            return []
        
        candidate_topics = md.get("topics_flat", [])
        upload_topics = upload_topics_list
        
        A = set([t.strip().lower() for t in parse_topics(upload_topics)])
        B = set([t.strip().lower() for t in parse_topics(candidate_topics)])
        
        if not A or not B:
            canonical_overlap = 0.0
        else:
            jaccard = len(A & B) / len(A | B)
            # Try semantic similarity if anchors available (like notebook)
            if topic_anchors is not None:
                sims = []
                for a, b in product(A, B):
                    if a in topic_anchors and b in topic_anchors:
                        va, vb = topic_anchors[a], topic_anchors[b]
                        sim = float(cosine_similarity([va], [vb])[0][0])
                        sims.append(sim)
                if sims:
                    canonical_overlap = max(jaccard, max(sims))
                else:
                    canonical_overlap = jaccard
            else:
                canonical_overlap = jaccard
        
        # FILTER: Check canonical topic overlap threshold (notebook line 2395)
        if canonical_overlap < CANONICAL_TOPIC_THRESHOLD:
            continue
        
        # NOTEBOOK ORDER: summary similarity SECOND (line 2398-2411)
        emb_array = np.array(emb)
        if len(emb_array) == 1536:
            # New composite format: compare summary portions
            candidate_summary_vec = emb_array[768:]
            summary_similarity = float(cosine_similarity(uploaded_summary.reshape(1, -1), candidate_summary_vec.reshape(1, -1))[0][0])
        else:
            # Old 768-dim format: check if has summary text
            old_summary = md.get("openai_topic_summary", "")
            if old_summary:
                # Encode the text summary and compare
                def encode_text(text):
                    vec = topic_model.encode(text, normalize_embeddings=True, show_progress_bar=False)
                    if vec.ndim == 2:
                        vec = vec.flatten()
                    return vec
                candidate_summary_vec = encode_text(old_summary)
                summary_similarity = float(cosine_similarity(uploaded_summary.reshape(1, -1), candidate_summary_vec.reshape(1, -1))[0][0])
            else:
                # Fallback: compare with base topic
                uploaded_base_topic = topic_vec[:768] if len(topic_vec) == 1536 else topic_vec
                summary_similarity = float(cosine_similarity(uploaded_base_topic.reshape(1, -1), emb_array.reshape(1, -1))[0][0])
        
        # FILTER: Check summary similarity threshold (notebook line 2411)
        if summary_similarity < SUMMARY_SIMILARITY_THRESHOLD:
            continue
        
        # Match stance by article ID
        article_id_base = md.get("id", "").split("::")[0]
        stance_data = stance_dict.get(article_id_base)
        stance_sim = 0.0
        tone_db = 0.0
        
        if stance_data is not None:
            stance_match_emb, s_md_stance = stance_data
            stance_sim = float(cosine_similarity([stance_vec], [stance_match_emb])[0][0])
        else:
            s_md_stance = None
        
        # Extract bias from stance metadata (not topic metadata!)
        bias_db = 0.0
        tone_db = 0.0
        
        # Get bias from stance metadata
        if s_md_stance:
            try:
                bias_db = float(s_md_stance.get("bias_score", 0.0))
            except (ValueError, TypeError):
                pass
            
            try:
                tone_db = float(s_md_stance.get("tone_score", bias_db))
            except (ValueError, TypeError):
                tone_db = bias_db
        
        bias_diff = abs(bias_uploaded - bias_db)
        tone_diff = abs(bias_uploaded - tone_db)
        
        anti_echo_score = (
            (w_T_canonical * canonical_overlap)
            + (w_T_summary * summary_similarity)
            - (w_S * stance_sim)
            - (w_B * bias_diff)
            - (w_Tone * tone_diff)
        )
        all_matches.append({
            "article_id": article_id_base,
            "title": md.get("title", "Untitled"),
            "source": md.get("source", "Unknown"),
            "bias_family": md.get("bias_family", ""),
            "bias_score": bias_db,
            "canonical_overlap": canonical_overlap,
            "summary_similarity": summary_similarity,
            "stance_similarity": stance_sim,
            "bias_diff": bias_diff,
            "tone_diff": tone_diff,
            "anti_echo_score": anti_echo_score,
            "url": md.get("url", "")
        })
    
    progress_bar_search.progress(1.0)

    st.caption(f"✓ {passed_summary} passed summary threshold ≥ {SUMMARY_SIMILARITY_THRESHOLD}, {passed_topic} passed topic threshold ≥ {CANONICAL_TOPIC_THRESHOLD}, {len(all_matches)} before deduplication")
    
    # Deduplicate by article_id, keep best match (notebook line 2431-2437)
    best_matches = {}
    for match in all_matches:
        aid = match.get("article_id", match.get("source", "unknown"))
        if aid not in best_matches or match["summary_similarity"] > best_matches[aid]["summary_similarity"]:
            best_matches[aid] = match
    
    st.caption(f"✓ After deduplication: {len(best_matches)} unique articles")

    progress_bar.progress(100)
    status_text.text("✓ Analysis complete!")
    
    if not best_matches:
        st.warning("⚠️ **No articles matched the search criteria after filtering.**")
        st.info("This could mean:\n1. No articles in the database share similar topics\n2. Thresholds are too strict (canonical ≥ 0.3, summary ≥ 0.8)")
        st.stop()
    
    df = pd.DataFrame(best_matches.values()).sort_values("anti_echo_score", ascending=False).head(100)
    
    st.markdown("---")
    
    # SECTION 1: Same Topic — Different Source Bias
    st.subheader("📰 Same Topic — Different Source Bias")
    st.markdown("*Articles covering the same topic but with different ideological perspectives*")
    
    diff_bias = df[df['bias_diff'] > 0.3].sort_values(['bias_diff', 'summary_similarity'], ascending=[False, False]).head(3)
    
    if not diff_bias.empty:
        for idx, (_, row) in enumerate(diff_bias.iterrows(), 1):
            st.markdown(f"### {idx}. [{row['title']}]({row['url'] if row.get('url') else '#'})")
            st.markdown(f"**Source:** {row['source']} | **Bias:** {interpret_bias(row['bias_score'])} ({row['bias_score']:.2f})")
            st.markdown(f"**Anti-Echo Score:** {row['anti_echo_score']:.3f} | **Bias Difference:** {row['bias_diff']:.2f} | **Summary Similarity:** {row['summary_similarity']:.3f}")
    else:
        st.info("No articles found with significant bias differences.")
    
    st.markdown("---")
    
    # SECTION 2: Same Topic — Opposite Stance  
    st.subheader("📝 Same Topic — Opposite Stance")
    st.markdown("*Articles covering the same topic but with opposing political stances*")
    
    opp_stance = df[df['stance_similarity'] < 0.7].sort_values(['stance_similarity', 'summary_similarity'], ascending=[True, False]).head(3)
    
    if not opp_stance.empty:
        for idx, (_, row) in enumerate(opp_stance.iterrows(), 1):
            st.markdown(f"### {idx}. [{row['title']}]({row['url'] if row.get('url') else '#'})")
            st.markdown(f"**Source:** {row['source']} | **Bias:** {interpret_bias(row['bias_score'])} ({row['bias_score']:.2f})")
            st.markdown(f"**Anti-Echo Score:** {row['anti_echo_score']:.3f} | **Stance Similarity:** {row['stance_similarity']:.3f} | **Summary Similarity:** {row['summary_similarity']:.3f}")
    else:
        st.info("No articles found with opposing stances.")
    
    st.markdown("---")
    
    # SECTION 3: Top Anti-Echo Candidates
    st.subheader("🏆 Top Anti-Echo Candidates (Best Overall Matches)")
    st.markdown("*Articles that best balance topic similarity with ideological diversity*")
    
    top_candidates = df.head(3)
    
    for idx, (_, row) in enumerate(top_candidates.iterrows(), 1):
        st.markdown(f"### {idx}. [{row['title']}]({row['url'] if row.get('url') else '#'})")
        st.markdown(f"**Source:** {row['source']} | **Bias:** {interpret_bias(row['bias_score'])} ({row['bias_score']:.2f})")
        st.markdown(f"**Anti-Echo Score:** {row['anti_echo_score']:.3f} | **Bias Diff:** {row['bias_diff']:.2f} | **Summary Sim:** {row['summary_similarity']:.3f} | **Stance Sim:** {row['stance_similarity']:.3f}")
    
    st.markdown("---")
    st.subheader("📊 All Results (Full Dataset)")
    st.dataframe(df[["title", "source", "anti_echo_score", "summary_similarity", "stance_similarity", "bias_score", "bias_family", "url"]], use_container_width=True, height=400)
    
    st.markdown("---")
    st.subheader("📥 Download Results")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📄 Download Full Results as CSV", 
        csv, 
        f"anti_echo_results.csv", 
        mime="text/csv",
        help="Download the complete analysis with all metrics"
    )

if st.button("🔄 Analyze Another Article"):
    st.session_state.source_confirmed = None
    st.rerun()
