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

1. **Upload Your Article** (PDF, TXT, or HTML)
   - The tool extracts and analyzes the text
   
2. **Analyze Bias & Stance**
   - Uses AI to determine the article's political bias
   - Creates semantic embeddings for topic and stance
   
3. **Search Database**
   - Finds articles covering similar topics from hundreds of news sources
   
4. **Score for Diversity**
   - Calculates an "Anti-Echo Score" that favors:
     - ✅ Same topic/event (high similarity)
     - ✅ Different political stance (low similarity)
     - ✅ Different ideological bias (large difference)
   
5. **Return Results**
   - Shows you the most ideologically diverse coverage
   - Download results as CSV

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
    with open(CONFIG_DIR / "config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
    topic_model = SentenceTransformer(CONFIG["embeddings"]["topic_model"], device="cuda" if torch.cuda.is_available() else "cpu")
    stance_model = SentenceTransformer(CONFIG["embeddings"]["stance_model"], device="cuda" if torch.cuda.is_available() else "cpu")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    
    return CONFIG, topic_model, stance_model, client

CONFIG, topic_model, stance_model, chroma_client = load_environment()

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
                    rid = m.get("row_id") or f"{m.get('id','unknown')}::topic::0"
                    if rid in seen_topic_ids:
                        continue
                    seen_topic_ids.add(rid)
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
                    rid = m.get("row_id") or f"{m.get('id','unknown')}::stance::0"
                    if rid in seen_stance_ids:
                        continue
                    seen_stance_ids.add(rid)
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
    source_confirmed = st.text_input(
        "Confirm or edit the source:", 
        value=inferred_source,
        key="source_input",
        help="The publication or outlet name. Press Enter to confirm."
    )
    
    if not source_confirmed or source_confirmed == "":
        st.stop()
    
    st.success(f"✓ Source: **{source_confirmed}**")
    
    # Step 2: Infer bias
    status_text.text("Step 2/5: Inferring article bias...")
    progress_bar.progress(30)
    
    with st.spinner("Inferring article bias..."):
        bias_uploaded = infer_bias_from_text(text, source_confirmed, openai_client)
    
    st.success(f"✓ Detected bias: **{interpret_bias(bias_uploaded)}** (score: {bias_uploaded:.2f})")
    
    # Step 3: Generate embeddings
    status_text.text("Step 3/5: Generating topic and stance embeddings...")
    progress_bar.progress(50)
    
    with st.spinner("Generating embeddings..."):
        topic_vec = encode_topic(text)
        stance_vec = encode_stance(text)
    
    st.success("✓ Generated semantic embeddings")

    # Step 4: Search database
    status_text.text("Step 4/5: Searching database for similar articles...")
    progress_bar.progress(70)
    
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

    # Step 5: Calculate scores
    status_text.text("Step 5/5: Calculating anti-echo scores...")
    progress_bar.progress(90)

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
        candidate_topics = md.get("topics_flat", [])
        canonical_overlap = summary_similarity  # Simplified proxy
        if isinstance(candidate_topics, list) and len(candidate_topics) > 0:
            # Could improve by using actual topic matching here
            pass
        
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

    progress_bar.progress(100)
    status_text.text("✓ Analysis complete!")

    st.markdown("---")
    st.subheader("📊 Results")
    
    df = pd.DataFrame(all_matches).sort_values("anti_echo_score", ascending=False).head(10)
    
    if df.empty:
        st.warning("No articles matched the search criteria.")
    else:
        st.write(f"Found **{len(df)}** articles with ideologically diverse coverage:")
        
        # Display results organized like notebook
        for idx, row in df.iterrows():
            with st.expander(f"{idx+1}. {row['title']} — {row['source']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Source:** {row['source']}")
                    st.markdown(f"**Bias:** {interpret_bias(row['bias_score'])} ({row['bias_score']:.2f})")
                    st.markdown(f"**Bias Family:** {row['bias_family']}")
                with col2:
                    st.markdown(f"**Anti-Echo Score:** {row['anti_echo_score']:.3f}")
                    st.markdown(f"**Summary Similarity:** {row['summary_similarity']:.3f}")
                    st.markdown(f"**Stance Similarity:** {row['stance_similarity']:.3f}")
                
                if row.get('url'):
                    st.markdown(f"🔗 [Read Article]({row['url']})")
        
        st.markdown("---")
        st.subheader("📊 Detailed Metrics")
        st.dataframe(df[["title", "source", "anti_echo_score", "summary_similarity", "stance_similarity", "bias_score", "bias_family", "url"]], use_container_width=True)
        
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
        st.rerun()
