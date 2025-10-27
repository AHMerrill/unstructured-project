# ================================================================
# app.py — Anti-Echo Chamber Streamlit Application
# ================================================================

import os, io, re, json, yaml, pdfplumber, traceback
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from huggingface_hub import list_repo_files, hf_hub_download
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
from bs4 import BeautifulSoup
import streamlit as st
import chromadb
import torch
import nltk

# ------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ------------------------------------------------------------
st.set_page_config(page_title="Anti-Echo Chamber", layout="wide")

PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_DIR   = PROJECT_ROOT / "config"
TMP          = PROJECT_ROOT / "tmp"
CHROMA_PATH  = PROJECT_ROOT / "chroma_db"
EPHEMERAL    = TMP / "ephemeral_embeddings"
for p in [TMP, EPHEMERAL, CHROMA_PATH]:
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD CONFIG AND CACHE MODELS / DATABASE
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_config_and_models():
    with open(CONFIG_DIR / "config.yaml", "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)

    # --- Load anchors & mappings ---
    anchors_npz = np.load(CONFIG_DIR / "topic_anchors.npz", allow_pickle=True)
    topic_anchors = {k: anchors_npz[k] for k in anchors_npz.files}
    with open(CONFIG_DIR / "source_bias.json", encoding="utf-8") as f:
        source_bias = json.load(f)
    with open(CONFIG_DIR / "political_leanings.json", encoding="utf-8") as f:
        leanings_map = json.load(f)
    with open(CONFIG_DIR / "implied_stances.json", encoding="utf-8") as f:
        stances_map = json.load(f)

    # --- Load models ---
    topic_model = SentenceTransformer(CONFIG["embeddings"]["topic_model"], device="cuda" if torch.cuda.is_available() else "cpu")
    stance_model = SentenceTransformer(CONFIG["embeddings"]["stance_model"], device="cuda" if torch.cuda.is_available() else "cpu")

    # --- Build / load Chroma ---
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    topic_name  = CONFIG["chroma_collections"]["topic"]
    stance_name = CONFIG["chroma_collections"]["stance"]

    try:
        topic_coll = client.get_collection(topic_name)
        stance_coll = client.get_collection(stance_name)
    except Exception:
        # rebuild from HF if missing
        HF_REPO = CONFIG["hf_dataset_id"]
        topic_coll  = client.create_collection(topic_name,  metadata={"hnsw:space": "cosine"})
        stance_coll = client.create_collection(stance_name, metadata={"hnsw:space": "cosine"})
        files = list_repo_files(HF_REPO, repo_type="dataset")
        batches = sorted({"/".join(f.split("/")[:2]) for f in files if f.startswith("batches/")})
        for batch in batches:
            try:
                t_npz  = hf_hub_download(HF_REPO, f"{batch}/embeddings_topic.npz",  repo_type="dataset")
                s_npz  = hf_hub_download(HF_REPO, f"{batch}/embeddings_stance.npz", repo_type="dataset")
                mt     = hf_hub_download(HF_REPO, f"{batch}/metadata_topic.jsonl",  repo_type="dataset")
                ms     = hf_hub_download(HF_REPO, f"{batch}/metadata_stance.jsonl", repo_type="dataset")

                def load_jsonl(fp):
                    with open(fp, "r", encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
                def load_npz(fp):
                    arr = np.load(fp, allow_pickle=False)
                    if isinstance(arr, np.lib.npyio.NpzFile):
                        for k in arr.files:
                            if arr[k].ndim == 2: return arr[k]
                        raise ValueError("no 2D arrays")
                    return arr
                t_embs, s_embs = load_npz(t_npz), load_npz(s_npz)
                t_meta, s_meta = load_jsonl(mt), load_jsonl(ms)
                topic_coll.upsert(
                    ids=[m.get("row_id", f"{m.get('id')}::topic::0") for m in t_meta],
                    embeddings=[e.tolist() for e in t_embs],
                    metadatas=t_meta)
                stance_coll.upsert(
                    ids=[m.get("row_id", f"{m.get('id')}::stance::0") for m in s_meta],
                    embeddings=[e.tolist() for e in s_embs],
                    metadatas=s_meta)
            except Exception as e:
                print("Rebuild batch failed", batch, e)

    return CONFIG, topic_model, stance_model, topic_coll, stance_coll, topic_anchors, source_bias, leanings_map, stances_map

CONFIG, topic_model, stance_model, topic_coll, stance_coll, topic_anchors, SOURCE_BIAS, LEANINGS_MAP, STANCES_MAP = load_config_and_models()

# ------------------------------------------------------------
# NLTK SETUP
# ------------------------------------------------------------
for pkg in ["punkt", "punkt_tab"]:
    try: nltk.data.find(f"tokenizers/{pkg}")
    except LookupError: nltk.download(pkg)

def sent_split(text):
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]

# ------------------------------------------------------------
# SESSION MANAGEMENT
# ------------------------------------------------------------
if "meta" not in st.session_state: st.session_state.meta = None
if "openai_key" not in st.session_state: st.session_state.openai_key = None
if "client" not in st.session_state: st.session_state.client = None
if "results" not in st.session_state: st.session_state.results = None

# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------
def bias_to_score(label):
    l = (label or "").lower().strip()
    if "progressive" in l or ("left" in l and "center" not in l): return -0.8
    if "center left" in l:  return -0.4
    if l == "center":       return 0.0
    if "center right" in l: return 0.4
    if "conservative" in l or "right" in l: return 0.8
    if "libertarian" in l:  return 0.6
    return 0.0

def interpret_bias(score):
    if score <= -0.6: return "Progressive / Left"
    if -0.6 < score <= -0.2: return "Center-Left"
    if -0.2 < score < 0.2: return "Center / Neutral"
    if 0.2 <= score < 0.6: return "Center-Right"
    if score >= 0.6: return "Conservative / Right"
    return "Unknown"

def encode_topic(texts):
    if isinstance(texts, str): texts=[texts]
    return topic_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

def encode_stance(text):
    return stance_model.encode(text, normalize_embeddings=True).reshape(1,-1)

def extract_text(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.read()
    if suffix == ".txt":
        return data.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "".join([p.extract_text() or "" for p in pdf.pages])
    if suffix in [".html",".htm"]:
        soup = BeautifulSoup(data.decode("utf-8","ignore"),"html.parser")
        for s in soup(["script","style"]): s.decompose()
        return soup.get_text(separator=" ")
    return ""

# ------------------------------------------------------------
# 1️⃣ API KEY INPUT
# ------------------------------------------------------------
st.sidebar.header("🔐 OpenAI API Key")
api_key_input = st.sidebar.text_input("Enter your OpenAI API key", type="password", value=st.session_state.openai_key or "")
if api_key_input:
    st.session_state.openai_key = api_key_input
    st.session_state.client = OpenAI(api_key=api_key_input)
    st.sidebar.success("Key stored securely in memory.")

# ------------------------------------------------------------
# 2️⃣ ARTICLE UPLOAD
# ------------------------------------------------------------
st.title("📰 Anti-Echo Chamber")
uploaded = st.file_uploader("Upload an article (.pdf, .txt, .html)")
if uploaded:
    text = extract_text(uploaded)
    st.write(f"Extracted **{len(text)}** characters.")

    # infer source
    client = st.session_state.client
    if not client: st.error("Enter OpenAI key first."); st.stop()

    domain_match = re.search(r"https?://([^/\s]+)", text)
    inferred_source = (domain_match.group(1).split(".")[0] if domain_match else None)
    if not inferred_source:
        prompt = f"Infer the most likely publication from this text:\n{text[:2000]}"
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=20)
        inferred_source = resp.choices[0].message.content.strip()
    st.write(f"Inferred source: **{inferred_source}**")

    # fuzzy match
    match, score, _ = process.extractOne(inferred_source, list(SOURCE_BIAS.keys()), scorer=fuzz.ratio)
    if score >= 85:
        bias_info = SOURCE_BIAS[match]
    else:
        prompt = f"""
Given the outlet name "{inferred_source}", infer its general political bias family.
Return JSON with bias_family, bias_score (-1 to 1), and short_rationale."""
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=128)
        raw = resp.choices[0].message.content
        try: bias_info=json.loads(re.sub(r"```json|```","",raw))
        except: bias_info={"bias_family":"unknown","bias_score":0.0,"short_rationale":raw}
    meta = {
        "filename": uploaded.name,
        "bias_family": bias_info.get("bias_family","unknown"),
        "bias_score": float(bias_info.get("bias_score",0.0)),
        "rationale": bias_info.get("short_rationale",""),
    }
    st.session_state.meta = meta
    st.success(f"Bias: {meta['bias_family']} ({meta['bias_score']})")

    # --------------------------------------------------------
    # 3️⃣ TOPIC EMBEDDING + SUMMARY
    # --------------------------------------------------------
    st.subheader("Topic Embedding + GPT Summary")
    prompt = f"Summarize in one sentence (max 20 words) what this article is about:\n{text[:2500]}"
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=40)
    topic_summary = resp.choices[0].message.content.strip()
    st.write(f"Summary: *{topic_summary}*")

    # cluster sentences
    sents = sent_split(text)
    emb = encode_topic(sents)
    k = min(max(1, len(sents)//8), 8)
    labels = AgglomerativeClustering(n_clusters=k).fit_predict(emb)
    segs = [" ".join([s for s,l in zip(sents,labels) if l==lab]) for lab in sorted(set(labels))]
    topic_vecs = np.vstack([encode_topic(seg).mean(axis=0) for seg in segs])
    primary_topic_vec = topic_vecs[0]
    summary_vec = encode_topic(topic_summary).flatten()
    composite_topic_vec = np.concatenate([primary_topic_vec, summary_vec])
    st.session_state.topic_vec = composite_topic_vec
    st.session_state.topic_summary = topic_summary
    st.success("Topic embeddings created.")

    # --------------------------------------------------------
    # 4️⃣ STANCE ANALYSIS
    # --------------------------------------------------------
    st.subheader("Stance Classification")
    stance_prompt = f"""
Classify the political leaning and implied stance of the article below.
Return JSON with political_leaning, implied_stance, summary.
Article: {text[:2000]}"""
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":stance_prompt}], max_tokens=256)
    raw = resp.choices[0].message.content
    try: stance_info=json.loads(raw)
    except Exception:
        stance_info={"political_leaning":"unknown","implied_stance":"unknown","summary":raw[:200]}
    st.json(stance_info)
    tone_score = bias_to_score(stance_info.get("political_leaning"))
    stance_vec = encode_stance("\n".join([stance_info.get("political_leaning",""), stance_info.get("implied_stance",""), stance_info.get("summary","")]))
    st.session_state.stance_vec = stance_vec
    st.session_state.tone_score = tone_score
    st.success("Stance embedding ready.")

# ------------------------------------------------------------
# 5️⃣ TUNABLES & RETRIEVAL
# ------------------------------------------------------------
if st.session_state.meta and st.session_state.topic_vec is not None:
    st.header("Retrieval & Anti-Echo Analysis")

    cols = st.columns(3)
    with cols[0]:
        w_T_canonical = st.slider("w_T_canonical",0.0,2.0,0.5,0.1)
        w_T_summary   = st.slider("w_T_summary",0.0,20.0,10.0,0.5)
    with cols[1]:
        w_S  = st.slider("w_S",0.0,5.0,1.0,0.1)
        w_B  = st.slider("w_B",0.0,5.0,0.8,0.1)
    with cols[2]:
        w_Tone = st.slider("w_Tone",0.0,5.0,0.3,0.1)
        CANONICAL_TOPIC_THRESHOLD = st.slider("Canonical topic threshold",0.0,1.0,0.3,0.05)
        SUMMARY_SIMILARITY_THRESHOLD = st.slider("Summary similarity threshold",0.0,1.0,0.8,0.05)

    if st.button("Run Anti-Echo Analysis"):
        topic_docs = topic_coll.get(include=["embeddings","metadatas"])
        stance_docs = stance_coll.get(include=["embeddings","metadatas"])
        topic_vec = st.session_state.topic_vec
        summary_vec = topic_vec[768:]
        stance_vec = st.session_state.stance_vec
        meta = st.session_state.meta
        bias_score_article = meta["bias_score"]
        tone_score_article = st.session_state.tone_score

        matches=[]
        for emb,md in zip(topic_docs["embeddings"], topic_docs["metadatas"]):
            emb=np.array(emb)
            if len(emb)==1536:
                summary_similarity = cosine_similarity(summary_vec.reshape(1,-1), emb[768:].reshape(1,-1))[0][0]
            else:
                continue
            if summary_similarity < SUMMARY_SIMILARITY_THRESHOLD: continue
            matches.append({"id":md.get("id"),"source":md.get("source",""),"title":md.get("title",""),
                            "url":md.get("url",""),"bias_family":md.get("bias_family",""),
                            "summary_similarity":summary_similarity})

        scores=[]
        for m in matches:
            bias_db=0.0;tone_db=0.0
            for s_md,s_emb in zip(stance_docs["metadatas"], stance_docs["embeddings"]):
                if s_md.get("id","").split("::")[0]==m["id"].split("::")[0]:
                    bias_db=float(s_md.get("bias_score",0.0)); tone_db=float(s_md.get("tone_score",0.0))
                    stance_sim=cosine_similarity(stance_vec, np.array(s_emb).reshape(1,-1))[0][0]
                    bias_diff=abs(bias_score_article-bias_db)
                    tone_diff=abs(tone_score_article-tone_db)
                    anti_echo=(w_T_summary*m["summary_similarity"])-(w_S*stance_sim)-(w_B*bias_diff)-(w_Tone*tone_diff)
                    scores.append({**m,"bias_score":bias_db,"stance_similarity":stance_sim,
                                   "bias_diff":bias_diff,"tone_diff":tone_diff,"anti_echo_score":anti_echo})
                    break
        df=pd.DataFrame(scores).sort_values("anti_echo_score",ascending=False)
        st.session_state.results=df

if st.session_state.results is not None and not st.session_state.results.empty:
    df=st.session_state.results
    st.subheader("Top Anti-Echo Candidates")
    st.dataframe(df.head(10))
    csv=df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name="anti_echo_results.csv")

    if st.button("🔁 Reset Article"):
        for k in ["meta","topic_vec","stance_vec","results","topic_summary"]: st.session_state.pop(k,None)
        st.experimental_rerun()
