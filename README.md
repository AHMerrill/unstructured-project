# anti echo chamber

Builds an embedding index of news and commentary for studying topic framing and sentiment polarity across sources.  
This repository handles scraping, embedding, batching, and coordination for the Hugging Face dataset [`zanimal/anti-echo-artifacts`](https://huggingface.co/datasets/zanimal/anti-echo-artifacts).

---

## Colab notebooks

### 1. Scraper and batch builder

<a target="_blank" href="https://colab.research.google.com/github/AHMerrill/anti-echo-chamber/blob/main/scraper_artifacts.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

_Run this notebook to scrape, embed, and publish new batches to the Hugging Face dataset._

---

### 2. Analysis and stance comparison

<a target="_blank" href="https://colab.research.google.com/github/AHMerrill/anti-echo-chamber/blob/main/anti_echo_chamber.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

_Run this notebook to rebuild the Chroma index from Hugging Face, upload an article, and find similar topics with opposing viewpoints._

---

# Anti Echo Chamber: Scraper and Embedding Pipeline

This repository implements the complete pipeline used to collect, classify, and embed full-text news articles for the Anti Echo Chamber project.  
The goal is to construct a dataset and retrieval engine that can surface *contrasting arguments* on the same topic by comparing both **topic similarity** and **stance divergence**.

---

## 1. System Overview

The pipeline operates in modular stages defined in `config.yaml`.  
It processes free, factual sources such as *Reuters* and *The Guardian UK* to create dual embedding spaces:  
- one for **topic**, capturing what each article is about  
- one for **stance**, capturing how it argues or frames the issue  

Additionally, the stance module cross-references each article’s tone with **its outlet’s known bias** using `source_bias.json` to determine whether the piece aligns with, opposes, or diverges from its usual ideological framing.

---

## 2. Stage Breakdown and Models Used

### Stage 1. Collection

**Purpose:** Download and structure full articles and metadata from accessible RSS feeds or Selenium scrapes.

**Libraries and Tools**
- `feedparser`, `requests`, `trafilatura`: RSS and HTML parsing
- `BeautifulSoup4`: fallback DOM extraction
- Configurable list of sources, using outlet definitions in `source_bias.json`
- Output fields: `source`, `url`, `title`, `date`, `author`, `section`, `content`

---

### Stage 2. Topic Embedding

**Purpose:** Encode semantic topics for clustering and retrieval.

**Model:** `intfloat/e5-base-v2`  
**Embedding Dimensionality:** 768  
**Pooling:** mean  
**Chunk Tokens:** 512  
**Normalization:** true  
**Collection:** `news_topic`

**Method:**
- Each article is chunked into ≤512 token windows.
- Each chunk is embedded using E5 and averaged to a single topic vector.
- Related topics are mapped using the taxonomy in `topics.json`.
- Topic overlap is detected via cosine similarity ≥ 0.4.

---

### Stage 3. Stance and Ideological Classification (LLM)

**Purpose:** Classify the article’s political leaning, implied stance, and rhetorical framing.

**Provider:** OpenAI  
**Model:** `gpt-4o-mini`  
**Temperature:** 0.4  
**Max Tokens:** 256  
**Mode:** `llm-classification`  

**Inputs:**  
- Cleaned article text  
- Reference ontologies:  
  - `political_leanings.json` (macro ideological families)  
  - `implied_stances.json` (policy-specific positions)  
  - `source_bias.json` (outlet bias metadata)

**LLM Prompt Behavior:**
1. Summarizes the article’s argument in one concise sentence.  
2. Assigns:
   - `political_leaning` (e.g., center left, libertarian, populist right)
   - `implied_stance` (e.g., pro regulation, market environmentalism, austerity)
3. Compares the output to the outlet’s **bias_family** and **bias_score** in `source_bias.json`.

**Tone Alignment Logic**
- If stance and tone are consistent with the outlet’s expected bias, the article is labeled **in-bias**.  
- If the stance significantly diverges (e.g., right-leaning argument in a left-leaning outlet), it is labeled **counter-bias**.  
- If no consistent match exists, it is **neutral or mixed**.

Example output:
```
{
  "political_leaning": "center left",
  "implied_stance": "pro regulation",
  "summary": "Argues that public oversight is necessary to keep markets fair.",
  "tone_alignment": "in-bias"
}
```

The OpenAI response is then combined into a unified text block for embedding.

---

### Stage 4. Stance Embedding (Hybrid)

**Purpose:** Create dense embeddings representing worldview, tone, and rhetorical stance.

**Model:** `all-mpnet-base-v2`  
**Concatenation Order:** `[political_leaning] + [implied_stance] + [summary]`  
**Max Length:** 4096 characters  
**Pooling:** mean  
**Collection:** `news_stance`

This “hybrid text” captures ideological direction and emotional framing in one consistent vector space.

---

### Stage 5. Storage and Dataset Export

**Local Database:** `ChromaDB`  
- Directories: `chroma_db/`  
- Metric: cosine distance  
- Collections:
  - `news_topic`
  - `news_stance`
- Automatic rebuild if missing.

**Batch Artifacts:**
- `embeddings_topic.npz`
- `embeddings_stance.npz`
- `metadata.jsonl`
- `manifest.json`
- Optional compression and push to:
  [Hugging Face Dataset: zanimal/anti-echo-artifacts](https://huggingface.co/datasets/zanimal/anti-echo-artifacts)

---

## 3. RSS Expansion

You can extend coverage by adding new RSS feeds.

```python
FEEDS = [
    "https://www.reuters.com/rssFeed/worldNews",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/politics/rss"
]
```

Each article will automatically flow through cleaning, classification, embedding, and ChromaDB insertion.

Output record format:
```
{
  "source": "guardian",
  "url": "...",
  "title": "...",
  "date": "...",
  "content": "...",
  "topic_vector": [...],
  "stance_vector": [...],
  "tone_alignment": "in-bias"
}
```

---

## 4. Selenium Scrape Ingestion

For pages requiring dynamic rendering, create a DataFrame with:

| column | type | description |
|---------|------|-------------|
| source | str | outlet identifier |
| url | str | canonical link |
| title | str | headline |
| date | str | ISO date |
| content | str | article body |
| author | str (optional) | author |
| section | str (optional) | category |

Then run:
```python
from pipeline import process_dataframe
process_dataframe(df)
```
This triggers classification, embeddings, and tone-bias alignment exactly as the RSS path.

---

## 5. Cross-Bias Comparison Logic

The analysis tool identifies ideological contrast across outlets and within outlets.

**Algorithm Steps:**

1. **Topic Matching**
   - Retrieve top-N articles by cosine similarity in `news_topic` (same subject).

2. **Stance Divergence**
   - Compute cosine distance in `news_stance`.
   - Large distance ⇒ opposite tone or worldview.

3. **Bias Contrast Evaluation**
   - Combine stance and tone alignment metadata.
   - Highlight pairs where:
     - Both discuss the same topic, and  
     - One is *in-bias* and the other *counter-bias* relative to their sources.  

Example:
- Input: Guardian article labeled *center left*, *pro regulation*, *in-bias*
- Retrieval: Wall Street Journal article labeled *center right*, *pro market*, *in-bias*
- Both map to “Economy / Regulation” topic  
→ flagged as a **contrast pair**

**Libraries Used**
- `chromadb` (dense retrieval)
- `numpy` (cosine similarity)
- optional `faiss` for scale

---

## 6. Logging and Diagnostics

- Logging level: `INFO`
- Failure records saved to `logs/`
- Every 100 records checkpointed
- Embedding validation checks:
  - Non-zero vectors
  - JSON integrity
  - Tone alignment computed successfully
  - Sample preview limited to 500 chars for manual inspection

---

## 7. Contributor Notes

- Only scrape open, legal text (Guardian, Reuters, AP, BBC, etc.).
- Do not upload paywalled or copyrighted content.
- Verify that each record includes valid text, stance, and tone alignment before embedding.
- When adding new sources, include their bias metadata in `source_bias.json` so the tone alignment logic remains accurate.

---
