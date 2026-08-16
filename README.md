# Research Lens: End-to-End Information Retrieval System

**Assignment 2 - Information Retrieval (AIMLCZG537/DSECLZG537)(S2-25)**

This Streamlit application implements the complete IR lifecycle: web crawling, text preprocessing, indexing, ranked search, PageRank, document classification, recommendations, and rigorous evaluation.

---

## Table of Contents

1. [Installation](#installation)
2. [Running the Application](#running-the-application)
3. [Dataset Description](#dataset-description)
4. [System Features](#system-features)
5. [File Structure](#file-structure)
6. [Workflow Guide](#workflow-guide)
7. [Evaluation Results](#evaluation-results)
8. [Demo Evidence](#demo-evidence)

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

**Required packages:**
- streamlit >= 1.32
- pandas >= 1.5
- requests >= 2.28
- beautifulsoup4 >= 4.12
- scikit-learn >= 1.3 (for document classification)
- numpy >= 1.24

---

## Running the Application

### Local Execution

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Cloud Deployment

The application is deployed at: **https://researchlens-g42.streamlit.app/**

---

## Dataset Description

### Supplied Reproducible Corpus

**Purpose:** Demonstration of IR concepts with curated, heterogeneous documents

**Composition:**
- **Size:** 18 documents
- **Categories:** 9 distinct categories
  - Information Retrieval
  - Search Engineering
  - Web Crawling
  - Evaluation
  - Recommendation
  - Ranking
  - Text Mining
  - Data Quality
  - Application Development

**Source Types:**
- Open research notes
- Engineering blogs
- Web standards guides
- Open textbooks
- Developer guides
- Data science journals

**Metadata Fields:**
- `doc_id`: Unique document identifier
- `title`: Document title
- `url`: Source URL
- `source`: Publication source
- `author`: Author name
- `published`: Publication date
- `category`: Document category

**Data Separation:**
- `data/metadata.csv` - Document metadata (title, URL, author, source, category, date)
- `data/documents.csv` - Document content and outgoing links
- `data/qrels.csv` - Graded relevance judgments (7 queries, 17 judgments)

### Dataset Provenance

**Initial Corpus:**
The baseline corpus is a **curated reproducible corpus** covering multiple IR-related source types (open research notes, engineering blogs, web standards guides, open textbooks, developer guides, data science journals). Its URLs are represented using `example.org` placeholders so that the collection is reproducible and independent of any live site's availability. The application additionally supports acquisition of genuine heterogeneous web content through the **Crawl & Collect** interface and via public CSV imports, enabling the system to scale to real-world heterogeneous collections.

**Expandability:**
The system supports acquiring heterogeneous data from multiple sources:

1. **Live Web Crawling:**
   - Navigate to "Crawl & Collect" page
   - Enter approved seed URLs (e.g., research blogs, documentation sites)
   - Configure depth (0-2 levels) and page limit
   - System performs BFS crawling with robots.txt compliance
   - Automatic duplicate URL detection and metadata extraction

2. **CSV Import:**
   - Navigate to "Index Management" page
   - Upload CSV with columns: `title`, `content`, `url` (optional), `category` (optional)
   - System performs duplicate detection before adding to corpus
   - Supports public datasets from Kaggle, UCI ML Repository, etc.

3. **API Integration:**
   - Extensible architecture supports API-based collection
   - Can integrate with arXiv, PubMed, Google Scholar APIs
   - Metadata-content separation facilitates API data ingestion

**Example Expansion Workflow:**
```
1. Start with supplied 18-document corpus
2. Crawl approved IR research blogs (depth=1, max=10 pages)
3. Import public IR dataset CSV (e.g., TREC documents)
4. Result: Heterogeneous corpus from multiple sources
```

**Reproducibility:**
- Supplied corpus ensures consistent evaluation across users
- All users start with identical baseline for fair comparison
- Crawled/imported documents are session-specific

---

## System Features

### 1. Dashboard
- Corpus statistics (documents, sources, vocabulary, link edges)
- Collection composition visualization
- Category distribution chart

### 2. Crawl & Collect
- **Breadth-First Search (BFS)** crawling
- **robots.txt compliance** for ethical collection
- **URL normalization** and duplicate detection
- **Configurable depth** (0-2 levels) and page limit
- **Metadata extraction** (title, URL, source, links)
- **Content filtering** (minimum 40 tokens)

### 3. Text Mining
- **3 Preprocessing Profiles:**
  - Raw: Lowercase tokenization only
  - Balanced: + Stopword removal (35 stopwords)
  - Aggressive: + Suffix stemming + min length 3
- **Corpus Statistics:** Token count, vocabulary size, mean document length
- **Document Profiling:** TF-IDF keyword extraction (top-12 keywords per document)
- **Document Classification:** Multinomial Naive Bayes classifier with accuracy, confusion matrix, classification report
- **Preprocessing Comparison:** Impact on vocabulary and retrieval quality
- **Feature Extraction Comparison:** BM25 vs TF-IDF across preprocessing profiles with MAP, MRR, NDCG@5

### 4. Index Management
- **Inverted Index:** Term counts, document frequencies, IDF scores, TF-IDF vectors
- **Duplicate Detection:**
  - Exact duplicates: SHA-256 content hashing
  - Near-duplicates: 3-shingle Jaccard similarity ≥ 0.28
- **Duplicate Removal:** One-click removal with automatic index rebuild
- **CSV Import:** Upload custom document collections
- **Corpus Reset:** Restore supplied reproducible corpus

### 5. Search
- **Ranking Algorithms:**
  - **BM25:** k1=1.5, b=0.75 (term frequency saturation + length normalization)
  - **TF-IDF:** Cosine similarity with normalized vectors
  - **Hybrid:** 0.82 × BM25 + 0.18 × PageRank (min-max normalized)
- **Query Expansion:** Pseudo-relevance feedback (top-3 docs, +2 terms)
- **Result Display:** Title, category, source, score, PageRank, content snippet, URL
- **Ranking Explanation:** "Why this order?" chart showing score vs PageRank

### 6. PageRank
- **Algorithm:** Iterative power method with damping factor 0.85
- **Iterations:** 40 (convergence)
- **Graph Construction:** Outgoing links from documents.csv
- **Dangling Nodes:** Uniform mass distribution
- **Integration:** Blended with BM25 in hybrid ranking

### 7. Recommendations
- **Content-Based:** TF-IDF cosine similarity
- **Hybrid:** 0.85 × content similarity + 0.15 × category affinity
- **Top-K Display:** Configurable K (3-10), similarity scores, recommendation scores
- **Visualization:** Bar chart of recommendation scores

### 8. Evaluation
- **Metrics:** Precision, Recall, F1, P@K, R@K, MAP, MRR, NDCG@K
- **Configurable K:** Slider (3-10) with dynamic metric labels (P@K, R@K, NDCG@K)
- **Comparative Analysis:** BM25 vs TF-IDF vs Hybrid ranking
- **Per-Query Inspection:** Ranked results with graded relevance
- **Visualization:** Bar chart comparing MAP, MRR, NDCG@K

### 9. Performance & Inference
- **Search Latency Analytics:** Median latency, latency by strategy
- **Required Inferences:**
  1. Highly relevant documents retrieved but poorly ranked
  2. Effect of duplicates and mitigation strategies
  3. Content-based vs collaborative recommendation
  4. End-to-end integration benefits
  5. Learnings from the experiment

---

## File Structure

```
IRAssg2-main/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── REPORT.md                   # Comprehensive assignment report with experimental results
├── data/
│   ├── metadata.csv            # Document metadata (18 documents)
│   ├── documents.csv           # Document content and links
│   └── qrels.csv               # Relevance judgments (7 queries, 17 judgments)
└── screenshots/
    ├── SCREENSHOTS.md          # Screenshot instructions
    ├── dashboard.png           # (To be added from Virtual Lab)
    ├── crawl.png
    ├── text_mining.png
    ├── index_management.png
    ├── search_bm25.png
    ├── search_hybrid.png
    ├── recommendations.png
    ├── evaluation.png
    └── performance.png
```

---

## Workflow Guide

### Suggested Demonstration Flow

1. **Dashboard** - Inspect corpus health and composition
   - View 18 documents across 9 categories
   - Check vocabulary size (321 index terms with Balanced preprocessing; index includes title + content)
   - Observe link graph (52 edges)

2. **Crawl & Collect** - (Optional) Add new documents
   - Enter approved seed URLs
   - Set depth=1, max pages=5
   - Observe BFS crawling with robots.txt compliance
   - New documents automatically added to corpus

3. **Text Mining** - Analyze preprocessing and classification
   - Compare Raw, Balanced, Aggressive preprocessing
   - View document TF-IDF keywords
   - Check document classification accuracy (dynamically computed, depends on corpus size)
   - Compare BM25 vs TF-IDF across preprocessing profiles

4. **Index Management** - Control duplicates
   - Check for exact and near-duplicates (0 in supplied corpus)
   - (Optional) Upload CSV to add documents
   - Remove duplicates if detected

5. **Search** - Query the corpus
   - Query: "how should I evaluate ranked retrieval?"
   - Try BM25, TF-IDF, Hybrid ranking
   - Compare results and PageRank scores
   - Enable pseudo-relevance feedback

6. **Recommendations** - Find similar documents
   - Select seed document (e.g., d04 - "Evaluating Search with NDCG and MRR")
   - Choose Hybrid (content + category) method
   - View Top-5 recommendations with similarity scores

7. **Evaluation** - Compare ranking strategies
   - Set K=5
   - View comparative metrics: MAP, MRR, NDCG@5
   - Observe: Hybrid achieves highest MAP (0.618) and NDCG@5 (0.693)
   - Inspect per-query results

8. **Performance & Inference** - Review analytics and learnings
   - Check search latency (measured dynamically)
   - Read required inference discussions
   - Understand system design decisions

---

## Evaluation Results

### Key Findings (K=5, Balanced Preprocessing)

| Strategy | MAP | MRR | NDCG@5 | Precision | Recall |
|----------|-----|-----|--------|-----------|--------|
| BM25 | 0.500 | 1.000 | 0.677 | 0.391 | 0.500 |
| TF-IDF | 0.500 | 1.000 | 0.677 | 0.391 | 0.500 |
| **Hybrid (BM25+PR)** | **0.618** | **1.000** | **0.693** | 0.141 | 1.000 |

**Observations:**
- **Hybrid ranking achieves best MAP (+23.6%) and NDCG@5 (+2.4%)**
- Perfect MRR (1.000) across all strategies - first result always relevant
- Hybrid trades precision for recall (100% recall, 14.1% precision)
- BM25 ≈ TF-IDF on this corpus due to uniform document lengths

### Preprocessing Impact

| Preprocessing | Method | MAP | NDCG@5 |
|---------------|--------|-----|--------|
| Raw | BM25 | 0.500 | 0.677 |
| Raw | TF-IDF | 0.500 | 0.677 |
| Balanced | BM25 | 0.500 | 0.677 |
| Balanced | TF-IDF | 0.500 | 0.677 |
| **Aggressive** | **BM25** | **0.548** | **0.711** |
| **Aggressive** | **TF-IDF** | **0.548** | **0.711** |

**Finding:** Aggressive preprocessing achieved the best MAP and NDCG@5 on this small corpus. This result is corpus-dependent; preprocessing should be selected using relevance-based evaluation rather than assumed to be universally optimal.

---

## Demo Evidence

### Screenshots

The final submission will include screenshots captured from the BITS Virtual Lab in the `screenshots/` directory and referenced in REPORT.md.

**Required screenshots (from BITS Virtual Lab):**
1. dashboard.png - Corpus statistics and composition
2. crawl.png - Web crawling interface
3. text_mining.png - Preprocessing, classification, feature comparison
4. index_management.png - Duplicate detection and removal
5. search_bm25.png - BM25 ranking results
6. search_hybrid.png - Hybrid ranking with PageRank and ranking explanation
7. recommendations.png - Top-K recommendations
8. evaluation.png - Comparative metrics
9. performance.png - Latency analytics and inferences

---

## Technical Details

### Algorithms Implemented

1. **BM25 Ranking:**
   ```
   score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))
   ```

2. **TF-IDF Cosine Similarity:**
   ```
   score(D, Q) = (D · Q) / (||D|| × ||Q||)
   ```

3. **PageRank:**
   ```
   PR(p) = (1 - d) / N + d × Σ (PR(q) / L(q))
   ```
   - Damping factor: d = 0.85
   - Iterations: 40

4. **Hybrid Ranking:**
   ```
   score(D, Q) = 0.82 × normalize(BM25(D, Q)) + 0.18 × normalize(PageRank(D))
   ```

5. **Document Classification:**
   - Algorithm: Multinomial Naive Bayes
   - Features: TF-IDF weighted term vectors
   - Smoothing: α = 0.1 (Laplace)

### Performance

- **Search Latency:** Measured dynamically by the application; depends on machine, Streamlit environment, corpus size, and query complexity
- **Index Build Time:** The supplied 18-document corpus is small enough for interactive index construction; formal latency is environment-dependent
- **Classification Accuracy:** Dynamically reported by the Text Mining dashboard; depends on corpus size and balance
- **Scalability:** O(n²) near-duplicate detection suitable for corpora up to ~1,000 documents

---

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'sklearn'**
   ```bash
   pip install scikit-learn
   ```

2. **Crawling Error: No module named 'requests'**
   ```bash
   pip install requests beautifulsoup4
   ```

3. **Streamlit Not Found**
   ```bash
   pip install streamlit
   ```

4. **Data Files Not Found**
   - Ensure you're running `streamlit run app.py` from the project root directory
   - Verify `data/` folder exists with metadata.csv, documents.csv, qrels.csv

---
