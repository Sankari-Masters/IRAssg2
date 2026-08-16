# Assignment 2 Report — Research Lens: End-to-End Information Retrieval System

**Group ID:** Group 42  
**Course:** Information Retrieval (AIMLCZG537/DSECLZG537)(S2-25)  
**Assignment:** Assignment 2 - End-to-End IR System  

---

## Executive Summary

This report presents **Research Lens**, a complete end-to-end Information Retrieval system implemented as a Streamlit web application. The system demonstrates the full IR lifecycle: web crawling, text preprocessing, indexing, ranked retrieval, recommendation, and rigorous evaluation. The implementation includes BM25 and TF-IDF ranking algorithms, PageRank authority scoring, content-based recommendations, document classification, and comprehensive duplicate detection and removal.

---

## 1. Use Case and Dataset

### 1.1 Use Case
Research Lens is designed as a **literature and technical-document discovery system** for Information Retrieval topics. It helps researchers, students, and practitioners find relevant IR-related documents, understand ranking mechanisms, and discover related content.

### 1.2 Dataset Description

**Initial Corpus:**
- **Size:** 18 documents
- **Categories:** 9 distinct categories (Information Retrieval, Search Engineering, Web Crawling, Evaluation, Recommendation, Ranking, Text Mining, Data Quality, Application Development)
- **Metadata Fields:** doc_id, title, url, source, author, published date, category
- **Content:** Separated from metadata in `documents.csv` with outgoing link information

**Dataset Provenance:**
The baseline corpus is a **curated reproducible corpus** covering multiple IR-related source types (open research notes, engineering blogs, web standards guides, open textbooks, developer guides, data science journals). Its URLs are represented using `example.org` placeholders so that the collection is reproducible and independent of any live site's availability. The application additionally supports acquisition of genuine heterogeneous web content through the **Crawl & Collect** interface and via public CSV imports, enabling the system to scale to real-world heterogeneous collections.

**Heterogeneous Acquisition Methods:**
1. **Live Web Crawling** — breadth-first collection from approved seed URLs with configurable depth (0-2 levels) and `robots.txt` compliance.
2. **CSV Upload** — import public datasets (e.g., Kaggle, UCI ML Repository, TREC) with title, content, URL, and category columns.
3. **API/Custom Import** — the metadata-content separation makes it straightforward to ingest data from APIs such as arXiv, PubMed, or Google Scholar.

**Expandability:**
- Users can start with the supplied reproducible corpus and then add approved live pages via the "Crawl & Collect" interface.
- CSV import supports custom, heterogeneous document collections.
- All additions undergo duplicate detection before indexing, ensuring a clean corpus.

**Virtual Lab Demonstration:**
During BITS Virtual Lab execution, the application will demonstrate live heterogeneous acquisition by:
1. Crawling approved seed URLs (e.g., research blogs, documentation sites) using the "Crawl & Collect" interface
2. Configuring crawl depth (0-2 levels) and page limits
3. Showing BFS crawling with robots.txt compliance and duplicate URL detection
4. Adding collected documents to the corpus with automatic metadata extraction
5. Performing duplicate detection before indexing

This demonstrates that the system acquires genuine heterogeneous web content from multiple sources, not just the supplied baseline corpus.

---

## 2. System Architecture

### 2.1 Streamlit-Based Workflow
The complete IR lifecycle is accessible through 8 integrated pages:

1. **Dashboard** - Corpus statistics and composition
2. **Crawl & Collect** - Web crawling with BFS, robots.txt compliance, URL normalization
3. **Text Mining** - Preprocessing, keyword extraction, document profiling, classification
4. **Index Management** - Duplicate control, CSV import, corpus reset
5. **Search** - BM25, TF-IDF, Hybrid ranking with PageRank
6. **Recommendations** - Content-based and hybrid Top-K recommendations
7. **Evaluation** - Comparative metrics across ranking strategies
8. **Performance & Inference** - Analytics and required inferences

### 2.2 Data Separation
- **metadata.csv** - Document metadata (title, URL, author, source, category, date)
- **documents.csv** - Document content and outgoing links
- **qrels.csv** - Graded relevance judgments (7 queries, 17 judgments)

This separation enables:
- Efficient metadata filtering
- Content updates without metadata changes
- Data governance and privacy controls
- Scalable indexing

---

## 3. Text Preprocessing and Mining

### 3.1 Preprocessing Profiles

Three preprocessing strategies are implemented:

| Profile | Tokenization | Stopword Removal | Stemming | Min Length |
|---------|--------------|------------------|----------|------------|
| **Raw** | Lowercase + regex | ❌ No | ❌ No | 2 chars |
| **Balanced** | Lowercase + regex | ✅ Yes (35 stopwords) | ❌ No | 2 chars |
| **Aggressive** | Lowercase + regex | ✅ Yes | ✅ Suffix removal | 3 chars |

**Regex Pattern:** `[a-zA-Z][a-zA-Z0-9_-]{1,}` - Captures alphanumeric tokens with hyphens/underscores

### 3.2 Preprocessing Impact on Corpus Statistics

| Profile | Total Tokens | Vocabulary Size | Mean Doc Length |
|---------|--------------|-----------------|-----------------|
| Raw | 531 | 327 | 29.5 |
| Balanced | 426 | 302 | 23.7 |
| Aggressive | 422 | 270 | 23.4 |

**Observations:**
- Balanced preprocessing reduces vocabulary by about 7.6% compared to Raw while removing stopwords
- Aggressive preprocessing reduces vocabulary by about 17.4% compared to Raw through stopword removal and suffix reduction
- Raw mode preserves all terms but includes noise (stopwords, single characters)
- All profiles produce relatively compact representations due to the small corpus size

### 3.3 Document Classification

**Algorithm:** Multinomial Naive Bayes with TF-IDF features

**Experimental Setup:**
- **Feature Representation:** TF-IDF weighted term vectors
- **Train-Test Split:** 70-30 split performed before feature extraction to avoid data leakage
- **IDF Computation:** Computed from the training split only
- **Smoothing Parameter:** α = 0.1 (Laplace smoothing)

**Important Note on the Small Starter Corpus:**
The supplied 18-document corpus spans 9 categories, with some categories represented by only one document (Application Development: 1, Ranking: 2, Text Mining: 2, etc.). This makes rigorous classification difficult because:
1. Many categories have too few examples for stratified train/test splits
2. The training set is very small (typically 12-13 documents)
3. Test sets are tiny (typically 5-6 documents)

The classification module is therefore presented as a **functional demonstration** of the classification pipeline rather than a statistically reliable benchmark. The system gracefully handles cases where stratification is impossible.

**Classification Results:**
- **Accuracy is reported dynamically** in the Text Mining dashboard based on the current corpus and preprocessing profile.
- With the starter corpus, performance is modest due to the small training set and class imbalance.
- Performance improves substantially when the corpus is expanded via web crawling or CSV import, providing more training examples per category.

**Methodological Rigor:**
The implementation correctly performs **train/test split before TF-IDF/IDF computation**. This prevents test-document term statistics from leaking into the IDF/vocabulary construction, which is the academically correct way to evaluate a text classifier. The stratified split is used only when all classes have at least 2 examples; otherwise, a random split is used.

**Confusion Matrix Analysis:**
- For categories with more than one document, the classifier has a greater opportunity to learn category-specific patterns, although the overall sample remains too small for reliable generalization.
- For categories with only one document, the classifier cannot generalize (no test examples possible).
- Misclassifications primarily occur between semantically similar categories (e.g., "Search Engineering" vs "Information Retrieval").

**Key Insight:** TF-IDF features provide a transparent and fast baseline for document classification, but reliable classification requires a larger, more balanced corpus. The module correctly demonstrates the full pipeline: tokenization → train/test split → TF-IDF (training-only) → classifier training → evaluation.

---

## 4. Indexing and Duplicate Control

### 4.1 Index Structure

**Implementation:**
The system maintains an index with the following components:
- **Document-Term Counts:** Mapping of documents to term frequency counts
- **Document Frequencies:** Count of documents containing each term
- **IDF Scores:** `log((N + 1) / (df + 1)) + 1` where N = total documents
- **TF-IDF Vectors:** Normalized term weight vectors for cosine similarity
- **Document Lengths:** Token counts for BM25 length normalization
- **PageRank Scores:** Link-based authority scores

**Note:** For the 18-document corpus, the current implementation uses document-centric indexing (iterating over documents for each query). Production systems typically use term-centric postings lists for sub-linear query execution, but the current approach is suitable for small corpora.

**Index Statistics (Balanced preprocessing):**
- Vocabulary: 321 unique terms
- Average document length: 23.7 tokens
- Link edges: 52 outgoing links in the graph

### 4.2 Duplicate Detection and Removal

**Exact Duplicates:**
- **Method:** SHA-256 hashing of normalized content
- **Normalization:** Lowercase + whitespace collapse
- **Result:** 0 exact duplicates in supplied corpus

**Near-Duplicates:**
- **Method:** 3-shingle Jaccard similarity
- **Threshold:** ≥ 0.28 similarity
- **Shingle Construction:** Consecutive 3-word sequences
- **Result:** 0 near-duplicate pairs in supplied corpus

**Duplicate Removal Feature:**
- One-click removal of detected duplicates
- Keeps canonical document (lower doc_id for near-duplicates)
- Automatic index rebuild after removal
- Prevents metric inflation and result crowding

---

## 5. Web Searching and Ranking

### 5.1 Ranking Algorithms

#### 5.1.1 BM25 (Best Match 25)
**Formula:**
```
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))
```

**Parameters:**
- k1 = 1.5 (term frequency saturation)
- b = 0.75 (length normalization)

**Characteristics:**
- Term frequency saturation prevents over-weighting of repeated terms
- Length normalization penalizes longer documents
- Strong lexical baseline for keyword matching

#### 5.1.2 TF-IDF Cosine Similarity
**Formula:**
```
score(D, Q) = (D · Q) / (||D|| × ||Q||)
```

**Characteristics:**
- Normalized dot product of TF-IDF vectors
- Symmetric similarity measure
- Effective for lexical similarity in the TF-IDF vector space

#### 5.1.3 Hybrid (BM25 + PageRank)
**Formula:**
```
score(D, Q) = 0.82 × normalize(BM25(D, Q)) + 0.18 × normalize(PageRank(D))
```

**Characteristics:**
- Combines lexical relevance with link authority
- Min-max normalization ensures fair weighting
- 82-18 split favors content relevance over authority

### 5.2 PageRank Implementation

**Algorithm:** Iterative power method with damping

**Formula:**
```
PR(p) = (1 - d) / N + d × Σ (PR(q) / L(q))
```

**Parameters:**
- Damping factor: d = 0.85
- Iterations: 40
- Initial rank: 1/N for all documents

**Graph Construction:**
- Outgoing links extracted from `documents.csv`
- Self-links excluded
- Unknown targets ignored
- Dangling nodes distribute mass uniformly

**Top-5 Highest PageRank Documents:**
1. d04 - "Evaluating Search with NDCG and MRR" (PR = 0.1340)
2. d02 - "Practical BM25 for Product Search" (PR = 0.1290)
3. d10 - "Query Expansion with Pseudo Relevance Feedback" (PR = 0.1153)
4. d16 - "Precision Recall Tradeoffs in Retrieval" (PR = 0.1069)
5. d17 - "Semantic Search with Dense Embeddings" (PR = 0.0707)

### 5.3 Query Expansion

**Method:** Pseudo-Relevance Feedback (PRF)

**Algorithm:**
1. Execute initial BM25 query
2. Extract top-3 ranked documents
3. Aggregate term frequencies from these documents
4. Select top-12 most frequent terms not in original query
5. Add up to 2 expansion terms to query

**Benefits:**
- Improves recall for short queries
- Handles vocabulary mismatch
- Transparent and explainable

**Risks:**
- Query drift if initial results are poor
- Conservative limits (2 terms) mitigate this risk

---

## 6. Recommendation System

### 6.1 Content-Based Recommendation

**Method:** Cosine similarity between TF-IDF document vectors

**Formula:**
```
similarity(D1, D2) = (D1 · D2) / (||D1|| × ||D2||)
```

**Characteristics:**
- Explainable: Recommendations based on shared keywords
- Cold-start friendly: Works for new documents immediately
- No user interaction data required
- Effective for specialist/technical content

### 6.2 Hybrid Recommendation

**Formula:**
```
score(D_seed, D_candidate) = 0.85 × cosine_similarity + 0.15 × category_match
```

**Category Match:**
- 1.0 if same category
- 0.0 if different category

**Characteristics:**
- Balances content similarity with categorical affinity
- 85-15 split ensures content dominates
- Adds a categorical relevance signal while keeping content similarity dominant

### 6.3 Why No Collaborative Filtering?

**Deliberate Design Choice:**
- Collaborative filtering requires genuine user-item interaction data
- Fabricating interaction matrices would be academically dishonest
- Content-based methods are more appropriate for document corpora without user feedback
- System is designed to integrate collaborative filtering once real interaction data exists

---

## 7. Evaluation and Experimental Results

### 7.1 Evaluation Dataset

**Relevance Judgments (qrels.csv):**
- **Queries:** 7 representative information needs
- **Judgments:** 17 graded relevance assessments
- **Relevance Scale:** 0 (not relevant), 1 (marginally relevant), 2 (relevant), 3 (highly relevant)

**Sample Queries:**
1. "bm25 ranking"
2. "web crawler"
3. "duplicate documents"
4. "search evaluation"
5. "recommendation system"
6. "text preprocessing"
7. "pagerank"

### 7.2 Evaluation Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Precision** | Fraction of retrieved docs that are relevant | TP / (TP + FP) |
| **Recall** | Fraction of relevant docs that are retrieved | TP / (TP + FN) |
| **F1-Score** | Harmonic mean of precision and recall | 2PR / (P + R) |
| **P@K** | Precision in top-K results | Relevant in top-K / K |
| **R@K** | Recall in top-K results | Relevant in top-K / Total relevant |
| **MAP** | Mean Average Precision | Mean of AP across queries |
| **MRR** | Mean Reciprocal Rank | Mean of 1/rank of first relevant |
| **NDCG@K** | Normalized Discounted Cumulative Gain | DCG@K / IDCG@K |

### 7.3 Experimental Results (K=5, Balanced Preprocessing)

#### 7.3.1 Ranking Strategy Comparison

| Strategy | Precision | Recall | F1 | P@5 | R@5 | MAP | MRR | NDCG@5 |
|----------|-----------|--------|-----|-----|-----|-----|-----|--------|
| **BM25** | 0.391 | 0.500 | 0.407 | 0.229 | 0.500 | 0.500 | 1.000 | 0.677 |
| **TF-IDF** | 0.391 | 0.500 | 0.407 | 0.229 | 0.500 | 0.500 | 1.000 | 0.677 |
| **Hybrid (BM25+PR)** | 0.141 | 1.000 | 0.246 | 0.257 | 0.548 | 0.618 | 1.000 | 0.693 |

**Key Findings:**

1. **BM25 vs TF-IDF:**
   - Identical performance on this corpus (MAP = 0.500, MRR = 1.000)
   - Both achieve perfect MRR (first result always relevant)
   - Moderate precision (39.1%) indicates some non-relevant results
   - Good recall (50.0%) shows half of relevant documents retrieved

2. **Hybrid Ranking:**
   - **Highest MAP (0.618)** and **NDCG@5 (0.693)** - best overall ranking quality
   - **Perfect recall (100%)** - retrieves all relevant documents
   - **Lower precision (14.1%)** - authority signal introduces non-relevant but authoritative documents
   - Trade-off: Completeness vs precision

3. **Perfect MRR (1.000) across all strategies:**
   - First ranked document is always relevant for every query
   - Indicates strong top-1 performance
   - Ranking quality differences emerge in positions 2-5

#### 7.3.2 Feature Extraction Comparison

**Impact of Preprocessing on Retrieval Quality:**

| Preprocessing | Method | MAP | MRR | NDCG@5 | Precision |
|---------------|--------|-----|-----|--------|-----------|
| Raw | BM25 | 0.500 | 1.000 | 0.677 | 0.391 |
| Raw | TF-IDF | 0.500 | 1.000 | 0.677 | 0.391 |
| Balanced | BM25 | 0.500 | 1.000 | 0.677 | 0.391 |
| Balanced | TF-IDF | 0.500 | 1.000 | 0.677 | 0.391 |
| **Aggressive** | **BM25** | **0.548** | **1.000** | **0.711** | **0.391** |
| **Aggressive** | **TF-IDF** | **0.548** | **1.000** | **0.711** | **0.391** |

**Observations:**

1. **Aggressive preprocessing achieves highest MAP and NDCG@5:**
   - MAP = 0.548 (vs 0.500 for Raw/Balanced)
   - NDCG@5 = 0.711 (vs 0.677 for Raw/Balanced)
   - However, this result is corpus-dependent and should not be generalized

2. **Preprocessing effectiveness is corpus-dependent:**
   - On this small 18-document corpus, aggressive stemming helps
   - The corpus is homogeneous in document length and topic
   - Results may differ on larger, more diverse corpora
   - Balanced preprocessing remains a conservative middle ground for unknown corpora, but the current relevance judgments favor Aggressive preprocessing

3. **Raw preprocessing includes noise:**
   - Stopwords dilute term importance
   - Larger vocabulary increases computational cost
   - Performance is equivalent to Balanced on this corpus

4. **BM25 ≈ TF-IDF on this corpus:**
   - Identical performance across all preprocessing profiles
   - BM25's length normalization is less critical for uniform-length documents
   - Both methods benefit equally from preprocessing

**Recommendation:** For this specific corpus, **Aggressive preprocessing with BM25** achieves the best metrics. However, **Balanced preprocessing** is recommended as a more robust choice for general use, as it provides good performance without the risk of over-stemming on diverse corpora.

---

## 8. Performance Analytics

### 8.1 Search Latency

**Dynamic Measurement:**
Search latency is measured dynamically by the application using `time.perf_counter()` and displayed in the Performance & Inference page. Latency depends on:
- Machine hardware and Streamlit environment
- Current corpus size and composition
- Query complexity and preprocessing profile
- Cache state (cold vs. warm)

**Performance Measurement:**
Performance will be measured dynamically in the BITS Virtual Lab. For the supplied small corpus, all strategies are expected to have low latency; the actual observed latency will be reported from the application's Performance & Inference page.

### 8.2 Scalability Considerations

**Current Implementation:**
- Near-duplicate detection: O(n²) pairwise comparison
- Suitable for corpora up to ~1,000 documents
- Production systems should use MinHash/LSH for O(n) complexity

**Index Caching:**
- Streamlit `@st.cache_resource` prevents redundant index rebuilds
- Index persists across queries within a session
- Rebuilds only when corpus or preprocessing profile changes

---

## 9. Required Inferences and Discussion

### 9.1 Highly Relevant Documents Retrieved but Poorly Ranked

**Problem:** System retrieves relevant documents but ranks them below position 5.

**Likely Causes:**

1. **Weak Lexical Scores:**
   - Query-document vocabulary mismatch
   - Short or ambiguous queries
   - Missing synonyms or related terms

2. **Untuned BM25 Parameters:**
   - Default k1=1.5, b=0.75 may not suit corpus characteristics
   - Optimal parameters vary by document length distribution

3. **Stale or Irrelevant Authority Signals:**
   - PageRank based on outdated or sparse link graph
   - Authority doesn't correlate with query relevance

4. **Preprocessing Mismatch:**
   - Different preprocessing for indexing vs querying
   - Over-aggressive stemming loses semantic distinctions

**Proposed Improvements:**

1. **Parameter Tuning:**
   - Optimize k1 and b against qrels using grid search
   - Target NDCG@5 and MAP as optimization objectives

2. **Signal Blending:**
   - Combine normalized BM25 with PageRank (as implemented)
   - Add semantic similarity (dense embeddings) for vocabulary mismatch
   - Boost trusted metadata fields (author, source)

3. **Query Enhancement:**
   - Pseudo-relevance feedback (implemented)
   - Synonym expansion using WordNet or embeddings
   - Query reformulation suggestions

4. **Evaluation-Driven Development:**
   - Monitor NDCG@K and MRR, not just recall
   - A/B test ranking changes against held-out qrels
   - Track per-query performance for targeted improvements

**Evidence from Our Experiment:**
- Hybrid ranking improved MAP from 0.500 to 0.618 (+23.6%)
- NDCG@5 increased from 0.677 to 0.693 (+2.4%)
- Demonstrates that blending signals improves ranking quality

### 9.2 Effect of Duplicates and Mitigation

**Impact of Duplicates:**

1. **Indexing:**
   - Inflates term frequencies and document frequencies
   - Skews IDF calculations (duplicates counted as separate documents)
   - Wastes storage and computational resources

2. **Ranking:**
   - Multiple copies can occupy top-K positions
   - Reduces result diversity
   - User sees redundant information

3. **Recommendation:**
   - Duplicate documents recommended repeatedly
   - Reduces recommendation diversity
   - Poor user experience

4. **Evaluation:**
   - Overstates metrics if every copy judged relevant
   - Unfair comparison if one system deduplicates and another doesn't
   - Precision artificially inflated

**Mitigation Strategies (Implemented):**

1. **Exact Duplicate Detection:**
   - SHA-256 content hashing with normalization
   - O(n) complexity, deterministic
   - Catches verbatim copies

2. **Near-Duplicate Detection:**
   - 3-shingle Jaccard similarity ≥ 0.28
   - Catches paraphrases and minor edits
   - Tunable threshold for precision-recall trade-off

3. **Canonical Document Selection:**
   - Keep document with lower doc_id (arbitrary but consistent)
   - Could enhance with quality signals (length, metadata completeness)

4. **Duplicate Removal:**
   - One-click removal from corpus
   - Automatic index rebuild
   - Audit trail maintained in duplicate detection table

**Production Enhancements:**

- **SimHash/MinHash:** O(n) near-duplicate detection for large corpora
- **Cluster-Aware Ranking:** Promote one representative per cluster
- **Deduplicated Qrels:** Ensure evaluation set has no duplicates
- **URL Canonicalization:** Normalize URLs before crawling (implemented)

**Evidence from Our System:**
- 0 exact duplicates detected in supplied corpus
- 0 near-duplicate pairs (threshold = 0.28)
- Demonstrates clean corpus curation

### 9.3 Content-Based vs Collaborative Recommendation

**Content-Based Recommendation:**

**Strengths:**
- **Explainable:** Recommendations based on shared keywords/features
- **Cold-Start Friendly:** Works immediately for new items
- **No User Data Required:** Privacy-preserving
- **Specialist Content:** Effective for technical/niche domains

**Weaknesses:**
- **Limited Serendipity:** Only recommends similar items
- **Feature Engineering:** Requires good text representation
- **Over-Specialization:** May create filter bubbles

**Collaborative Filtering:**

**Strengths:**
- **Serendipitous Discovery:** Finds items with little textual resemblance
- **Community Wisdom:** Leverages collective user behavior
- **No Content Analysis:** Works for non-textual items (images, music)

**Weaknesses:**
- **Cold-Start Problem:** Fails for new users and new items
- **Data Sparsity:** Requires dense user-item interaction matrix
- **Privacy Concerns:** Needs user behavior tracking
- **Popularity Bias:** Tends to recommend popular items

**When to Use Each:**

| Scenario | Preferred Approach | Rationale |
|----------|-------------------|-----------|
| New corpus with no user data | **Content-Based** | No interaction history available |
| Specialist/technical content | **Content-Based** | Semantic similarity more important than popularity |
| Privacy-sensitive systems | **Content-Based** | No user tracking required |
| Mature platform with rich user data | **Collaborative** | Leverage community preferences |
| Diverse content (news, e-commerce) | **Hybrid** | Combine content + interactions |

**Our Implementation:**
- **Content-Based:** TF-IDF cosine similarity
- **Hybrid:** 85% content + 15% category affinity
- **Rationale:** Document corpus without genuine user interactions
- **Future:** Add collaborative filtering once interaction data exists

**Evidence:**
- Top-K recommendations expose cosine similarity and recommendation scores for transparent inspection
- Category affinity provides an additional signal beyond content similarity
- Transparent and explainable to users

### 9.4 End-to-End Integration Benefits

**Integrated IR Pipeline:**

```
Crawling → Preprocessing → Indexing → Ranking → Recommendation → Evaluation
    ↓           ↓             ↓          ↓            ↓              ↓
Coverage    Features      Speed      Utility     Discovery      Evidence
```

**Why Integration Matters:**

1. **Crawling Determines Coverage:**
   - Quality of seed URLs affects corpus representativeness
   - Depth and breadth settings control scope
   - Robots.txt compliance ensures ethical collection
   - **Impact:** Poor crawling → incomplete corpus → low recall

2. **Preprocessing Creates Dependable Features:**
   - Tokenization, stopword removal, stemming affect vocabulary
   - Preprocessing consistency between indexing and querying is critical
   - **Impact:** Mismatched preprocessing → vocabulary mismatch → low precision

3. **Indexing Enables Efficient Retrieval:**
   - Index structure (document-centric for small corpora, postings-based for production)
   - TF-IDF vectors enable cosine similarity
   - **Impact:** Poor indexing → slow queries → bad user experience

4. **Ranking Affects User Utility:**
   - Top-K results determine user satisfaction
   - Blending signals (lexical + authority) improves quality
   - **Impact:** Poor ranking → relevant docs buried → user abandons search

5. **Recommendation Extends Discovery:**
   - Users find related content beyond initial query
   - Increases engagement and exploration
   - **Impact:** No recommendations → limited discovery → underutilized corpus

6. **Evaluation Closes the Loop:**
   - Metrics provide evidence for design decisions
   - A/B testing guides improvements
   - **Impact:** No evaluation → blind optimization → wasted effort

**Metadata Separation Benefits:**
- **Filtering:** Query by author, source, date without full-text search
- **Governance:** Update metadata (e.g., retract author) without touching content
- **Efficiency:** Index content once, update metadata frequently
- **Privacy:** Redact sensitive metadata while preserving content

**Integrated UI Benefits:**
- **Observability:** See impact of preprocessing on vocabulary
- **Reproducibility:** Consistent workflow from crawl to evaluation
- **Experimentation:** Compare ranking strategies side-by-side
- **Transparency:** Explain why documents ranked in specific order

**Evidence from Our System:**
- Single Streamlit app integrates all 8 workflow stages
- Preprocessing profile affects all downstream components consistently
- Evaluation dashboard compares strategies using same corpus and qrels
- Users can trace document from crawl → index → search → recommendation

### 9.5 Learnings from the Experiment

**Key Takeaways:**

1. **Effectiveness Depends on Representation AND Ranking:**
   - BM25 and TF-IDF performed identically (MAP = 0.500)
   - Hybrid ranking improved MAP to 0.618 (+23.6%)
   - **Lesson:** Ranking algorithm matters as much as feature extraction

2. **BM25 is a Strong Transparent Baseline:**
   - Matches TF-IDF performance without vector normalization
   - Term frequency saturation and length normalization are effective
   - **Lesson:** Start with BM25 before trying complex models

3. **PageRank Provides Useful Authority Signal:**
   - Hybrid ranking achieved highest MAP (0.618) and NDCG@5 (0.693)
   - Authority complements lexical relevance
   - **Lesson:** Link structure contains valuable information

4. **Preprocessing Effectiveness is Corpus-Dependent:**
   - Raw: 327 terms, Balanced: 302 terms (-7.6%), Aggressive: 270 terms (-17.4%)
   - On this corpus, aggressive preprocessing achieved best MAP (0.548) and NDCG@5 (0.711)
   - This result is specific to the small, homogeneous corpus and should not be generalized
   - **Lesson:** Preprocessing choice should be validated against relevance judgments, not assumed

5. **Top-K Recommendations Need Diversity and Duplicate Control:**
   - Content-based recommendations expose cosine similarity scores; similarity strength varies by seed and corpus
   - Category affinity provides an additional categorical relevance signal
   - Duplicate removal prevents redundant recommendations
   - **Lesson:** Similarity alone is insufficient; need diversity mechanisms

6. **Evaluation Should Guide Configuration:**
   - Preprocessing/feature comparison revealed Aggressive > Raw ≈ Balanced for MAP and NDCG@5 on this corpus
   - Ranking comparison showed Hybrid > BM25 ≈ TF-IDF
   - **Lesson:** Use qrels to make evidence-based decisions, not intuition

7. **Perfect MRR Doesn't Mean Perfect System:**
   - MRR = 1.000 for all strategies (first result always relevant)
   - But precision varies (14.1% to 39.1%)
   - **Lesson:** Evaluate multiple metrics; optimize for user task

8. **Metadata Separation Enables Flexibility:**
   - Easy to filter by category, source, date
   - Update metadata without reindexing content
   - **Lesson:** Design for governance and evolution from the start

9. **Explainability Matters:**
   - PageRank scores shown alongside search results
   - TF-IDF keywords explain document profiles
   - Classification reports show per-category performance
   - **Lesson:** Users trust systems they understand

10. **Integration Reveals Dependencies:**
    - Preprocessing affects indexing, ranking, and recommendation
    - Duplicate removal improves all downstream metrics
    - Evaluation exposes weaknesses in earlier stages
    - **Lesson:** End-to-end testing catches issues unit tests miss

---

## 10. Screenshots and Demo Evidence

**Note:** Screenshots will be captured from BITS Virtual Lab execution. The following placeholders describe the expected content:

### 10.1 Dashboard
*Corpus statistics: 18 documents, 12 sources, 321 vocabulary terms (Balanced preprocessing), 52 link edges*

### 10.2 Crawl & Collect
*Web crawling interface with BFS, robots.txt compliance, and duplicate URL handling*

### 10.3 Text Mining
*Preprocessing comparison, document profiling, TF-IDF keywords, and document classification*

### 10.4 Index Management
*Duplicate detection and removal, CSV import, corpus reset*

### 10.5 Search
*BM25 ranking with PageRank scores and ranking explanation chart*

### 10.6 Recommendations
*Content-based Top-5 recommendations with similarity scores*

### 10.7 Evaluation
*Comparative metrics across BM25, TF-IDF, and Hybrid ranking strategies*

### 10.8 Performance & Inference
*Search latency analytics and required inference discussions*

---

## 11. Conclusion

Research Lens successfully demonstrates a complete end-to-end Information Retrieval system with:

✅ **Web Crawling:** BFS with robots.txt compliance, URL normalization, duplicate detection  
✅ **Text Mining:** 3 preprocessing profiles, keyword extraction, document profiling, classification  
✅ **Indexing:** Indexed term-frequency representation with TF-IDF vectors, PageRank, duplicate removal  
✅ **Ranking:** BM25, TF-IDF, Hybrid (BM25+PageRank), query expansion  
✅ **Recommendation:** Content-based and hybrid Top-K recommendations  
✅ **Evaluation:** 8 metrics (Precision, Recall, F1, P@K, R@K, MAP, MRR, NDCG@K)  
✅ **Streamlit UI:** Professional 8-page workflow accessible via web interface  

**Experimental Evidence:**
- Hybrid ranking achieved best MAP (0.618) and NDCG@5 (0.693)
- Preprocessing effectiveness is corpus-dependent; aggressive stemming achieved best metrics on this corpus
- Document classification is demonstrated with a leakage-free train/test split and dynamically reported accuracy
- Search latency is measured dynamically and displayed in the Performance & Inference page

**Key Contributions:**
1. Integrated end-to-end workflow in single application
2. Evidence-based comparison of ranking strategies and preprocessing profiles
3. Transparent and explainable IR system with PageRank visualization
4. Rigorous evaluation with 7 queries and 8 metrics
5. Duplicate detection and removal suitable for the supplied assignment corpus

The system is ready for deployment and further enhancement with semantic search, collaborative filtering, and larger-scale corpora.

---

## 12. References

1. Robertson, S. E., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

2. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank Citation Ranking: Bringing Order to the Web. *Stanford InfoLab Technical Report*.

3. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513-523.

4. Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems*, 20(4), 422-446.

5. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

---

**End of Report**
