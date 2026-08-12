# Research Lens: End-to-End Information Retrieval System

This Streamlit application implements the complete IR lifecycle required by Assignment 2: collection/crawling, duplicate control, metadata-content separation, preprocessing, indexing, ranked search, PageRank, recommendations, and evaluated retrieval.

## Run locally

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed by Streamlit. All workflows are operated from the UI. The included corpus is a reproducible starter collection; users may add approved pages from **Crawl & Collect** or upload a CSV on **Index Management**.

## Files

- `app.py` — the Streamlit application
- `data/metadata.csv` — document metadata, separate from content
- `data/documents.csv` — document text and outgoing links
- `data/qrels.csv` — graded relevance judgments for the evaluation dashboard

## Suggested demonstration flow

1. Open Dashboard and inspect corpus health.
2. On Crawl & Collect, enter approved seeds, select a depth, and collect pages.
3. Rebuild the index in Index Management.
4. Search with BM25, TF-IDF, or Hybrid ranking and inspect PageRank explanations.
5. Select a result in Recommendations to obtain Top-K similar documents.
6. Run the Evaluation dashboard and discuss the comparative metrics and inferences.
