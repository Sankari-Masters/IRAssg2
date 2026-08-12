# Assignment 2 Report — Research Lens

## Use case and collection

Research Lens is a literature and technical-document discovery system for Information Retrieval topics. The supplied reproducible collection has 18 documents across open research notes, engineering blogs, web standards guides, textbooks, developer guides, and data-science journals. It separates document metadata (`data/metadata.csv`) from document bodies and outgoing links (`data/documents.csv`).

The Streamlit **Crawl & Collect** screen additionally accepts multiple approved seed URLs, supports depth 0–2 and a page cap, applies breadth-first collection, checks `robots.txt`, normalises URLs, keeps a visited set, rejects duplicate URLs, and persists accepted document metadata separately from extracted HTML text. **Index Management** accepts a user CSV to demonstrate a heterogeneous public dataset input.

## Text mining and index

The application provides Raw, Balanced and Aggressive preprocessing profiles. Balanced uses lowercase tokenisation and stop-word removal; Aggressive also applies a lightweight suffix reduction. It calculates corpus frequency distributions, document length, vocabulary size, and explainable TF-IDF keywords. The inverted representation stores term counts, document frequencies, TF-IDF vectors and lengths.

Exact duplicates are identified with SHA-256 normalised-content hashes. Near duplicates are displayed when 3-shingle Jaccard similarity is at least 0.28. This makes duplicate handling inspectable before retrieval or evaluation.

## Retrieval and ranking

Search supports BM25, cosine TF-IDF and a hybrid score of 0.82 normalised BM25 + 0.18 normalised PageRank. PageRank is iterated over the available outgoing-link graph. Optional pseudo-relevance feedback contributes up to two carefully limited expansion terms from the initial top three documents. Result cards include title, source, category, score, authority, text snippet and link; the ranking chart exposes the score/authority relationship.

## Recommendation

Content-based recommendations are cosine similarities between TF-IDF profiles. The hybrid option is 0.85 content score plus 0.15 same-category affinity, and displays Top-K items with both similarity and final score. This deliberately avoids inventing user interactions: collaborative filtering should only be enabled after the host system has collected legitimate user-item feedback.

## Evaluation design

`data/qrels.csv` contains graded relevance judgments for seven representative information needs. The Evaluation dashboard compares all three rankers and calculates Precision, Recall, F1, Precision@5, Recall@5, MAP, MRR and NDCG@5. It also offers a per-query table showing the ranked documents alongside their graded relevance. This enables live, reproducible experimental evidence rather than screenshots of a static notebook.

## Inferences

1. High recall with poor early ranking indicates scoring, parameter, query-understanding or authority-signal problems. Tune against qrels, combine complementary signals and track NDCG/MRR.
2. Duplicates waste index capacity, crowd top results, reduce recommendation diversity and can inflate metrics. Canonicalisation, hashes, shingling/MinHash and cluster-aware evaluation mitigate this.
3. Content-based recommendation is transparent and solves new-item cold start; collaborative filtering finds serendipitous community preferences but needs dense, genuine interactions. A hybrid is appropriate after interaction data exists.
4. Collection quality affects features; features affect candidates; indexing enables speed; ranking affects user utility; recommendations broaden discovery; and evaluation measures the total system. Integrated controls make these dependencies observable.
5. The work demonstrates that a strong lexical baseline, measured configuration changes, clean metadata and duplicate control are more reliable than unmeasured complexity.

## Demonstration checklist

Run `streamlit run app.py`, take screenshots of Dashboard, Crawl & Collect, Search, Recommendations, Evaluation and Performance & Inference screens in the virtual lab, and record the required virtual-lab evidence. These screenshots and portal execution cannot be created truthfully without access to the student's lab session.
