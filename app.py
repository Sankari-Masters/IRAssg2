"""Research Lens — a Streamlit end-to-end Information Retrieval demonstration."""
from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, defaultdict, deque
from datetime import date
from io import StringIO
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import streamlit as st

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:  # surfaced in the UI if crawling is selected before installation
    requests = None
    BeautifulSoup = None

try:
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np
    sklearn_available = True
except ImportError:
    sklearn_available = False


st.set_page_config(page_title="Research Lens | IR System", page_icon="🔎", layout="wide")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "with", "when", "where",
    "which", "will", "can", "may", "than", "through", "using", "use", "how", "their", "its",
}


@st.cache_data(show_spinner=False)
def load_starter_data():
    metadata = pd.read_csv("data/metadata.csv", dtype=str).fillna("")
    documents = pd.read_csv("data/documents.csv", dtype=str).fillna("")
    qrels = pd.read_csv("data/qrels.csv", dtype={"query": str, "doc_id": str, "relevance": int})
    return metadata, documents, qrels


def normalise_url(url: str) -> str:
    url, _ = urldefrag(url.strip())
    parsed = urlparse(url)
    return parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()).geturl().rstrip("/")


def tokens(text: str, mode: str = "Balanced") -> list[str]:
    result = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", str(text).lower())
    if mode != "Raw":
        result = [x for x in result if x not in STOPWORDS]
    if mode == "Aggressive":
        result = [re.sub(r"(ing|edly|ed|ies|s)$", "", x) for x in result]
        result = [x for x in result if len(x) > 2]
    return result


def merge_corpus(metadata: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    return metadata.merge(documents, on="doc_id", how="inner").fillna("")


def minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


@st.cache_resource(show_spinner=False)
def build_index(corpus_json: str, mode: str):
    corpus = pd.read_json(StringIO(corpus_json), orient="records")
    doc_terms, df, term_counts, lengths = {}, Counter(), {}, {}
    for row in corpus.itertuples():
        stream = tokens(f"{row.title} {row.content}", mode)
        counts = Counter(stream)
        doc_terms[row.doc_id] = counts
        term_counts[row.doc_id] = counts
        lengths[row.doc_id] = max(len(stream), 1)
        for word in counts:
            df[word] += 1
    n_docs = max(len(corpus), 1)
    idf = {word: math.log((n_docs + 1) / (count + 1)) + 1 for word, count in df.items()}
    vectors, norms = {}, {}
    for doc_id, counts in doc_terms.items():
        vector = {word: (count / lengths[doc_id]) * idf[word] for word, count in counts.items()}
        vectors[doc_id] = vector
        norms[doc_id] = math.sqrt(sum(v * v for v in vector.values())) or 1.0
    # PageRank on supplied/crawled outgoing links. Unknown targets are ignored.
    ids = set(corpus.doc_id)
    url_to_id = {normalise_url(row.url): row.doc_id for row in corpus.itertuples() if row.url}
    graph = {}
    for row in corpus.itertuples():
        raw = str(getattr(row, "links", ""))
        outgoing = {x for x in raw.split("|") if x in ids}
        outgoing |= {url_to_id[x] for x in raw.split("|") if normalise_url(x) in url_to_id}
        graph[row.doc_id] = outgoing - {row.doc_id}
    rank = {doc_id: 1 / n_docs for doc_id in ids}
    for _ in range(40):
        updated = {doc_id: 0.15 / n_docs for doc_id in ids}
        dangling = sum(rank[d] for d, edges in graph.items() if not edges)
        for doc_id in ids:
            updated[doc_id] += 0.85 * dangling / n_docs
        for source, edges in graph.items():
            if edges:
                share = 0.85 * rank[source] / len(edges)
                for target in edges:
                    updated[target] += share
        rank = updated
    return {"idf": idf, "counts": term_counts, "lengths": lengths, "avg_len": sum(lengths.values()) / n_docs,
            "vectors": vectors, "norms": norms, "pagerank": rank, "df": df, "graph": graph}


def lexical_scores(query: str, index: dict, mode: str, method: str, expand: bool = False) -> tuple[dict, list[str]]:
    qterms = tokens(query, mode)
    if not qterms:
        return {}, []
    if expand:
        # Small, transparent pseudo-relevance feedback from the top BM25 documents.
        initial, _ = lexical_scores(query, index, mode, "BM25", False)
        feedback = Counter()
        for doc_id in sorted(initial, key=initial.get, reverse=True)[:3]:
            feedback.update(index["counts"][doc_id])
        additions = [word for word, _ in feedback.most_common(12) if word not in qterms][:2]
        qterms += additions
    if method == "TF-IDF":
        qcount = Counter(qterms)
        qvec = {w: (c / len(qterms)) * index["idf"].get(w, 0) for w, c in qcount.items()}
        qnorm = math.sqrt(sum(x * x for x in qvec.values())) or 1.0
        scores = {}
        for doc_id, vector in index["vectors"].items():
            dot = sum(qvec.get(w, 0) * vector.get(w, 0) for w in qvec)
            scores[doc_id] = dot / (qnorm * index["norms"][doc_id])
        return scores, qterms
    scores, k1, b = {}, 1.5, 0.75
    for doc_id, counts in index["counts"].items():
        score = 0.0
        for word in qterms:
            frequency = counts.get(word, 0)
            if frequency:
                denom = frequency + k1 * (1 - b + b * index["lengths"][doc_id] / index["avg_len"])
                score += index["idf"].get(word, 0) * frequency * (k1 + 1) / denom
        scores[doc_id] = score
    return scores, qterms


def ranked_results(query: str, corpus: pd.DataFrame, index: dict, mode: str, strategy: str, expand: bool = False):
    method = "TF-IDF" if strategy == "TF-IDF" else "BM25"
    lexical, used_terms = lexical_scores(query, index, mode, method, expand)
    if not lexical:
        return pd.DataFrame(), used_terms
    if strategy == "Hybrid (BM25 + PageRank)":
        bm25, _ = lexical_scores(query, index, mode, "BM25", expand)
        a, p = minmax(bm25), minmax(index["pagerank"])
        scores = {key: 0.82 * a[key] + 0.18 * p[key] for key in bm25}
    else:
        scores = lexical
    output = corpus.copy()
    output["score"] = output.doc_id.map(scores).fillna(0.0)
    output["pagerank"] = output.doc_id.map(index["pagerank"]).fillna(0.0)
    output = output[output.score > 0].sort_values("score", ascending=False)
    output["rank"] = range(1, len(output) + 1)
    return output, used_terms


def exact_and_near_duplicates(corpus: pd.DataFrame):
    hashes, exact = {}, []
    for row in corpus.itertuples():
        digest = hashlib.sha256(re.sub(r"\s+", " ", row.content.lower()).encode()).hexdigest()
        if digest in hashes:
            exact.append((row.doc_id, hashes[digest], 1.0))
        hashes[digest] = row.doc_id
    near = []
    rows = list(corpus.itertuples())
    for i, first in enumerate(rows):
        a = {" ".join(tokens(first.content)[j:j + 3]) for j in range(max(len(tokens(first.content)) - 2, 0))}
        for second in rows[i + 1:]:
            b = {" ".join(tokens(second.content)[j:j + 3]) for j in range(max(len(tokens(second.content)) - 2, 0))}
            sim = len(a & b) / len(a | b) if a | b else 0
            if sim >= 0.28:
                near.append((first.doc_id, second.doc_id, round(sim, 2)))
    return exact, near


def metrics(ranked: list[str], relevant: dict[str, int], k: int = 5):
    binary = [1 if relevant.get(doc, 0) > 0 else 0 for doc in ranked]
    total_rel = sum(x > 0 for x in relevant.values())
    top = binary[:k]
    precision = sum(binary) / len(binary) if binary else 0
    recall = sum(binary) / total_rel if total_rel else 0
    p_k = sum(top) / k
    r_k = sum(top) / total_rel if total_rel else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    ap, hits, reciprocal = 0, 0, 0
    for position, hit in enumerate(binary, 1):
        if hit:
            hits += 1
            ap += hits / position
            if reciprocal == 0:
                reciprocal = 1 / position
    ap /= total_rel or 1
    dcg = sum((2 ** relevant.get(doc, 0) - 1) / math.log2(pos + 1) for pos, doc in enumerate(ranked[:k], 1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(pos + 1) for pos, rel in enumerate(ideal, 1))
    return {"Precision": precision, "Recall": recall, "F1": f1, f"P@{k}": p_k, f"R@{k}": r_k, "MAP": ap, "MRR": reciprocal, f"NDCG@{k}": dcg / idcg if idcg else 0}


def crawl(seeds: list[str], depth_limit: int, max_pages: int):
    if requests is None or BeautifulSoup is None:
        raise RuntimeError("Install requirements.txt before using live crawling.")
    queue, seen, found = deque((normalise_url(x), 0) for x in seeds), set(), []
    robots = {}
    while queue and len(found) < max_pages:
        target, depth = queue.popleft()
        if target in seen or not target.startswith(("http://", "https://")):
            continue
        seen.add(target)
        host = urlparse(target).netloc
        if host not in robots:
            parser = RobotFileParser(urljoin(target, "/robots.txt"))
            try:
                parser.read()
            except Exception:
                pass
            robots[host] = parser
        if not robots[host].can_fetch("ResearchLensAssignmentBot", target):
            continue
        try:
            response = requests.get(target, timeout=8, headers={"User-Agent": "ResearchLensAssignmentBot/1.0"})
            if response.status_code != 200 or "text/html" not in response.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "noscript"]):
                tag.decompose()
            content = " ".join(soup.stripped_strings)
            if len(tokens(content, "Raw")) < 40:
                continue
            links = []
            if depth < depth_limit:
                for anchor in soup.find_all("a", href=True):
                    link = normalise_url(urljoin(target, anchor["href"]))
                    if urlparse(link).netloc == host and link not in seen:
                        queue.append((link, depth + 1))
                    if link.startswith("http"):
                        links.append(link)
            found.append({"url": target, "title": soup.title.get_text(strip=True) if soup.title else target,
                          "content": content[:20000], "links": "|".join(links[:30]), "source": host})
        except Exception:
            continue
    return found, len(seen)


metadata, documents, qrels = load_starter_data()
if "metadata" not in st.session_state:
    st.session_state.metadata, st.session_state.documents = metadata.copy(), documents.copy()
if "run_log" not in st.session_state:
    st.session_state.run_log = []

current = merge_corpus(st.session_state.metadata, st.session_state.documents)
st.sidebar.title("🔎 Research Lens")
st.sidebar.caption("End-to-end Information Retrieval")
page = st.sidebar.radio("Workflow", ["Dashboard", "Crawl & Collect", "Text Mining", "Index Management", "Search", "Recommendations", "Evaluation", "Performance & Inference"])
mode = st.sidebar.selectbox("Preprocessing profile", ["Balanced", "Raw", "Aggressive"], help="Applied consistently while indexing and querying.")
index = build_index(current.to_json(orient="records"), mode)

if page == "Dashboard":
    st.title("Research Lens Dashboard")
    st.caption("A reproducible, Streamlit-only workflow for collecting, indexing, ranking and evaluating documents.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(current))
    c2.metric("Sources", current.source.nunique())
    c3.metric("Vocabulary", len(index["df"]))
    c4.metric("Link edges", sum(len(x) for x in index["graph"].values()))
    st.subheader("Collection composition")
    left, right = st.columns(2)
    with left:
        st.bar_chart(current.category.value_counts())
    with right:
        st.dataframe(current[["doc_id", "title", "source", "category", "published"]], hide_index=True, use_container_width=True)
    st.info("Workflow: collect approved pages → inspect text features → rebuild/query index → explain ranking → recommend related documents → compare metrics.")

elif page == "Crawl & Collect":
    st.title("Crawl & Collect")
    st.caption("Uses a breadth-first frontier, canonical URL handling, a visited set and robots.txt checks. Crawl only sites you are allowed to collect.")
    seeds = st.text_area("Seed URLs (one per line)", placeholder="https://example.org/article\nhttps://example.org/guide")
    a, b = st.columns(2)
    depth = a.slider("Crawling depth", 0, 2, 0)
    maximum = b.slider("Maximum pages", 1, 20, 5)
    if st.button("Start approved crawl", type="primary"):
        urls = [x.strip() for x in seeds.splitlines() if x.strip()]
        if not urls:
            st.warning("Enter at least one approved seed URL.")
        else:
            with st.spinner("Collecting accessible HTML pages..."):
                try:
                    pages, visited = crawl(urls, depth, maximum)
                    if not pages:
                        st.warning(f"No eligible pages were collected after checking {visited} URLs.")
                    else:
                        old_urls = set(st.session_state.metadata.url.map(normalise_url))
                        accepted = [p for p in pages if normalise_url(p["url"]) not in old_urls]
                        start = len(st.session_state.metadata) + 1
                        new_meta, new_docs = [], []
                        for offset, item in enumerate(accepted):
                            doc_id = f"c{start + offset:03d}"
                            new_meta.append({"doc_id": doc_id, "title": item["title"][:180], "url": item["url"], "source": item["source"], "author": "Crawler", "published": str(date.today()), "category": "Crawled"})
                            new_docs.append({"doc_id": doc_id, "content": item["content"], "links": item["links"]})
                        if new_meta:
                            st.session_state.metadata = pd.concat([st.session_state.metadata, pd.DataFrame(new_meta)], ignore_index=True)
                            st.session_state.documents = pd.concat([st.session_state.documents, pd.DataFrame(new_docs)], ignore_index=True)
                        st.success(f"Collected {len(accepted)} new documents ({len(pages) - len(accepted)} duplicate URLs skipped); checked {visited} URLs.")
                        st.dataframe(pd.DataFrame(pages)[["title", "url", "source"]], hide_index=True, use_container_width=True)
                except Exception as exc:
                    st.error(str(exc))

elif page == "Text Mining":
    st.title("Text Preprocessing & Mining")
    corpus_tokens = [tokens(text, mode) for text in current.content]
    flat = [word for doc in corpus_tokens for word in doc]
    frequency = Counter(flat)
    c1, c2, c3 = st.columns(3)
    c1.metric("Tokens", f"{len(flat):,}")
    c2.metric("Unique terms", f"{len(frequency):,}")
    c3.metric("Mean doc length", f"{len(flat) / max(len(current), 1):.1f}")
    st.subheader("Most frequent terms")
    terms = pd.DataFrame(frequency.most_common(20), columns=["term", "frequency"]).set_index("term")
    st.bar_chart(terms)
    st.subheader("Document profile and TF-IDF keywords")
    selection = st.selectbox("Document", current.doc_id, format_func=lambda x: f"{x} — {current.loc[current.doc_id == x, 'title'].iloc[0]}")
    row = current[current.doc_id == selection].iloc[0]
    vector = index["vectors"][selection]
    keywords = sorted(vector, key=vector.get, reverse=True)[:12]
    st.write({"title": row.title, "category": row.category, "source": row.source, "word_count": len(tokens(row.content, mode))})
    st.dataframe(pd.DataFrame({"keyword": keywords, "tf_idf_weight": [round(vector[x], 4) for x in keywords]}), hide_index=True)
    st.subheader("Preprocessing comparison")
    comparison = []
    for profile in ["Raw", "Balanced", "Aggressive"]:
        all_terms = [word for text in current.content for word in tokens(text, profile)]
        comparison.append({"profile": profile, "tokens": len(all_terms), "vocabulary": len(set(all_terms)), "mean length": round(len(all_terms) / len(current), 1)})
    st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
    
    st.subheader("Feature extraction & retrieval comparison")
    st.caption("Comparing BM25 vs TF-IDF ranking strategies across preprocessing profiles using evaluation metrics")
    if len(qrels) > 0:
        feature_comparison = []
        for preproc in ["Raw", "Balanced", "Aggressive"]:
            temp_index = build_index(current.to_json(orient="records"), preproc)
            for method in ["BM25", "TF-IDF"]:
                eval_results = []
                for query, group in qrels.groupby("query"):
                    ranked, _ = ranked_results(query, current, temp_index, preproc, method)
                    relevant = dict(zip(group.doc_id, group.relevance))
                    eval_results.append(metrics(ranked.doc_id.tolist(), relevant, 5))
                mean_metrics = pd.DataFrame(eval_results).mean()
                feature_comparison.append({
                    "Preprocessing": preproc,
                    "Method": method,
                    "MAP": round(mean_metrics["MAP"], 3),
                    "MRR": round(mean_metrics["MRR"], 3),
                    "NDCG@5": round(mean_metrics["NDCG@5"], 3),
                    "Precision": round(mean_metrics["Precision"], 3)
                })
        
        comp_df = pd.DataFrame(feature_comparison)
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
        st.write("**Key Observations:**")
        st.write("- BM25 and TF-IDF perform similarly on this corpus; BM25 may have an advantage on collections with greater document-length variation")
        st.write("- Balanced preprocessing often provides the best trade-off between vocabulary coverage and noise reduction")
        st.write("- Aggressive preprocessing may hurt recall by over-stemming query terms")
    else:
        st.info("Feature extraction comparison requires relevance judgments (qrels.csv)")
    
    st.subheader("Document Classification")
    if sklearn_available and len(current) >= 10 and current.category.nunique() >= 2:
        st.caption("Automated category prediction using TF-IDF features and Multinomial Naive Bayes classifier")
        try:
            # Build raw term counts per document and split by doc_id before computing IDF/TF-IDF
            doc_terms = {row.doc_id: tokens(f"{row.title} {row.content}", mode) for row in current.itertuples()}
            labels = {row.doc_id: row.category for row in current.itertuples()}
            ids = list(doc_terms.keys())
            y = [labels[did] for did in ids]
            
            # Split documents first to prevent data leakage
            # Only stratify if every category has at least 2 examples
            class_counts = Counter(y)
            min_class_count = min(class_counts.values()) if class_counts else 0
            stratify = y if min_class_count >= 2 else None
            train_ids, test_ids, y_train, y_test = train_test_split(ids, y, test_size=0.3, random_state=42, stratify=stratify)
            
            # Build vocabulary and IDF from training documents only
            train_counts = Counter()
            for did in train_ids:
                train_counts.update(set(doc_terms[did]))
            vocab = sorted([term for term, count in train_counts.items() if count > 0])
            n_train = max(len(train_ids), 1)
            idf_train = {term: math.log((n_train + 1) / (train_counts.get(term, 0) + 1)) + 1 for term in vocab}
            
            # Compute TF-IDF vectors for train and test using training IDF
            def make_tfidf(dids):
                rows = []
                for did in dids:
                    counts = Counter(doc_terms[did])
                    total = sum(counts.values()) or 1
                    rows.append([counts.get(term, 0) / total * idf_train.get(term, 0) for term in vocab])
                return np.array(rows)
            
            X_train = make_tfidf(train_ids)
            X_test = make_tfidf(test_ids)
            
            # Train classifier
            clf = MultinomialNB(alpha=0.1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Display results
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Classification Accuracy", f"{accuracy:.2%}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Confusion Matrix**")
                all_labels = sorted(np.unique(y))
                cm = confusion_matrix(y_test, y_pred, labels=all_labels)
                cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
                st.dataframe(cm_df, use_container_width=True)
            
            with col2:
                st.write("**Classification Report**")
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
            
            st.info(f"Trained on {len(train_ids)} documents, tested on {len(test_ids)} documents. TF-IDF vocabulary and IDF were computed from the training split only to avoid data leakage. Stratified split: {stratify is not None}.")
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            st.info("The starter corpus contains categories with very few examples. Classification is presented as a functional demonstration rather than a statistically reliable benchmark.")
    elif not sklearn_available:
        st.warning("Install scikit-learn to enable document classification: `pip install scikit-learn`")
    else:
        st.info("Document classification requires at least 10 documents with 2 or more categories.")

elif page == "Index Management":
    st.title("Index Management")
    st.caption("Metadata and document bodies are maintained in separate collections and joined only for retrieval.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Metadata records", len(st.session_state.metadata))
    c2.metric("Content records", len(st.session_state.documents))
    c3.metric("Inverted-index terms", len(index["df"]))
    exact, near = exact_and_near_duplicates(current)
    st.subheader("Duplicate control")
    st.write(f"Exact duplicates: **{len(exact)}** · Near-duplicate pairs (3-shingle Jaccard ≥ 0.28): **{len(near)}**")
    if near or exact:
        dup_df = pd.DataFrame(exact + near, columns=["document", "canonical/paired document", "similarity"])
        st.dataframe(dup_df, hide_index=True)
        if st.button("Remove duplicates from corpus", type="primary"):
            # Collect all duplicate doc_ids to remove (keep canonical)
            to_remove = set()
            for dup_id, canonical_id, sim in exact:
                to_remove.add(dup_id)  # Remove the duplicate, keep canonical
            for doc1, doc2, sim in near:
                # For near-duplicates, keep the one with lower doc_id (arbitrary but consistent)
                to_remove.add(max(doc1, doc2))
            
            # Remove from both metadata and documents
            st.session_state.metadata = st.session_state.metadata[~st.session_state.metadata.doc_id.isin(to_remove)]
            st.session_state.documents = st.session_state.documents[~st.session_state.documents.doc_id.isin(to_remove)]
            st.success(f"Removed {len(to_remove)} duplicate documents from the corpus. Index will rebuild automatically.")
            st.rerun()
    else:
        st.success("No duplicate or near-duplicate pair crossed the configured threshold.")
    st.subheader("Import a collection")
    upload = st.file_uploader("Upload a CSV with title, content, url (optional), category (optional)", type="csv")
    if upload and st.button("Add uploaded records"):
        incoming = pd.read_csv(upload).fillna("")
        needed = {"title", "content"}
        if not needed.issubset(incoming.columns):
            st.error("The CSV must include title and content columns.")
        else:
            start = len(st.session_state.metadata) + 1
            metas, bodies = [], []
            known_hashes = {hashlib.sha256(re.sub(r"\s+", " ", x.lower()).encode()).hexdigest() for x in st.session_state.documents.content}
            for offset, row in incoming.iterrows():
                digest = hashlib.sha256(re.sub(r"\s+", " ", str(row.content).lower()).encode()).hexdigest()
                if digest in known_hashes:
                    continue
                doc_id = f"u{start + offset:03d}"
                metas.append({"doc_id": doc_id, "title": row.title, "url": row.get("url", ""), "source": row.get("source", "Upload"), "author": row.get("author", "Unknown"), "published": str(row.get("published", date.today())), "category": row.get("category", "Uploaded")})
                bodies.append({"doc_id": doc_id, "content": row.content, "links": row.get("links", "")})
                known_hashes.add(digest)
            st.session_state.metadata = pd.concat([st.session_state.metadata, pd.DataFrame(metas)], ignore_index=True)
            st.session_state.documents = pd.concat([st.session_state.documents, pd.DataFrame(bodies)], ignore_index=True)
            st.success(f"Added {len(metas)} non-duplicate records. The index will rebuild automatically.")
    if st.button("Reset to supplied reproducible corpus"):
        st.session_state.metadata, st.session_state.documents = metadata.copy(), documents.copy()
        st.rerun()

elif page == "Search":
    st.title("Intelligent Web Search")
    query = st.text_input("Search the indexed collection", placeholder="e.g., how should I evaluate ranked retrieval?")
    c1, c2, c3 = st.columns([1, 1, 1])
    strategy = c1.selectbox("Ranking strategy", ["BM25", "TF-IDF", "Hybrid (BM25 + PageRank)"])
    expansion = c2.toggle("Pseudo relevance feedback", help="Adds up to two terms found in the initial top three BM25 results.")
    cutoff = c3.slider("Results", 3, 12, 6)
    if query:
        started = time.perf_counter()
        results, used_terms = ranked_results(query, current, index, mode, strategy, expansion)
        elapsed = (time.perf_counter() - started) * 1000
        st.session_state.run_log.append({"query": query, "strategy": strategy, "latency_ms": elapsed, "results": len(results)})
        st.caption(f"{len(results)} results in {elapsed:.2f} ms · processed terms: {', '.join(used_terms)}")
        if results.empty:
            st.warning("No matching documents. Try fewer terms or the Raw preprocessing profile.")
        else:
            for row in results.head(cutoff).itertuples():
                st.markdown(f"### {row.rank}. {row.title}")
                st.caption(f"{row.category} · {row.source} · Score {row.score:.4f} · PageRank {row.pagerank:.4f}")
                st.write(row.content[:350] + ("…" if len(row.content) > 350 else ""))
                if row.url:
                    st.markdown(f"[Open source]({row.url})")
            st.subheader("Why this order?")
            explain = results.head(cutoff)[["title", "score", "pagerank"]].copy().set_index("title")
            st.bar_chart(explain)

elif page == "Recommendations":
    st.title("Recommendation Panel")
    st.caption("Content-based Top-K recommendations use cosine similarity between TF-IDF document profiles. A small category affinity is added for a transparent hybrid option.")
    selected = st.selectbox("Seed document", current.doc_id, format_func=lambda x: f"{x} — {current.loc[current.doc_id == x, 'title'].iloc[0]}")
    method = st.radio("Recommendation method", ["Content-based", "Hybrid (content + category)"], horizontal=True)
    top_k = st.slider("Top-K", 3, 10, 5)
    seed_vector, seed_norm = index["vectors"][selected], index["norms"][selected]
    selected_category = current.loc[current.doc_id == selected, "category"].iloc[0]
    candidates = []
    for row in current.itertuples():
        if row.doc_id == selected:
            continue
        dot = sum(seed_vector.get(w, 0) * index["vectors"][row.doc_id].get(w, 0) for w in seed_vector)
        similarity = dot / (seed_norm * index["norms"][row.doc_id])
        score = .85 * similarity + .15 * (row.category == selected_category) if method.startswith("Hybrid") else similarity
        candidates.append({"doc_id": row.doc_id, "title": row.title, "category": row.category, "similarity": similarity, "recommendation_score": score})
    recommendations = pd.DataFrame(candidates).sort_values("recommendation_score", ascending=False).head(top_k)
    st.dataframe(recommendations.style.format({"similarity": "{:.3f}", "recommendation_score": "{:.3f}"}), hide_index=True, use_container_width=True)
    st.bar_chart(recommendations.set_index("title")[["recommendation_score"]])
    st.info("Collaborative filtering is discussed in the inference section; it requires genuine user-item interactions, which this document corpus deliberately does not fabricate.")

elif page == "Evaluation":
    st.title("Evaluation Dashboard")
    st.caption("Scores are computed against the included graded qrels. This separates relevance judgment from the system being evaluated.")
    k = st.slider("Evaluation cutoff K", 3, 10, 5)
    evaluations = []
    for strategy in ["BM25", "TF-IDF", "Hybrid (BM25 + PageRank)"]:
        values = []
        for query, group in qrels.groupby("query"):
            ranked, _ = ranked_results(query, current, index, mode, strategy)
            relevant = dict(zip(group.doc_id, group.relevance))
            values.append(metrics(ranked.doc_id.tolist(), relevant, k))
        mean = pd.DataFrame(values).mean().to_dict()
        evaluations.append({"strategy": strategy, **mean})
    table = pd.DataFrame(evaluations).set_index("strategy")
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
    st.subheader("Comparative ranking quality")
    chart_cols = ["MAP", "MRR", f"NDCG@{k}"]
    st.bar_chart(table[[col for col in chart_cols if col in table.columns]])
    st.subheader("Per-query inspection")
    inspected_query = st.selectbox("Judged query", sorted(qrels["query"].unique()))
    show_strategy = st.selectbox("Inspect strategy", ["BM25", "TF-IDF", "Hybrid (BM25 + PageRank)"])
    result, _ = ranked_results(inspected_query, current, index, mode, show_strategy)
    judged = dict(zip(qrels[qrels.query == inspected_query].doc_id, qrels[qrels.query == inspected_query].relevance))
    display = result[["rank", "doc_id", "title", "score"]].copy()
    display["graded_relevance"] = display.doc_id.map(judged).fillna(0).astype(int)
    st.dataframe(display.head(10), hide_index=True, use_container_width=True)

else:
    st.title("Performance Analytics & Inference")
    if st.session_state.run_log:
        history = pd.DataFrame(st.session_state.run_log)
        st.metric("Median observed search latency", f"{history.latency_ms.median():.2f} ms")
        st.dataframe(history.tail(20), hide_index=True, use_container_width=True)
        st.bar_chart(history.groupby("strategy").latency_ms.mean())
    else:
        st.info("Run a few searches to populate in-session latency analytics.")
    st.subheader("Required inferences and discussion")
    with st.expander("1. Highly relevant documents retrieved but poorly ranked", expanded=True):
        st.write("Likely causes are a weak lexical score, untuned BM25 parameters, short/ambiguous queries, stale link authority, and a mismatch between indexing and query preprocessing. Improve ranking by tuning on judged qrels, blending normalized BM25 with PageRank or semantic similarity, using conservative query expansion, boosting trusted metadata fields, and monitoring NDCG/MRR rather than recall alone.")
    with st.expander("2. Effect of duplicates and mitigation"):
        st.write("Copies inflate term statistics and can occupy several top ranks, reduce result diversity, make similar-item recommendations repetitive, and overstate evaluation if every copy is judged relevant. This application uses canonical URLs plus exact SHA-256 content hashes and reports near-duplicates using 3-shingle Jaccard similarity. Production systems also use SimHash/MinHash, canonical-document selection, cluster-aware ranking, and deduplicated qrels.")
    with st.expander("3. Content-based versus collaborative recommendation"):
        st.write("Content-based recommendation is explainable, immediately useful for new items, and needs only text/metadata; it is preferable for a new corpus, specialist topics, or privacy-sensitive systems. Collaborative filtering can discover items with little textual resemblance and may better express community taste, but it needs sufficient genuine user-item interactions and suffers from cold start and sparsity. A hybrid is strongest once reliable feedback exists.")
    with st.expander("4. Why end-to-end integration matters"):
        st.write("Crawling determines coverage and collection quality; preprocessing creates dependable features; indexing enables efficient candidate retrieval; ranking orders candidates by utility; recommendation extends discovery beyond a query; and evaluation closes the loop with evidence. Keeping metadata separate enables filtering and governance throughout. Each stage affects downstream quality, so an integrated UI makes the trade-offs observable and reproducible.")
    with st.expander("5. Learnings from the experiment"):
        st.write("The comparison shows that effectiveness depends on both representation and ranking, not merely finding a relevant document. BM25 is a strong transparent baseline; PageRank provides a useful authority signal when link data is meaningful; preprocessing changes vocabulary substantially; and Top-K recommendations need diversity and duplicate control. The evaluation table should guide configuration choices for the target collection rather than relying on intuition.")
