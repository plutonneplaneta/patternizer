#!/usr/bin/env python3
"""
pipeline_max_ru.py
Программа-«максимум» для сбора, предобработки, эмбеддинга, кластеризации и сетевого анализа
русскоязычных корпусных данных. Модульная, конфигурируемая, готовая к расширению.
"""

import os
import sys
import argparse
import asyncio
import aiohttp
import aiofiles
import json
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from readability import Document
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from tqdm.asyncio import tqdm as atq
from multiprocessing import Pool, cpu_count

# NLP
from natasha import (
    Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger,
    Doc, NewsNERTagger, NewsSyntaxParser
)
import pymorphy2
from razdel import tokenize as razdel_tokenize

# Embeddings and clustering
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer  # Для custom topics

# Graphs & viz
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree  # Для оптимизации edges

# IO
import yaml
from tqdm import tqdm

# Logging with improved formatter
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

#############################################
#  CONFIGURATION (с custom validation)
#############################################
DEFAULT_CONFIG = {
    "storage_dir": "./data",
    "corpus_file": "corpus.parquet",
    "scrape": {
        "concurrency": 6,
        "timeout": 20,
        "user_agent": "PattenizerBot/1.0 (+mailto:your-email@example.com)",
        "retry_attempts": 3,
        "delay": 1  # sec between requests
    },
    "nlp": {
        "use_natasha": True,
        "pos_filter": ["NOUN", "VERB", "ADJ"]  # Добавлено для фильтра
    },
    "embeddings": {
        "model_name": "sberbank-ai/sbert_large_nlu_ru",
        "batch_size": 32
    },
    "umap": {"n_neighbors": 15, "min_dist": 0.1, "n_components": 5, "metric": "cosine"},
    "hdbscan": {"min_cluster_size": 10, "min_samples": 5, "metric": "euclidean"},
    "pattern_similarity_threshold": 0.72,
    "visualization": {"network_html": "patterns_network.html"},
    "parallel": {"chunks": 1000}  # Для scalability
}

def validate_config(cfg):
    # Custom type/value checks (аналог pydantic)
    if not isinstance(cfg["scrape"]["concurrency"], int) or cfg["scrape"]["concurrency"] <= 0:
        raise ValueError("scrape.concurrency must be positive int")
    if not isinstance(cfg["embeddings"]["batch_size"], int) or cfg["embeddings"]["batch_size"] <= 0:
        raise ValueError("embeddings.batch_size must be positive int")
    # Добавить больше проверок по мере надобности
    logger.info("Config validated successfully")

#############################################
#  Utility: filesystem
#############################################
def ensure_dirs(cfg):
    Path(cfg["storage_dir"]).mkdir(parents=True, exist_ok=True)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

#############################################
#  1) ASYNC SCRAPER (с robots.txt, retries, delay)
#############################################
async def can_fetch(session, url, user_agent):
    rp = RobotFileParser()
    robots_url = urljoin(url, "/robots.txt")
    try:
        async with session.get(robots_url) as resp:
            if resp.status == 200:
                txt = await resp.text()
                rp.parse(txt.splitlines())
                return rp.can_fetch(user_agent, url)
    except:
        return True  # Assume yes if error
    return True

async def fetch(session, url, cfg):
    headers = {"User-Agent": cfg["scrape"]["user_agent"]}
    timeout = aiohttp.ClientTimeout(total=cfg["scrape"]["timeout"])
    for attempt in range(cfg["scrape"]["retry_attempts"]):
        try:
            if not await can_fetch(session, url, headers["User-Agent"]):
                logger.warning("Robots.txt disallows %s", url)
                return None
            async with session.get(url, headers=headers, timeout=timeout) as resp:
                if resp.status == 200 and 'text' in resp.headers.get('Content-Type',''):
                    return await resp.text()
                else:
                    logger.warning("Bad status %s for %s", resp.status, url)
                    return None
        except Exception as e:
            logger.debug("Fetch error %s (attempt %d): %s", url, attempt+1, e)
            await asyncio.sleep(cfg["scrape"]["delay"])
    return None

def extract_text_from_html(html: str) -> str:
    try:
        doc = Document(html)
        content = doc.summary()
        text = re.sub(r'<[^>]+>', ' ', content)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 200:
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.debug("Readability failed: %s", e)
        return re.sub(r'<[^>]+>', ' ', html)

async def scrape_urls(urls: List[str], cfg, out_path):
    concurrency = cfg["scrape"]["concurrency"]
    connector = aiohttp.TCPConnector(limit_per_host=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch(session, url, cfg) for url in urls]
        results = []
        for coro in atq.as_completed(tasks):
            html = await coro
            results.append(html)
            await asyncio.sleep(cfg["scrape"]["delay"])  # Global delay
        texts = []
        for url, html in zip(urls, results):
            if html:
                text = extract_text_from_html(html)
                texts.append({"url": url, "text": text})
        await asyncio.to_thread(save_json, out_path, texts)
        return texts

#############################################
#  2) PREPROCESSING (расширенные stop-words, POS-filter, parallel)
#############################################
STOP_WORDS = set([
    "и", "в", "на", "с", "по", "для", "как", "что", "это", "не", "а", "то", "от", "к", "у", "из", "о", "за", "но", "или",
    "если", "при", "до", "после", "над", "под", "между", "через", "во", "со", "без", "ни", "же", "ли", "бы", "да", "нет",
    "он", "она", "они", "мы", "вы", "ты", "я", "его", "ее", "их", "нас", "вас", "тебя", "меня", "себя", "мой", "твой", 
    "свой", "наш", "ваш", "этот", "тот", "сам", "каждый", "весь", "все", "всё", "какой", "такой", "который", "где", 
    "когда", "куда", "откуда", "почему", "потому", "здесь", "там", "тогда", "теперь", "уже", "еще", "ещё", "всегда", 
    "никогда", "иногда", "часто", "редко", "очень", "слишком", "почти", "совсем", "только", "лишь", "даже", "вот", 
    "там", "тут", "здесь", "так", "то", "сегодня", "завтра", "вчера", "утром", "днем", "вечером", "ночью", "более", 
    "менее", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять", "много", "мало", 
    "несколько", "кажется", "возможно", "вероятно", "потому что", "хотя", "однако", "ведь", "именно", "просто", "сейчас"
])  # Расширенный список ~200 из RNC-inspired

class Preprocessor:
    def __init__(self):
        logger.info("Initializing Natasha models...")
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        self.pymorphy = pymorphy2.MorphAnalyzer()

    def normalize_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def doc_to_lemmas(self, text: str, pos_filter: List[str]) -> List[str]:
        text = self.normalize_text(text)
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        lemmas = []
        for token in doc.tokens:
            if token.pos in pos_filter:  # POS-filter
                lemma = token.lemma or token.text.lower()
                lemmas.append(lemma)
            elif token.pos is None:
                parsed = self.pymorphy.parse(token.text)[0]
                if parsed.tag.POS in pos_filter:
                    lemmas.append(parsed.normal_form)
        return lemmas

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 2]

def parallel_preprocess(texts, cfg):
    pre = Preprocessor()
    pos_filter = cfg["nlp"]["pos_filter"]
    def process_chunk(chunk):
        processed = []
        for t in chunk:
            lemmas = pre.doc_to_lemmas(t, pos_filter)
            toks = pre.filter_tokens(lemmas)
            processed.append(" ".join(toks))
        return processed
    chunk_size = cfg["parallel"]["chunks"]
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(process_chunk, chunks), total=len(chunks)))
    return [item for sublist in results for item in sublist]

#############################################
#  3) EMBEDDINGS (с GPU, adaptive batch)
#############################################
def compute_embeddings(sentences: List[str], cfg):
    model_name = cfg["embeddings"]["model_name"]
    bs = cfg["embeddings"]["batch_size"]
    if len(sentences) > 1000:
        bs *= 2  # Adaptive
    logger.info("Loading embedding model: %s", model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    logger.info("Computing embeddings on %s for %d sentences (batch %d)...", device, len(sentences), bs)
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=bs)
    return embeddings, model

#############################################
#  4) CLUSTERING (с grid search, custom topics)
#############################################
def grid_search_params(embeddings, cfg):
    # Simple grid для tuning (аналог hyperopt)
    umap_grids = [{"n_neighbors": nn} for nn in [10, 15, 20]]
    hdb_grids = [{"min_cluster_size": ms} for ms in [5, 10, 15]]
    best_score = -1
    best_params = {}
    for um_p in umap_grids:
        for hdb_p in hdb_grids:
            um = umap.UMAP(**um_p)
            emb_umap = um.fit_transform(embeddings)
            clusterer = hdbscan.HDBSCAN(**hdb_p)
            labels = clusterer.fit_predict(emb_umap)
            score = clusterer.relative_validity_ if len(set(labels)) > 1 else -1  # DBCV score
            if score > best_score:
                best_score = score
                best_params = {"umap": um_p, "hdbscan": hdb_p}
    logger.info("Best params from grid: %s (score %f)", best_params, best_score)
    return best_params.get("umap", cfg["umap"]), best_params.get("hdbscan", cfg["hdbscan"])

def cluster_embeddings(embeddings: np.ndarray, cfg):
    um_params, hdb_params = grid_search_params(embeddings, cfg)
    um = umap.UMAP(**um_params)
    emb_umap = um.fit_transform(embeddings)
    logger.info("UMAP reduced shape: %s", emb_umap.shape)
    clusterer = hdbscan.HDBSCAN(**hdb_params)
    labels = clusterer.fit_predict(emb_umap)
    return labels, emb_umap, clusterer

def extract_topics_per_cluster(texts, labels):
    # Custom TF-IDF для topic labels (аналог BERTopic)
    unique_labels = set(labels) - {-1}
    topics = {}
    for lbl in unique_labels:
        cluster_texts = [texts[i] for i in range(len(labels)) if labels[i] == lbl]
        if len(cluster_texts) > 1:
            vectorizer = TfidfVectorizer(max_features=5)
            tfidf = vectorizer.fit_transform(cluster_texts)
            top_words = vectorizer.get_feature_names_out()
            topics[lbl] = list(top_words)
    return topics

#############################################
#  5) BUILD PATTERN NETWORK (оптимизировано KDTree, community detection)
#############################################
def build_pattern_network_from_clusters(labels, items, embeddings, cfg):
    df = pd.DataFrame({"item": items, "label": labels})
    cluster_ids = sorted([int(x) for x in set(labels) if x != -1])
    centroids = {}
    cluster_members = {}
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        centroids[cid] = np.mean(embeddings[idx], axis=0)
        cluster_members[cid] = [items[i] for i in idx]
    # Оптимизация edges с KDTree
    cent_array = np.array(list(centroids.values()))
    tree = KDTree(cent_array)
    pairs = tree.query_pairs(r=1 - cfg["pattern_similarity_threshold"], p=np.inf, output_type='ndarray')  # Для cosine approx
    G = nx.Graph()
    for cid, members in cluster_members.items():
        G.add_node(cid, label=" / ".join(cluster_members[cid][:3]), size=len(members))
    for i, j in pairs:
        ci = cent_array[i]; cj = cent_array[j]
        sim = cosine_similarity(ci.reshape(1,-1), cj.reshape(1,-1))[0,0]
        if sim >= cfg["pattern_similarity_threshold"]:
            G.add_edge(cluster_ids[i], cluster_ids[j], weight=float(sim))
    # Community detection
    communities = nx.community.louvain_communities(G, weight="weight")
    logger.info("Detected %d communities", len(communities))
    return G, cluster_members, centroids, communities

#############################################
# 6) METRICS & VISUALIZATION
#############################################
def compute_graph_metrics(G):
    metrics = {}
    metrics["num_nodes"] = G.number_of_nodes()
    metrics["num_edges"] = G.number_of_edges()
    metrics["density"] = nx.density(G)
    try:
        metrics["avg_clustering"] = nx.average_clustering(G, weight="weight")
    except Exception:
        metrics["avg_clustering"] = None
    metrics["degree_centrality"] = nx.degree_centrality(G)
    metrics["betweenness"] = nx.betweenness_centrality(G, weight="weight")
    return metrics

def visualize_network_pyvis(G, out_html, cluster_members):
    net = Network(notebook=True, height="800px", width="100%", bgcolor="#ffffff")  # notebook=True для удобства
    for n, data in G.nodes(data=True):
        label = str(n) + " | " + (data.get("label") or "")
        size = data.get("size", 5) * 8
        net.add_node(n, label=label, title="<br>".join(cluster_members.get(n,[])[:10]), value=size)
    for u,v,d in G.edges(data=True):
        net.add_edge(u, v, value=d.get("weight", 1.0))
    net.show(out_html)
    logger.info("Network saved to %s", out_html)

#############################################
#  7) FULL PIPELINE RUN (с chunks, topics, communities)
#############################################
def run_full_pipeline(corpus_json_path: str, cfg):
    ensure_dirs(cfg)
    validate_config(cfg)
    if corpus_json_path.endswith(".parquet"):
        df = pd.read_parquet(corpus_json_path)
    else:
        with open(corpus_json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        df = pd.DataFrame(items)
    # Parallel preprocess
    logger.info("Lemmatizing %d documents in parallel...", len(df))
    texts = parallel_preprocess(df['text'].tolist(), cfg)
    texts = [t for t in texts if len(t.split()) > 3]
    # Embeddings
    embeddings, embed_model = compute_embeddings(texts, cfg)
    # Clustering
    labels, emb_umap, clusterer = cluster_embeddings(embeddings, cfg)
    topics = extract_topics_per_cluster(texts, labels)
    # Build network
    G, clusters, centroids, communities = build_pattern_network_from_clusters(labels, texts, embeddings, cfg)
    metrics = compute_graph_metrics(G)
    logger.info("Graph metrics: %s", metrics)
    # Save outputs
    out_dir = cfg["storage_dir"]
    save_json(os.path.join(out_dir, "pipeline_cfg.json"), cfg)
    save_json(os.path.join(out_dir, "clusters.json"), clusters)
    save_json(os.path.join(out_dir, "topics.json"), topics)
    save_json(os.path.join(out_dir, "communities.json"), communities)
    nx.write_gexf(G, os.path.join(out_dir, "patterns_network.gexf"))
    visualize_network_pyvis(G, os.path.join(out_dir, cfg["visualization"]["network_html"]), clusters)
    # UMAP plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=emb_umap[:,0], y=emb_umap[:,1], hue=labels, palette="tab10", legend=False, s=5)
    plt.title("UMAP projection colored by HDBSCAN labels")
    plt.savefig(os.path.join(out_dir, "umap_clusters.png"), dpi=300)
    logger.info("Saved UMAP and network visualizations in %s", out_dir)
    return {"metrics": metrics, "out_dir": out_dir}

#############################################
# CLI
#############################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect","run","all"], default="run")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input", type=str, default=None, help="input JSON/Parquet (for run mode)")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        cfg.update(load_json(args.config))

    ensure_dirs(cfg)

    if args.mode == "collect":
        urls = [
            "https://lenta.ru/news/2010/06/01/example/",  # Реальные URLы добавить
        ]
        out_path = os.path.join(cfg["storage_dir"], "collected.json")
        asyncio.run(scrape_urls(urls, cfg, out_path))
        logger.info("Collection finished. Saved to %s", out_path)
        return

    if args.mode == "run":
        if not args.input:
            print("Provide --input path to JSON of documents")
            return
        res = run_full_pipeline(args.input, cfg)
        logger.info("Pipeline complete: %s", res)
        return

    if args.mode == "all":
        # Collect then run (example)
        urls = []  # Заполнить
        out_path = os.path.join(cfg["storage_dir"], "collected.json")
        asyncio.run(scrape_urls(urls, cfg, out_path))
        res = run_full_pipeline(out_path, cfg)
        logger.info("All complete: %s", res)
        return

if __name__ == "__main__":
    main()

#############################################
#  BASIC UNITTESTS (stdlib unittest)
#############################################
import unittest

class TestPipeline(unittest.TestCase):
    def test_validate_config(self):
        cfg = DEFAULT_CONFIG.copy()
        self.assertIsNone(validate_config(cfg))  # No raise
        cfg["scrape"]["concurrency"] = -1
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_normalize_text(self):
        pre = Preprocessor()
        text = "  test   with spaces  "
        self.assertEqual(pre.normalize_text(text), "test with spaces")

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)  # Run tests if needed
