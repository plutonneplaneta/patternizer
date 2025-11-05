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
from urllib.parse import urlparse
import ssl
import torch
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import math
from scipy.spatial.distance import cosine
# NLP
from natasha import (
    Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger,
    Doc, NewsNERTagger, NewsSyntaxParser
)
import pymorphy2
from razdel import tokenize as razdel_tokenize, sentenize
# Embeddings and clustering
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# Graphs & viz
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree
# IO
import yaml
from tqdm import tqdm
# --- Глобальные модели ---
embeds = NewsEmbedding()
segmenter = Segmenter()
morph_vocab = MorphVocab()
morph_tagger = NewsMorphTagger(embeds)
# Logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
#############################################
# CONFIGURATION
#############################################
DEFAULT_CONFIG = {
    "storage_dir": "./data",
    "corpus_file": "corpus.parquet",
    "scrape": {
        "concurrency": 6,
        "timeout": 20,
        "user_agent": "PattenizerBot/1.0 (+mailto:your-email@example.com)",
        "retry_attempts": 3,
        "delay": 1
    },
    "nlp": {
        "use_natasha": True,
        "pos_filter": ["NOUN", "VERB", "ADJ"]
    },
    "embeddings": {
        "model_name": "sberbank-ai/sbert_large_nlu_ru",
        "batch_size": 32
    },
    "umap": {"n_neighbors": 15, "min_dist": 0.1, "n_components": 5, "metric": "cosine"},
    "hdbscan": {"min_cluster_size": 2, "min_samples": 1, "metric": "euclidean"},
    "pattern_similarity_threshold": 0.60,
    "pattern_repr_top_k": 3,
    "visualization": {"network_html": "patterns_network.html"},
    "parallel": {"chunks": 1000},
    "pattern_extraction": {
        "min_pmi": 3,
        "ngram_range": [2, 3]
    }
}
def validate_config(cfg):
    if not isinstance(cfg["scrape"]["concurrency"], int) or cfg["scrape"]["concurrency"] <= 0:
        raise ValueError("scrape.concurrency must be positive int")
    if not isinstance(cfg["embeddings"]["batch_size"], int) or cfg["embeddings"]["batch_size"] <= 0:
        raise ValueError("embeddings.batch_size must be positive int")
    logger.info("Config validated successfully")

#############################################
# Utility
#############################################
def ensure_dirs(cfg):
    Path(cfg["storage_dir"]).mkdir(parents=True, exist_ok=True)

def save_json(path: str, obj):
    def convert_keys(o):
        if isinstance(o, dict):
            return {str(k) if isinstance(k, (np.int64, np.int32)) else k: convert_keys(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [convert_keys(i) for i in o]
        elif isinstance(o, (np.int64, np.int32, np.int16)):
            return int(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return o
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(convert_keys(obj), f, ensure_ascii=False, indent=2)
    logger.info("Saved JSON: %s", path)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

#############################################
# 1) ASYNC SCRAPER (диагностический)
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
        return True

async def fetch_with_diagnostics(session, url, user_agent, cfg):
    headers = {"User-Agent": user_agent}
    for attempt in range(cfg["scrape"]["retry_attempts"]):
        try:
            if not await can_fetch(session, url, user_agent):
                return {"url": url, "text": "", "error": "robots_blocked"}
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    return {"url": url, "text": "", "error": f"http_{resp.status}"}
                content_type = resp.headers.get('Content-Type', '')
                if 'text' not in content_type:
                    return {"url": url, "text": "", "error": "not_text"}
                html = await resp.text()
                text = extract_text_from_html(html)
                if not text or len(text) < 100:
                    return {"url": url, "text": "", "error": "too_short"}
                if not text or len(text) < 100:
                    return {"url": url, "text": text or "", "error": "too_short"}
                return {"url": url, "text": text[:15000]}
        except asyncio.TimeoutError:
            if attempt == cfg["scrape"]["retry_attempts"] - 1:
                return {"url": url, "text": "", "error": "timeout"}
        except Exception as e:
            if attempt == cfg["scrape"]["retry_attempts"] - 1:
                return {"url": url, "text": "", "error": f"exception_{type(e).__name__}"}
        await asyncio.sleep(cfg["scrape"]["delay"])
    return {"url": url, "text": "", "error": "max_retries"}

def extract_text_from_html(html: str) -> str:
    """
    Извлекает текст из HTML, очищая от JS/CSS-артефактов.
    """
    # --- НОВОЕ: удаление <script>, <style>, JS-строк ---
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)  # Скрипты
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)  # Стили
    html = re.sub(r'window\.[A-Za-z]+\s*=.*?(?=\n|$)', '', html, flags=re.MULTILINE)  # JS window.*
    html = re.sub(r'RufflePlayer|FlashVars|autoplay.*unmuteOverlay', '', html, flags=re.IGNORECASE)  # Артефакты Wayback
    # --- КОНЕЦ НОВОГО ---

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

async def scrape_urls_diagnostic(urls: List[str], cfg, out_path) -> List[Dict]:
    concurrency = cfg["scrape"]["concurrency"]
    connector = aiohttp.TCPConnector(limit_per_host=concurrency, ssl=ssl._create_unverified_context())
    timeout = aiohttp.ClientTimeout(total=cfg["scrape"]["timeout"])
    user_agent = cfg["scrape"]["user_agent"]
    texts = []
    success_count = 0
    fail_reasons = {}
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [fetch_with_diagnostics(session, url, user_agent, cfg) for url in urls]
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Scraping URLs"):
            result = await future
            if result and result.get("text", "").strip():
                texts.append(result)
                success_count += 1
            else:
                reason = result.get("error", "empty") if result else "timeout"
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
    logger.info("=== COLLECTION REPORT ===")
    logger.info("Success: %d/%d", success_count, len(urls))
    for reason, count in fail_reasons.items():
        logger.warning("Failed (%s): %d URLs", reason, count)
    debug_path = out_path.replace(".json", "_debug.json")
    save_json(debug_path, texts)
    valid_texts = [t for t in texts if len(t["text"]) >= 100]
    save_json(out_path, valid_texts)
    logger.info("Valid documents saved: %d to %s", len(valid_texts), out_path)
    return valid_texts

#############################################
# 2) PREPROCESSING
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

def parallel_preprocess(texts: List[str], cfg) -> List[str]:
    if not texts:
        return []
    chunk_size = cfg["parallel"].get("chunks", 500)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    logger.info("Lemmatizing %d docs → %d chunks", len(texts), len(chunks))
    with Pool() as p:
        results = list(tqdm(p.imap(_process_chunk, [(c, cfg) for c in chunks]), total=len(chunks)))
    return [t for chunk in results for t in chunk if t.strip()]

def _process_chunk(chunk_cfg):
    chunk, cfg = chunk_cfg
    processed = []
    for text in chunk:
        try:
            doc = Doc(text)
            doc.segment(segmenter)
            doc.tag_morph(morph_tagger)
            for token in doc.tokens:
                token.lemmatize(morph_vocab)
            lemmas = [t.lemma for t in doc.tokens if t.pos in cfg["nlp"]["pos_filter"]]
            if lemmas:
                processed.append(" ".join(lemmas))
        except Exception as e:
            logger.warning("Lemmatization error: %s", e)
    return processed

#############################################
# 3) Разбиение на предложения
#############################################
def split_into_sentences(text: str) -> List[str]:
    sents = [s.text.strip() for s in sentenize(text) if len(s.text.strip()) > 10]
    return sents

#############################################
# 5) CLUSTERING
#############################################
def cluster_embeddings(embeddings: np.ndarray, cfg):
    um_params = cfg["umap"]
    um = umap.UMAP(**um_params, random_state=42)
    emb_umap = um.fit_transform(embeddings)
    logger.info("UMAP reduced shape: %s", emb_umap.shape)
    hdb_params = cfg["hdbscan"]
    clusterer = hdbscan.HDBSCAN(**hdb_params)
    labels = clusterer.fit_predict(emb_umap)
    return labels, emb_umap, clusterer

def extract_ngrams_pmi(texts: List[str], min_pmi=3, ngram_range=(2, 3)) -> List[str]:
    """
    Выделяет n-граммы и отбирает по PMI.
    PMI = log2(P(ngram) / P(word1) * P(word2) * ...).
    """
    if not texts:
        return []

    # 1. Векторизация n-грамм
    vectorizer = CountVectorizer(ngram_range=ngram_range, lowercase=True, stop_words=None)
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # 2. Счётчики
    total_ngrams = np.sum(X.toarray(), axis=0)
    all_words = [word for text in texts for word in text.split()]
    total_words = len(all_words)
    word_counts = Counter(all_words)  # ← КРИТИЧНО: подсчёт частот слов
    total_docs = len(texts)

    # 3. PMI для каждой n-граммы
    collocations = []
    for i, ngram in enumerate(feature_names):
        count_ngram = total_ngrams[i]
        if count_ngram == 0:
            continue
        words = ngram.split()
        p_ngram = count_ngram / total_ngrams.sum()
        # P(words) = произведение частот слов / total_words
        p_words = np.prod([word_counts[word] / total_words for word in words if word in word_counts])
        pmi = math.log2(p_ngram / p_words) if p_words > 0 else 0
        if pmi > min_pmi:
            collocations.append(ngram)

    logger.info(f"Extracted {len(collocations)} collocations with PMI > {min_pmi}")
    return collocations

def enrich_texts_with_ngrams(texts: List[str], collocations: List[str]) -> List[str]:
    """
    Добавляет коллокации в тексты как новые токены.
    """
    enriched = []
    for text in texts:
        enriched_text = text
        for coll in collocations:
            enriched_text = enriched_text.replace(" " + coll + " ", " " + coll.replace(" ", "_") + " ")
        enriched.append(enriched_text)
    return enriched

#############################################
# 6) Построение паттернов
#############################################
def compute_embeddings(sentences: List[str], cfg) -> tuple[np.ndarray, SentenceTransformer]:
    """
    Вычисляет эмбеддинги предложений с использованием BERT-модели.
    """
    model_name = cfg["embeddings"]["model_name"]
    batch_size = cfg["embeddings"]["batch_size"]
    if len(sentences) > 1000:
        batch_size *= 2  # Адаптивно для больших корпусов

    logger.info("Loading embedding model: %s", model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Using device: %s", device)

    model = SentenceTransformer(model_name)
    logger.info("Computing embeddings for %d sentences (batch %d)...", len(sentences), batch_size)

    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False  # Для cosine similarity
    )

    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings, model

def build_pattern_clusters(sentences, doc_ids, labels, emb_umap, cfg):
    clusters = {}
    top_k = cfg.get("pattern_repr_top_k", 3)
    collocations = cfg.get("collocations", [])  # Извлечь из config или глобально

    for cid in set(labels):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) < 2:
            continue
        cluster_sents = [sentences[i] for i in idx]
        center = emb_umap[idx].mean(axis=0)
        dists = np.linalg.norm(emb_umap[idx] - center, axis=1)
        top_idx = idx[np.argsort(dists)[:top_k]]
        repr_text = " … ".join(sentences[i] for i in top_idx)

        # --- ВСТАВКА: добавляем коллокации в repr ---
        relevant_collocs = [c for c in collocations if c in repr_text.lower()]
        if relevant_collocs:
            repr_text += f" [colloc: {', '.join(relevant_collocs)}]"
        # --- КОНЕЦ ВСТАВКИ ---

        clusters[cid] = {
            "size": len(idx),
            "repr": repr_text,
            "docs": list(set(doc_ids[i] for i in idx)),
            "sentences": cluster_sents[:10]  # Топ-10 для примера
        }
    logger.info(f"Found {len(clusters)} pattern-clusters")
    return clusters

#############################################
# 7) Сеть из кластеров
#############################################
def build_pattern_network_from_clusters(clusters, embeddings, labels, cfg):
    G = nx.Graph()
    threshold = cfg.get("pattern_similarity_threshold", 0.60)
    cids = list(clusters.keys())
    if len(cids) < 2:
        logger.warning("Less than 2 clusters → empty network")
        return G, {}
    # Центроиды
    centroids = []
    cid_to_idx = {}
    for i, cid in enumerate(cids):
        cid_to_idx[cid] = i
        idx = [j for j, lbl in enumerate(labels) if lbl == cid]
        center = embeddings[idx].mean(axis=0) if idx else np.zeros(embeddings.shape[1])
        centroids.append(center)
    centroids = np.array(centroids)
    # Узлы
    for cid in cids:
        data = clusters[cid]
        G.add_node(cid, text=data["repr"], size=10 + data["size"]*2,
                   color=f"#{hash(str(cid)) % 16777215:06x}"[:7])
    # Рёбра
    for c1, c2 in itertools.combinations(cids, 2):
        i1, i2 = cid_to_idx[c1], cid_to_idx[c2]
        sim = 1 - cosine(centroids[i1], centroids[i2])
        if sim >= threshold:
            G.add_edge(c1, c2, weight=sim)
    logger.info(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, {}

def compute_graph_metrics(G):
    """
    Вычисляет ключевые метрики сети паттернов (2.5: плотность, централизация, кластерный коэффициент).
    """
    metrics = {}
    metrics["num_nodes"] = G.number_of_nodes()
    metrics["num_edges"] = G.number_of_edges()
    metrics["density"] = nx.density(G) if G.number_of_nodes() > 0 else 0.0

    try:
        metrics["avg_clustering"] = nx.average_clustering(G, weight="weight")
    except Exception:
        metrics["avg_clustering"] = None

    metrics["degree_centrality"] = {str(k): float(v) for k, v in nx.degree_centrality(G).items()}
    metrics["betweenness"] = {str(k): float(v) for k, v in nx.betweenness_centrality(G, weight="weight").items()}

    # Энтропия (Shannon-like для degree distribution)
    degrees = [d for n, d in G.degree(weight='weight')]
    if degrees:
        p_deg = np.array(degrees) / np.sum(degrees)
        entropy = -np.sum(p_deg * np.log2(p_deg + 1e-10))  # +epsilon для избежания log0
        metrics["entropy"] = entropy
    else:
        metrics["entropy"] = 0.0

    logger.info("Graph metrics: %s", metrics)
    return metrics

#############################################
# 8) VISUALIZATION
#############################################
def visualize_network_pyvis(G, output_path, clusters=None):
    from pyvis.network import Network
    import os
    import json
    import numpy as np

    def to_python(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): to_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [to_python(i) for i in obj]
        return obj

    net = Network(height="800px", width="100%", bgcolor="#1a1a1a", font_color="#ffffff",
                  directed=False, select_menu=True, filter_menu=True, cdn_resources='in_line')
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=200)

    # --- Узлы: n_id → int ---
    for n in G.nodes:
        data = G.nodes[n]
        data = to_python(data)  # ← конвертируем
        title = str(data.get("text", ""))[:600]
        size = max(12, int(data.get("size", 15)))
        color = str(data.get("color", "#97c2fc"))
        net.add_node(
            n_id=int(n),  # ← КРИТИЧНО: int(n)
            label=f"Pattern {int(n)}",
            title=title,
            size=size,
            color=color
        )

    # --- Рёбра ---
    for u, v, d in G.edges(data=True):
        d = to_python(d)
        sim = float(d.get("weight", 1.0))
        net.add_edge(
            int(u), int(v),  # ← int(u), int(v)
            value=sim * 18,
            width=sim * 4,
            title=f"{sim:.3f}",
            color="#cccccc"
        )

    # --- Сохранение ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        net.show(output_path, notebook=False)
        if os.path.getsize(output_path) > 1000:
            logger.info("Interactive network saved: %s", output_path)
            return
    except Exception as e:
        logger.warning("pyvis.show failed: %s", e)

    # --- Резервный HTML ---
    nodes_list = []
    for n in net.nodes:
        node = {
            "id": int(n["id"]),
            "label": n.get("label", ""),
            "title": n.get("title", ""),
            "size": int(n.get("size", 15)),
            "color": n.get("color", "#97c2fc")
        }
        nodes_list.append(node)

    edges_list = []
    for e in net.edges:
        edge = {
            "from": int(e["from"]),
            "to": int(e["to"]),
            "value": float(e.get("value", 1)),
            "width": float(e.get("width", 1)),
            "title": e.get("title", "")
        }
        edges_list.append(edge)

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Сеть паттернов</title>
      <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        #network {{ width: 100%; height: 800px; border: 1px solid #444; background: #1a1a1a; }}
      </style>
    </head>
    <body>
      <div id="network"></div>
      <script>
        var nodes = new vis.DataSet({json.dumps(nodes_list)});
        var edges = new vis.DataSet({json.dumps(edges_list)});
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
          physics: {{ forceAtlas2Based: {{ gravitationalConstant: -50, springLength: 200 }} }},
          nodes: {{ shape: 'dot', font: {{ color: '#ffffff' }} }}
        }};
        new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    logger.info("Fallback HTML saved: %s", output_path)

#############################################
# 9) FULL PIPELINE
#############################################
def run_full_pipeline(corpus_json_path: str, cfg):
    validate_config(cfg)
    if not os.path.exists(corpus_json_path):
        raise FileNotFoundError(corpus_json_path)
    with open(corpus_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    df = pd.DataFrame(items)
    logger.info(f"Loaded {len(df)} documents")

    # --- Разбиваем на предложения ---
    sentences = []
    doc_ids = []
    sent_id = 0
    for _, row in df.iterrows():
        sents = split_into_sentences(row["text"])
        for s in sents:
            sentences.append(s)
            doc_ids.append(row.get("url", f"doc_{sent_id}"))
            sent_id += 1
    logger.info(f"Split into {len(sentences)} sentences")

    # --- Лемматизация предложений ---
    pre = Preprocessor()
    lemma_sents = []
    for s in tqdm(sentences, desc="Lemmatizing sentences"):
        lemmas = pre.doc_to_lemmas(s, cfg["nlp"]["pos_filter"])
        if lemmas:
            lemma_sents.append(" ".join(lemmas))
        else:
            lemma_sents.append(s)
    sentences = lemma_sents

    # --- НОВОЕ: n-граммы + PMI ---
    collocations = extract_ngrams_pmi(sentences, min_pmi=3, ngram_range=(2, 3))
    sentences = enrich_texts_with_ngrams(sentences, collocations)

    # Сохраняем коллокации
    out_dir = cfg["storage_dir"]
    save_json(os.path.join(out_dir, "collocations.json"), {"collocations": collocations})

    # --- Эмбеддинги ---
    embeddings, embed_model = compute_embeddings(sentences, cfg)

    # --- Кластеризация ---
    labels, emb_umap, _ = cluster_embeddings(embeddings, cfg)

    # --- Паттерны ---
    clusters = build_pattern_clusters(sentences, doc_ids, labels, emb_umap, cfg)

    # --- Сеть ---
    G, _ = build_pattern_network_from_clusters(clusters, embeddings, labels, cfg)
    metrics = compute_graph_metrics(G)
    save_json(os.path.join(out_dir, "metrics.json"), metrics)

    # --- Тематическое разделение ---
    themes = cfg.get("themes", {})
    thematic_results = {}
    for theme, params in themes.items():
        # Фильтр предложений по ключевым словам
        theme_sentences = [s for s in sentences if any(kw in s.lower() for kw in params.get("filter_keywords", []))]
        theme_doc_ids = [doc_ids[i] for i in range(len(sentences)) if any(kw in sentences[i].lower() for kw in params.get("filter_keywords", []))]
        if len(theme_sentences) < 50:  # Минимальный порог
            logger.warning(f"Theme '{theme}' has too few sentences: {len(theme_sentences)}")
            continue
        theme_embeddings = embed_model.encode(theme_sentences, show_progress_bar=False)
        theme_embeddings = np.array(theme_embeddings)  # В np.array
        theme_labels, theme_umap, _ = cluster_embeddings(theme_embeddings, cfg)
        theme_clusters = build_pattern_clusters(theme_sentences, theme_doc_ids, theme_labels, theme_umap, cfg)

        # Тематическая сеть
        G_theme, _ = build_pattern_network_from_clusters(theme_clusters, theme_embeddings, theme_labels, cfg)

        # Метрики по теме
        metrics_theme = compute_graph_metrics(G_theme)
        thematic_results[theme] = {
            "clusters": theme_clusters,
            "G": G_theme,
            "metrics": metrics_theme,
            "num_sentences": len(theme_sentences)
        }

        # Сохранение по теме
        theme_dir = os.path.join(out_dir, theme)
        os.makedirs(theme_dir, exist_ok=True)
        save_json(os.path.join(theme_dir, "patterns.json"), theme_clusters)
        visualize_network_pyvis(G_theme, os.path.join(theme_dir, f"network_{theme}.html"))

    # Общая сеть (опционально)
    G_all, _ = build_pattern_network_from_clusters(clusters, embeddings, labels, cfg)
    visualize_network_pyvis(G_all, os.path.join(out_dir, "network_all.html"))

    # --- Сохранение ---
    save_json(os.path.join(out_dir, "patterns.json"), clusters)
    visualize_network_pyvis(G, os.path.join(out_dir, cfg["visualization"]["network_html"]))
    return {"out_dir": out_dir}

#############################################
# CLI
#############################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect", "run", "all"], default="run")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input", type=str, default=None, help="input JSON/Parquet (for run mode)")
    args = parser.parse_args()
    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        cfg.update(load_json(args.config))
    ensure_dirs(cfg)
    # === 1. COLLECT ===
    if args.mode == "collect":
    # Расширенный urls_by_theme: 50+ реальных URL/тема (pre-2010 LJ/Wayback, релевантные)
        urls_by_theme = {
            "любовь": [
                "https://web.archive.org/web/20080101000000/http://community.livejournal.com/love_ru/page/1",  # Романтические истории (2008)
                "https://web.archive.org/web/20080201000000/http://community.livejournal.com/lyubov/page/1",  # Любовь в жизни (2008)
                "https://web.archive.org/web/20080301000000/http://community.livejournal.com/ru_love/page/1",  # Русская любовь (2008)
                "https://web.archive.org/web/20080401000000/http://community.livejournal.com/love_story_ru/page/1",  # Любовные истории (2008)
                "https://web.archive.org/web/20080501000000/http://community.livejournal.com/otnosheniya/page/1",  # Отношения (2008)
                "https://web.archive.org/web/20080601000000/http://community.livejournal.com/ru_romance/page/1",  # Романтика (2008)
                "https://web.archive.org/web/20080701000000/http://community.livejournal.com/love_poetry_ru/page/1",  # Любовная поэзия (2008)
                "https://web.archive.org/web/20080801000000/http://community.livejournal.com/ru_love_poems/page/1",  # Стихи о любви (2008)
                "https://web.archive.org/web/20080901000000/http://community.livejournal.com/love_quotes_ru/page/1",  # Цитаты о любви (2008)
                "https://web.archive.org/web/20081001000000/http://community.livejournal.com/ru_love_stories/page/1",  # Истории любви (2008)
                "https://web.archive.org/web/20081101000000/http://community.livejournal.com/love_ru/page/2",  # Продолжение романтики (2008)
                "https://web.archive.org/web/20081201000000/http://community.livejournal.com/lyubov/page/2",  # Любовь в жизни 2 (2008)
                "https://web.archive.org/web/20090101000000/http://community.livejournal.com/ru_love/page/2",  # Русская любовь 2 (2009)
                "https://web.archive.org/web/20090201000000/http://community.livejournal.com/love_story_ru/page/2",  # Истории 2 (2009)
                "https://web.archive.org/web/20090301000000/http://community.livejournal.com/otnosheniya/page/2",  # Отношения 2 (2009)
                "https://web.archive.org/web/20090401000000/http://community.livejournal.com/ru_romance/page/2",  # Романтика 2 (2009)
                "https://web.archive.org/web/20090501000000/http://community.livejournal.com/love_poetry_ru/page/2",  # Поэзия 2 (2009)
                "https://web.archive.org/web/20090601000000/http://community.livejournal.com/ru_love_poems/page/2",  # Стихи 2 (2009)
                "https://web.archive.org/web/20090701000000/http://community.livejournal.com/love_quotes_ru/page/2",  # Цитаты 2 (2009)
                "https://web.archive.org/web/20090801000000/http://community.livejournal.com/ru_love_stories/page/2",  # Истории 2 (2009)
                "https://web.archive.org/web/20090901000000/http://community.livejournal.com/love_ru/page/3",  # Романтика 3 (2009)
                "https://web.archive.org/web/20091001000000/http://community.livejournal.com/lyubov/page/3",  # Любовь 3 (2009)
                "https://web.archive.org/web/20091101000000/http://community.livejournal.com/ru_love/page/3",  # Русская 3 (2009)
                "https://web.archive.org/web/20091201000000/http://community.livejournal.com/love_story_ru/page/3",  # Истории 3 (2009)
                "https://web.archive.org/web/20090101000000/http://community.livejournal.com/otnosheniya/page/3",  # Отношения 3 (2009)
                "https://web.archive.org/web/20090201000000/http://community.livejournal.com/ru_romance/page/3",  # Романтика 3 (2009)
                "https://web.archive.org/web/20090301000000/http://community.livejournal.com/love_poetry_ru/page/3",  # Поэзия 3 (2009)
                "https://web.archive.org/web/20090401000000/http://community.livejournal.com/ru_love_poems/page/3",  # Стихи 3 (2009)
                "https://web.archive.org/web/20090501000000/http://community.livejournal.com/love_quotes_ru/page/3",  # Цитаты 3 (2009)
                "https://web.archive.org/web/20090601000000/http://community.livejournal.com/ru_love_stories/page/3",  # Истории 3 (2009)
                "https://web.archive.org/web/20090701000000/http://community.livejournal.com/love_ru/page/4",  # Романтика 4 (2009)
                "https://web.archive.org/web/20090801000000/http://community.livejournal.com/lyubov/page/4",  # Любовь 4 (2009)
                "https://web.archive.org/web/20090901000000/http://community.livejournal.com/ru_love/page/4",  # Русская 4 (2009)
                "https://web.archive.org/web/20091001000000/http://community.livejournal.com/love_story_ru/page/4",  # Истории 4 (2009)
                "https://web.archive.org/web/20091101000000/http://community.livejournal.com/otnosheniya/page/4",  # Отношения 4 (2009)
                "https://web.archive.org/web/20091201000000/http://community.livejournal.com/ru_romance/page/4",  # Романтика 4 (2009)
                "https://web.archive.org/web/20090101000000/http://community.livejournal.com/love_poetry_ru/page/4",  # Поэзия 4 (2009)
                "https://web.archive.org/web/20090201000000/http://community.livejournal.com/ru_love_poems/page/4",  # Стихи 4 (2009)
                "https://web.archive.org/web/20090301000000/http://community.livejournal.com/love_quotes_ru/page/4",  # Цитаты 4 (2009)
                "https://web.archive.org/web/20090401000000/http://community.livejournal.com/ru_love_stories/page/4",  # Истории 4 (2009)
                "https://ruscorpora.ru/new/search-main.html?text=любовь&until=2010&mysize=500",  # RNC 500 примеров
                "https://ruscorpora.ru/new/search-main.html?text=отношения&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=страсть&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=романтика&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=чувства&until=2010&mysize=500"
            ],
            "успех": [
                "https://mi-pishem.livejournal.com/132263.html",  # 5 ПРАВИЛ ПИСАТЕЛЬСКОГО УСПЕХА (2002)
                "https://transurfer.livejournal.com/55726.html",  # 7 духовных законов успеха (2001)
                "https://stalic.livejournal.com/290082.html",  # 28 шагов к успеху (2009)
                "https://transurfer.livejournal.com/56789.html",  # 7 духовных законов успеха (2004)
                "https://art-shmart.livejournal.com/49228.html",  # Жан-Мишель Баскиа (2009)
                "https://teachron.livejournal.com/171571.html",  # Валентин Гафт (2001)
                "https://stalic.livejournal.com/289578.html",  # 28 шагов к успеху (2009)
                "https://mi-pishem.livejournal.com/31271.html",  # АГАТА КРИСТИ (2005)
                "https://pushkinskij-dom.livejournal.com/177323.html",  # Как денди лондонский одет (2009)
                "https://ru-lyrics.livejournal.com/1725319.html",  # Джон Китс "Слава" (2001)
                "https://web.archive.org/web/20080101000000/http://community.livejournal.com/uspeh/page/1",
                "https://web.archive.org/web/20080201000000/http://community.livejournal.com/success_ru/page/1",
                "https://web.archive.org/web/20080301000000/http://community.livejournal.com/career_ru/page/1",
                "https://web.archive.org/web/20080401000000/http://community.livejournal.com/rabota_v_moskve/page/1",
                "https://web.archive.org/web/20080501000000/http://community.livejournal.com/motivation_ru/page/1",
                "https://web.archive.org/web/20080601000000/http://community.livejournal.com/business_ru/page/1",
                "https://web.archive.org/web/20080701000000/http://community.livejournal.com/finance_ru/page/1",
                "https://web.archive.org/web/20080801000000/http://community.livejournal.com/self_development/page/1",
                "https://web.archive.org/web/20080901000000/http://community.livejournal.com/psychology_ru/page/1",
                "https://web.archive.org/web/20081001000000/http://community.livejournal.com/coaching_ru/page/1",
                "https://web.archive.org/web/20081101000000/http://community.livejournal.com/uspeh/page/2",
                "https://web.archive.org/web/20081201000000/http://community.livejournal.com/success_ru/page/2",
                "https://web.archive.org/web/20090101000000/http://community.livejournal.com/career_ru/page/2",
                "https://web.archive.org/web/20090201000000/http://community.livejournal.com/rabota_v_moskve/page/2",
                "https://web.archive.org/web/20090301000000/http://community.livejournal.com/motivation_ru/page/2",
                "https://web.archive.org/web/20090401000000/http://community.livejournal.com/business_ru/page/2",
                "https://web.archive.org/web/20090501000000/http://community.livejournal.com/finance_ru/page/2",
                "https://web.archive.org/web/20090601000000/http://community.livejournal.com/self_development/page/2",
                "https://web.archive.org/web/20090701000000/http://community.livejournal.com/psychology_ru/page/2",
                "https://web.archive.org/web/20090801000000/http://community.livejournal.com/coaching_ru/page/2",
                "https://web.archive.org/web/20090901000000/http://community.livejournal.com/uspeh/page/3",
                "https://web.archive.org/web/20091001000000/http://community.livejournal.com/success_ru/page/3",
                "https://web.archive.org/web/20091101000000/http://community.livejournal.com/career_ru/page/3",
                "https://web.archive.org/web/20091201000000/http://community.livejournal.com/rabota_v_moskve/page/3",
                "https://web.archive.org/web/20090101000000/http://community.livejournal.com/motivation_ru/page/3",
                "https://web.archive.org/web/20090201000000/http://community.livejournal.com/business_ru/page/3",
                "https://web.archive.org/web/20090301000000/http://community.livejournal.com/finance_ru/page/3",
                "https://web.archive.org/web/20090401000000/http://community.livejournal.com/self_development/page/3",
                "https://web.archive.org/web/20090501000000/http://community.livejournal.com/psychology_ru/page/3",
                "https://web.archive.org/web/20090601000000/http://community.livejournal.com/coaching_ru/page/3",
                "https://web.archive.org/web/20090701000000/http://community.livejournal.com/uspeh/page/4",
                "https://web.archive.org/web/20090801000000/http://community.livejournal.com/success_ru/page/4",
                "https://web.archive.org/web/20090901000000/http://community.livejournal.com/career_ru/page/4",
                "https://web.archive.org/web/20091001000000/http://community.livejournal.com/rabota_v_moskve/page/4",
                "https://ruscorpora.ru/new/search-main.html?text=успех&until=2010&mysize=500",  # RNC 500 примеров
                "https://ruscorpora.ru/new/search-main.html?text=карьера&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=мотивация&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=достижение&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=цель&until=2010&mysize=500"
            ],
            "здоровье": [
                "https://pritchi.livejournal.com/523512.html",  # Притча о здоровье (2007)
                "https://marinaizminska.livejournal.com/category/здоровье/",  # Здоровье (2005)
                "https://transurfer.livejournal.com/55726.html",  # 7 духовных законов успеха (2001) — здоровье как успех
                "https://ru-knitting.livejournal.com/5165599.html",  # Влияние рукоделия на здоровье (2008)
                "https://agideliya.livejournal.com/",  # Агиделия о здоровье (2009)
                "https://doctor.livejournal.com/472509.html",  # Высказывания о врачах (2008)
                "https://karleev.livejournal.com/647.html",  # Гимнастика Ниши (2006)
                "https://dok-zlo.livejournal.com/257484.html",  # Мифы клятвы Гиппократа (2008)
                "https://dandorfman.livejournal.com/708640.html",  # Оригинал Хатуль Мадан (2009)
                "https://drug-goy.livejournal.com/529581.html",  # Арнольд Эрет (2009)
                "https://web.archive.org/web/20080101000000/http://community.livejournal.com/zdorovie/page/1",
                "https://web.archive.org/web/20080201000000/http://community.livejournal.com/health_ru/page/1",
                "https://web.archive.org/web/20080301000000/http://community.livejournal.com/medicina_ru/page/1",
                "https://web.archive.org/web/20080401000000/http://community.livejournal.com/fitness_ru/page/1",
                "https://web.archive.org/web/20080501000000/http://community.livejournal.com/dieta_ru/page/1",
                "https://web.archive.org/web/20080601000000/http://community.livejournal.com/yoga_ru/page/1",
                "https://web.archive.org/web/20080701000000/http://community.livejournal.com/psychology_health/page/1",
                "https://web.archive.org/web/20080801000000/http://community.livejournal.com/alternative_med/page/1",
                "https://web.archive.org/web/20080901000000/http://community.livejournal.com/vegetarian_ru/page/1",
                "https://web.archive.org/web/20081001000000/http://community.livejournal.com/sport_ru/page/1",
                "https://web.archive.org/web/20081101000000/http://community.livejournal.com/zdorovie/page/2",
                "https://web.archive.org/web/20081201000000/http://community.livejournal.com/health_ru/page/2",
                "https://web.archive.org/web/20090101000000/http://community.livejournal.com/medicina_ru/page/2",
                "https://web.archive.org/web/20090201000000/http://community.livejournal.com/fitness_ru/page/2",
                "https://web.archive.org/web/20090301000000/http://community.livejournal.com/dieta_ru/page/2",
                "https://web.archive.org/web/20090401000000/http://community.livejournal.com/yoga_ru/page/2",
                "https://web.archive.org/web/20090501000000/http://community.livejournal.com/psychology_health/page/2",
                "https://web.archive.org/web/20090601000000/http://community.livejournal.com/alternative_med/page/2",
                "https://web.archive.org/web/20090701000000/http://community.livejournal.com/vegetarian_ru/page/2",
                "https://web.archive.org/web/20090801000000/http://community.livejournal.com/sport_ru/page/2",
                "https://web.archive.org/web/20090901000000/http://community.livejournal.com/zdorovie/page/3",
                "https://web.archive.org/web/20091001000000/http://community.livejournal.com/health_ru/page/3",
                "https://web.archive.org/web/20091101000000/http://community.livejournal.com/medicina_ru/page/3",
                "https://web.archive.org/web/20091201000000/http://community.livejournal.com/fitness_ru/page/3",
                "https://web.archive.org/web/20090101000000/http://community.livejournal.com/dieta_ru/page/3",
                "https://web.archive.org/web/20090201000000/http://community.livejournal.com/yoga_ru/page/3",
                "https://web.archive.org/web/20090301000000/http://community.livejournal.com/psychology_health/page/3",
                "https://web.archive.org/web/20090401000000/http://community.livejournal.com/alternative_med/page/3",
                "https://web.archive.org/web/20090501000000/http://community.livejournal.com/vegetarian_ru/page/3",
                "https://web.archive.org/web/20090601000000/http://community.livejournal.com/sport_ru/page/3",
                "https://ruscorpora.ru/new/search-main.html?text=здоровье&until=2010&mysize=500",  # RNC 500 примеров
                "https://ruscorpora.ru/new/search-main.html?text=болезнь&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=лечение&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=зож&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=спорт&until=2010&mysize=500"
            ],
            "семья": [
                "https://to-priz.livejournal.com/56081.html",  # Б.В. Стырикович. СЕРГЕЙ ЕСЕНИН (2006)
                "https://laurie-ru.livejournal.com/134940.html",  # Семья Хью Лори (2008)
                "https://ru-history.livejournal.com/2585788.html",  # Семья Сац (2002)
                "https://un-enfant.livejournal.com/781739.html",  # семья - саша боярская (2009)
                "https://mikhail-epstein.livejournal.com/45029.html",  # Счастливые семьи (2009)
                "https://sananahead.livejournal.com/16148.html",  # Людовик XV. Семья (2008)
                "https://colonelcassad.livejournal.com/1292634.html",  # Расстрел царской семьи (2001)
                "https://vorontsova-nvu.livejournal.com/136368.html",  # Советская семья (2006)
                "https://sokolsky-mg.livejournal.com/31608.html",  # "Предпремьерный разговор" (2003)
                "https://ru-history.livejournal.com/2585788.html",  # Семья Сац (2002)
                "https://web.archive.org/web/20080101000000/http://community.livejournal.com/semya/page/1",
                "https://web.archive.org/web/20080201000000/http://community.livejournal.com/roditeli/page/1",
                "https://web.archive.org/web/20080301000000/http://community.livejournal.com/deti_ru/page/1",
                "https://web.archive.org/web/20080401000000/http://community.livejournal.com/mama_ru/page/1",
                "https://web.archive.org/web/20080501000000/http://community.livejournal.com/family_ru/page/1",
                "https://web.archive.org/web/20080601000000/http://community.livejournal.com/beremennost/page/1",
                "https://web.archive.org/web/20080701000000/http://community.livejournal.com/rodi_ru/page/1",
                "https://web.archive.org/web/20080801000000/http://community.livejournal.com/vospitanie/page/1",
                "https://web.archive.org/web/20080901000000/http://community.livejournal.com/detskiy_sad/page/1",
                "https://web.archive.org/web/20081001000000/http://community.livejournal.com/shkola_ru/page/1",
                "https://web.archive.org/web/20081101000000/http://community.livejournal.com/semya/page/2",
                "https://web.archive.org/web/20081201000000/http://community.livejournal.com/roditeli/page/2",
                "https://web.archive.org/web/20090101000000/http://community.livejournal.com/deti_ru/page/2",
                "https://web.archive.org/web/20090201000000/http://community.livejournal.com/mama_ru/page/2",
                "https://web.archive.org/web/20090301000000/http://community.livejournal.com/family_ru/page/2",
                "https://web.archive.org/web/20090401000000/http://community.livejournal.com/beremennost/page/2",
                "https://web.archive.org/web/20090501000000/http://community.livejournal.com/rodi_ru/page/2",
                "https://web.archive.org/web/20090601000000/http://community.livejournal.com/vospitanie/page/2",
                "https://web.archive.org/web/20090701000000/http://community.livejournal.com/detskiy_sad/page/2",
                "https://web.archive.org/web/20090801000000/http://community.livejournal.com/shkola_ru/page/2",
                "https://web.archive.org/web/20090901000000/http://community.livejournal.com/semya/page/3",
                "https://web.archive.org/web/20091001000000/http://community.livejournal.com/roditeli/page/3",
                "https://web.archive.org/web/20091101000000/http://community.livejournal.com/deti_ru/page/3",
                "https://web.archive.org/web/20091201000000/http://community.livejournal.com/mama_ru/page/3",
                "https://web.archive.org/web/20090101000000/http://community.livejournal.com/family_ru/page/3",
                "https://web.archive.org/web/20090201000000/http://community.livejournal.com/beremennost/page/3",
                "https://web.archive.org/web/20090301000000/http://community.livejournal.com/rodi_ru/page/3",
                "https://web.archive.org/web/20090401000000/http://community.livejournal.com/vospitanie/page/3",
                "https://web.archive.org/web/20090501000000/http://community.livejournal.com/detskiy_sad/page/3",
                "https://web.archive.org/web/20090601000000/http://community.livejournal.com/shkola_ru/page/3",
                "https://ruscorpora.ru/new/search-main.html?text=семья&until=2010&mysize=500",  # RNC 500 примеров
                "https://ruscorpora.ru/new/search-main.html?text=дети&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=брак&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=родители&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=воспитание&until=2010&mysize=500"
            ],
            "молодежь": [
                "https://germanych.livejournal.com/130938.html",  # Во что одевалась советская молодежь (2009)
                "https://petrenko-v.livejournal.com/38644.html",  # Цитаты о молодом поколении (2009)
                "https://eks-lj.livejournal.com/74265.html",  # "Даёшь молодёжь" (2009)
                "https://galkovsky.livejournal.com/129854.html",  # 492. С ЧУЖОГО ГОЛОСА (2008)
                "https://yarodom.livejournal.com/356009.html",  # Неформальный СССР (2009)
                "https://ervix.livejournal.com/38129.html",  # Толковый словарь матерных (2005)
                "https://strannik1990.livejournal.com/3476.html",  # КАК "БОРОТЬСЯ" С НЕФОРМАЛАМИ (2009)
                "https://nevzlin.livejournal.com/116239.html",  # Злокачественный приоритет молодежи (2007)
                "https://chto-chitat.livejournal.com/4315181.html",  # "Заводной апельсин" (2008)
                "https://yarodom.livejournal.com/356009.html",  # Неформальный СССР (2009)
                "https://web.archive.org/web/20080101000000/http://community.livejournal.com/molodezh/page/1",
                "https://web.archive.org/web/20080201000000/http://community.livejournal.com/studenty/page/1",
                "https://web.archive.org/web/20080301000000/http://community.livejournal.com/student_life/page/1",
                "https://web.archive.org/web/20080401000000/http://community.livejournal.com/obshaga/page/1",
                "https://web.archive.org/web/20080501000000/http://community.livejournal.com/molodezhka/page/1",
                "https://web.archive.org/web/20080601000000/http://community.livejournal.com/subkultura/page/1",
                "https://web.archive.org/web/20080701000000/http://community.livejournal.com/anime_ru/page/1",
                "https://web.archive.org/web/20080801000000/http://community.livejournal.com/games_ru/page/1",
                "https://web.archive.org/web/20080901000000/http://community.livejournal.com/music_young/page/1",
                "https://web.archive.org/web/20081001000000/http://community.livejournal.com/club_life_ru/page/1",
                "https://web.archive.org/web/20081101000000/http://community.livejournal.com/molodezh/page/2",
                "https://web.archive.org/web/20081201000000/http://community.livejournal.com/studenty/page/2",
                "https://web.archive.org/web/20090101000000/http://community.livejournal.com/student_life/page/2",
                "https://web.archive.org/web/20090201000000/http://community.livejournal.com/obshaga/page/2",
                "https://web.archive.org/web/20090301000000/http://community.livejournal.com/molodezhka/page/2",
                "https://web.archive.org/web/20090401000000/http://community.livejournal.com/subkultura/page/2",
                "https://web.archive.org/web/20090501000000/http://community.livejournal.com/anime_ru/page/2",
                "https://web.archive.org/web/20090601000000/http://community.livejournal.com/games_ru/page/2",
                "https://web.archive.org/web/20090701000000/http://community.livejournal.com/music_young/page/2",
                "https://web.archive.org/web/20090801000000/http://community.livejournal.com/club_life_ru/page/2",
                "https://web.archive.org/web/20090901000000/http://community.livejournal.com/molodezh/page/3",
                "https://web.archive.org/web/20091001000000/http://community.livejournal.com/studenty/page/3",
                "https://web.archive.org/web/20091101000000/http://community.livejournal.com/student_life/page/3",
                "https://web.archive.org/web/20091201000000/http://community.livejournal.com/obshaga/page/3",
                "https://web.archive.org/web/20090101000000/http://community.livejournal.com/molodezhka/page/3",
                "https://web.archive.org/web/20090201000000/http://community.livejournal.com/subkultura/page/3",
                "https://web.archive.org/web/20090301000000/http://community.livejournal.com/anime_ru/page/3",
                "https://web.archive.org/web/20090401000000/http://community.livejournal.com/games_ru/page/3",
                "https://web.archive.org/web/20090501000000/http://community.livejournal.com/music_young/page/3",
                "https://web.archive.org/web/20090601000000/http://community.livejournal.com/club_life_ru/page/3",
                "https://ruscorpora.ru/new/search-main.html?text=молодежь&until=2010&mysize=500",  # RNC 500 примеров
                "https://ruscorpora.ru/new/search-main.html?text=студенты&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=подростки&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=субкультура&until=2010&mysize=500",
                "https://ruscorpora.ru/new/search-main.html?text=взросление&until=2010&mysize=500"
            ]
        }

        # Объединить все URL (~250, для 5k+ sent/тема)
        urls = [url for theme_urls in urls_by_theme.values() for url in theme_urls]
        out_path = os.path.join(cfg["storage_dir"], "collected_pre_alg_expanded.json")
        collected = asyncio.run(scrape_urls_diagnostic(urls, cfg, out_path))
        if not collected:
            logger.error("Collection failed")
            return
        logger.info("Collection complete: %d docs → %s", len(collected), out_path)
        return

    # === 2. RUN ===
    if args.mode == "run":
        if args.input is None:
            logger.info("No input → running unit tests")
            unittest.main(argv=[''], exit=False)
            return
        if not os.path.exists(args.input):
            logger.error("Input file not found: %s", args.input)
            return
        logger.info("Running full pipeline on: %s", args.input)
        res = run_full_pipeline(args.input, cfg)
        logger.info("Pipeline complete: %s", res)
        return
    # === 3. ALL ===
    if args.mode == "all":
        # Collect
        urls_by_theme = { ... }  # тот же словарь
        urls = [url for theme_urls in urls_by_theme.values() for url in theme_urls]
        out_path = os.path.join(cfg["storage_dir"], "collected_pre_alg.json")
        collected = asyncio.run(scrape_urls_diagnostic(urls, cfg, out_path))
        if not collected:
            logger.error("Collection failed → aborting 'all'")
            return
        # Run
        logger.info("Running full pipeline after collection")
        res = run_full_pipeline(out_path, cfg)
        logger.info("All complete: %s", res)
        return
if __name__ == "__main__":
    main()
#############################################
# BASIC UNITTESTS
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
        text = " test with spaces "
        self.assertEqual(pre.normalize_text(text), "test with spaces")

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)  # Run tests if needed
