from corus import load_taiga_social
import logging
from pathlib import Path
from typing import List, Dict, Set
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math
import json
import re

# NLP
import pymorphy2
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
# Embeddings
from sentence_transformers import SentenceTransformer
# Graphs
import networkx as nx
from pyvis.network import Network

# ----------------------------
# Конфигурация
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Natasha / pymorphy
embeds = NewsEmbedding()
segmenter = Segmenter()
morph_vocab = MorphVocab()
morph_tagger = NewsMorphTagger(embeds)
pymorphy = pymorphy2.MorphAnalyzer()

# темы
themes = ['любовь', 'успех', 'здоровье', 'семья', 'молодёжь']

CONFIG = {
    "storage_dir": "./data",
    "social_file": "social.tar.gz",
    "pattern_extraction": {
        "min_pmi": 2.0,  # Уменьшили с 5.0
        "min_doc_freq": 2,  # Уменьшили с 5
        "min_count": 2,  # Уменьшили с 5
        "ngram_range": (2, 4)  # Расширили до 4-грамм
    },
    "embeddings": {
        "model_name": "cointegrated/rubert-tiny2", 
        "batch_size": 32
    },
    "pattern_similarity_threshold": 0.6,  # Уменьшили порог для больше связей
    "max_examples_per_theme": 2000,
    "nlp": {
        "pos_filter": ["NOUN", "ADJ", "VERB", "ADV", "PROPN"]  # Добавили имена собственные
    },
    "quality_filtering": {
        "min_quality_score": 0.2,  # Уменьшили порог
        "max_patterns_per_theme": 100,  # Уменьшили максимальное количество
        "semantic_similarity_threshold": 0.2  # Уменьшили порог семантической схожести
    },
    "diversity": {
        "max_similar_patterns": 10,  # Увеличили количество похожих паттернов
        "subsumption_similarity_threshold": 0.7  # Увеличили порог для фильтрации вложенных
    },
    "keyword_extraction": {
        "num_keywords_per_theme": 50  # Увеличили количество ключевых слов
    }
}

# Расширенный набор русских стоп-слов
RUSSIAN_STOPWORDS = {
    "и", "в", "во", "не", "на", "я", "что", "он", "она", "они", "это", "как", "с", "со", "а", "по", 
    "только", "у", "за", "от", "для", "же", "бы", "быть", "без", "над", "про", "из", "или", "к", 
    "до", "о", "об", "их", "его", "ее", "ими", "наш", "ваш", "тут", "там", "же", "как-то", "вот",
    "но", "да", "ли", "был", "была", "было", "были", "еще", "уже", "нет", "даже", "мне", "меня",
    "мной", "тебе", "тебя", "тобой", "ему", "его", "им", "ней", "нее", "ею", "нам", "нас", "нами",
    "вам", "вас", "вами", "им", "их", "ними", "себя", "себе", "собой", "мой", "твой", "свой",
    "наш", "ваш", "его", "ее", "их", "этот", "тот", "такой", "такая", "такое", "такие", "все",
    "всё", "вся", "всю", "весь", "всем", "всеми", "сам", "сама", "само", "сами", "свои", "своих",
    "то", "это", "того", "этому", "та", "те", "тем", "теми", "ту", "той", "тех", "таких", "таким",
    "такими", "такому", "таком", "такая", "такие", "такую", "такою", "таким", "такими", "таком",
    "такому", "таков", "такова", "таково", "таковы", "таковым", "таковыми", "таковому", "таковом"
}

# ----------------------------
# Утилиты
# ----------------------------
def ensure_dirs(storage_dir):
    Path(storage_dir).mkdir(parents=True, exist_ok=True)

def cast_numpy_types(obj):
    import numpy as np
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): cast_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [cast_numpy_types(i) for i in obj]
    return obj

# ----------------------------
# 1. Сбор и сохранение корпуса по темам
# ----------------------------
def fetch_examples_from_taiga(themes: List[str]) -> List[Dict]:
    social_path = Path(CONFIG["storage_dir"]) / CONFIG["social_file"]
    examples = []
    theme_counts = {theme: 0 for theme in themes}

    records_social = load_taiga_social(str(social_path))
    for record in records_social:
        try:
            title = getattr(record, 'title', '') or ''
            text = getattr(record, 'text', '') or ''
            author = getattr(record, 'author', '') or ''
            url = getattr(record, 'url', '') or ''
            for theme in themes:
                if theme_counts[theme] >= CONFIG["max_examples_per_theme"]:
                    continue
                # простая тематическая фильтрация по наличию слова темы
                if theme.lower() in text.lower() or theme.lower() in title.lower():
                    examples.append({
                        'theme': theme,
                        'text': text.strip(),
                        'source': url,
                        'year': None,
                        'author': author
                    })
                    theme_counts[theme] += 1
                    if theme_counts[theme] % 100 == 0:
                        logger.info(f"'{theme}': собрано {theme_counts[theme]} текстов")
        except Exception as e:
            logger.warning(f"Ошибка при чтении записи: {e}")
            continue

    logger.info(f"Всего собрано текстов: {len(examples)}")
    return examples

def collect_corpus_by_theme(themes: List[str]) -> Dict[str, List[str]]:
    ensure_dirs(CONFIG["storage_dir"])
    examples = fetch_examples_from_taiga(themes)
    corpus_by_theme = {theme: [] for theme in themes}
    for ex in examples:
        corpus_by_theme[ex['theme']].append(ex['text'])

    # Сохраняем только по темам (никаких объединённых файлов)
    for theme, texts in corpus_by_theme.items():
        path = Path(CONFIG["storage_dir"]) / f"corpus_{theme}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        logger.info(f"Сохранено {len(texts)} текстов для темы '{theme}' -> {path}")
    return corpus_by_theme

# ----------------------------
# 2. Лемматизация и автоматическое извлечение ключевых слов
# ----------------------------
def lemmatize_text(text: str, pos_filter: List[str]) -> str:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        if token.pos in pos_filter:
            lemmas.append(token.lemma.lower())
    return ' '.join(lemmas)

def extract_theme_keywords_automatically(corpus_by_theme: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Автоматически извлекает ключевые слова для каждой темы с помощью TF-IDF и хи-квадрат
    """
    num_keywords = CONFIG["keyword_extraction"]["num_keywords_per_theme"]
    
    # Собираем все тексты и создаем метки тем
    all_texts = []
    theme_labels = []
    
    for theme, texts in corpus_by_theme.items():
        all_texts.extend(texts)
        theme_labels.extend([theme] * len(texts))
    
    if not all_texts:
        return {}
    
    # Создаем TF-IDF матрицу
    tfidf = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.9,  # Увеличили max_df
        token_pattern=r'(?u)\b[\w-]+\b',
        stop_words=list(RUSSIAN_STOPWORDS)
    )
    
    X_tfidf = tfidf.fit_transform(all_texts)
    feature_names = tfidf.get_feature_names_out()
    
    # Для каждой темы выбираем лучшие ключевые слова
    theme_keywords = {}
    
    for theme in corpus_by_theme.keys():
        # Создаем бинарные метки для текущей темы
        y_theme = [1 if label == theme else 0 for label in theme_labels]
        
        if sum(y_theme) == 0:
            continue
            
        # Используем хи-квадрат для отбора признаков
        chi2_selector = SelectKBest(chi2, k=min(num_keywords * 3, X_tfidf.shape[1]))
        X_theme = chi2_selector.fit_transform(X_tfidf, y_theme)
        
        # Получаем индексы и оценки лучших признаков
        scores = chi2_selector.scores_
        indices = np.argsort(scores)[::-1]
        
        # Отбираем ключевые слова
        keywords = []
        for idx in indices[:num_keywords * 2]:
            word = feature_names[idx]
            # Ослабили фильтрацию коротких слов
            if len(word) >= 2 and word not in RUSSIAN_STOPWORDS:
                keywords.append(word)
        
        # Дополнительно фильтруем по TF-IDF внутри темы
        theme_indices = [i for i, label in enumerate(theme_labels) if label == theme]
        if theme_indices:
            theme_tfidf = X_tfidf[theme_indices]
            word_scores = np.asarray(theme_tfidf.mean(axis=0)).flatten()
            word_ranking = np.argsort(word_scores)[::-1]
            
            final_keywords = []
            for idx in word_ranking:
                if len(final_keywords) >= num_keywords:
                    break
                word = feature_names[idx]
                if word in keywords and word not in final_keywords:
                    final_keywords.append(word)
            
            theme_keywords[theme] = final_keywords
            logger.info(f"Авто-ключевые слова для '{theme}': {len(final_keywords)} слов")
    
    return theme_keywords

# ----------------------------
# 3. Извлечение и фильтрация паттернов
# ----------------------------
def extract_clean_ngrams_from_theme(lemmatized_texts: List[str],
                                    min_pmi: float,
                                    min_doc_freq: int,
                                    min_count: int,
                                    ngram_range: tuple,
                                    theme_keywords_lemmas: List[str] = None) -> List[Dict]:
    """
    Возвращает список словарей-паттернов с улучшенной фильтрацией
    """
    if not lemmatized_texts:
        return []

    # CountVectorizer для подсчёта частот и doc-freq
    vect = CountVectorizer(ngram_range=ngram_range, token_pattern=r"(?u)\b\w+\b")
    X = vect.fit_transform(lemmatized_texts)
    feature_names = vect.get_feature_names_out()
    counts = X.sum(axis=0).A1
    doc_freq = (X > 0).sum(axis=0).A1

    # словарь частот слов для PMI
    all_words = [w for text in lemmatized_texts for w in text.split()]
    total_words = len(all_words)
    word_counts = Counter(all_words)

    patterns = []
    total_ngrams_sum = counts.sum() if counts.sum() > 0 else 1
    
    for i, ngram in enumerate(feature_names):
        c = int(counts[i])
        df = int(doc_freq[i])
        
        # Базовые фильтры по частоте
        if c < min_count or df < min_doc_freq:
            continue
            
        words = ngram.split()
        
        # ФИЛЬТР 1: короткие слова (пропускаем слова короче 2 символов)
        if any(len(w) < 2 for w in words): 
            continue
            
        # ФИЛЬТР 2: если n-gram состоит ИСКЛЮЧИТЕЛЬНО из стоп-слов
        if all(w in RUSSIAN_STOPWORDS for w in words):
            continue
            
        # ФИЛЬТР 3: если это одиночное слово И оно стоп-слово
        if len(words) == 1 and words[0] in RUSSIAN_STOPWORDS:
            continue
            
        # ФИЛЬТР 4: повторы слов в n-gram
        if len(set(words)) < len(words):
            continue

        # ФИЛЬТР 5: проверяем, что n-gram содержит хотя бы одно значащее слово
        meaningful_words = [w for w in words if w not in RUSSIAN_STOPWORDS and len(w) >= 2]  # Уменьшили до 2
        if not meaningful_words:
            continue
            
        # ФИЛЬТР 6: ОПЦИОНАЛЬНОЕ ВХОЖДЕНИЕ КЛЮЧЕВЫХ СЛОВ ТЕМЫ
        # Сделаем это мягче: если есть ключевые слова, проверяем, но не строго
        if theme_keywords_lemmas:
            has_keyword = any(keyword in words for keyword in theme_keywords_lemmas)
            # Если нет ключевых слов, все равно оставляем, но с пониженным приоритетом
            if not has_keyword:
                # Пропускаем только если PMI очень низкий
                pass

        # PMI calculation
        p_ngram = c / total_ngrams_sum
        p_words = 1.0
        valid = True
        for w in words:
            wc = word_counts.get(w, 0)
            if wc == 0:
                valid = False
                break
            p_words *= (wc / total_words)
        if not valid:
            continue
            
        pmi = math.log2(p_ngram / p_words) if p_words > 0 else 0.0
        if pmi >= min_pmi:
            # Определяем, какие ключевые слова присутствуют в n-gram
            present_keywords = []
            if theme_keywords_lemmas:
                present_keywords = [kw for kw in theme_keywords_lemmas if kw in words]
                
            patterns.append({
                "ngram": ngram, 
                "count": c, 
                "doc_freq": df, 
                "pmi": float(pmi),
                "keywords": present_keywords,
                "meaningful_words": meaningful_words,
                "has_keyword": len(present_keywords) > 0
            })

    # Сортируем: сначала с ключевыми словами, потом по PMI
    patterns = sorted(patterns, key=lambda x: (x["has_keyword"], x["pmi"]), reverse=True)
    logger.info(f"Базовое извлечение: {len(patterns)} паттернов для темы")
    
    return patterns

def filter_patterns_by_semantic_relevance(patterns: List[Dict], 
                                        theme: str,
                                        model: SentenceTransformer) -> List[Dict]:
    """
    Фильтрует паттерны по семантической близости к теме через эмбеддинги
    """
    if not patterns:
        return []
    
    similarity_threshold = CONFIG["quality_filtering"]["semantic_similarity_threshold"]
    
    # Создаем эмбеддинги для названия темы
    theme_embedding = model.encode([theme], convert_to_tensor=True)
    
    # Создаем эмбеддинги для всех паттернов
    pattern_texts = [p["ngram"] for p in patterns]
    pattern_embeddings = model.encode(pattern_texts, convert_to_tensor=True)
    
    # Вычисляем косинусную схожесть
    similarities = cosine_similarity(theme_embedding.cpu(), pattern_embeddings.cpu())[0]
    
    # Фильтруем паттерны
    filtered_patterns = []
    for i, pattern in enumerate(patterns):
        pattern["semantic_similarity"] = float(similarities[i])
        # Не фильтруем строго, а используем оценку в качестве метрики
        filtered_patterns.append(pattern)
    
    logger.info(f"Семантическая оценка: {len(patterns)} паттернов")
    return filtered_patterns

def calculate_pattern_quality_score(pattern: Dict) -> float:
    """
    Вычисляет композитную оценку качества паттерна на основе нескольких метрик
    """
    # Веса для разных метрик
    if pattern.get("has_keyword", False):
        # Паттерны с ключевыми словами получают бонус
        alpha, beta, gamma, delta = 0.3, 0.25, 0.25, 0.2
    else:
        alpha, beta, gamma, delta = 0.4, 0.3, 0.2, 0.1
    
    # Нормализуем метрики
    normalized_pmi = min(pattern.get("pmi", 0) / 8.0, 1.0)  # уменьшили max PMI
    normalized_df = min(pattern.get("doc_freq", 0) / 20.0, 1.0)  # уменьшили max doc_freq
    semantic_sim = pattern.get("semantic_similarity", 0.3)  # по умолчанию ниже
    has_keyword = 1.0 if pattern.get("has_keyword", False) else 0.0
    
    # Композитная оценка
    score = (alpha * normalized_pmi + 
             beta * normalized_df + 
             gamma * semantic_sim +
             delta * has_keyword)
    
    return score

def filter_patterns_by_quality(patterns: List[Dict]) -> List[Dict]:
    """
    Фильтрует паттерны по композитной оценке качества
    """
    min_quality_score = CONFIG["quality_filtering"]["min_quality_score"]
    top_k = CONFIG["quality_filtering"]["max_patterns_per_theme"]
    
    for pattern in patterns:
        pattern["quality_score"] = calculate_pattern_quality_score(pattern)
    
    # Сортируем по оценке качества
    patterns.sort(key=lambda x: x["quality_score"], reverse=True)
    
    # Берем топ-K, но не фильтруем по минимальному порогу слишком строго
    filtered = patterns[:top_k]
    
    logger.info(f"Качественная фильтрация: {len(patterns)} -> {len(filtered)} паттернов")
    return filtered

def advanced_filter_subsumed_ngrams(patterns: List[Dict], 
                                  model: SentenceTransformer) -> List[Dict]:
    """
    Продвинутая фильтрация вложенных n-грамм с учетом семантического перекрытия
    """
    if len(patterns) <= 1:
        return patterns
    
    similarity_threshold = CONFIG["diversity"]["subsumption_similarity_threshold"]
    
    # Создаем эмбеддинги для всех паттернов
    pattern_texts = [p["ngram"] for p in patterns]
    pattern_embeddings = model.encode(pattern_texts)
    
    # Строим матрицу схожести
    similarity_matrix = cosine_similarity(pattern_embeddings)
    
    # Сортируем паттерны по убыванию качества и длины
    patterns_with_idx = [(i, p) for i, p in enumerate(patterns)]
    patterns_with_idx.sort(key=lambda x: (x[1]["quality_score"], len(x[1]["ngram"].split())), reverse=True)
    
    filtered_indices = []
    
    for i, pattern in patterns_with_idx:
        keep_pattern = True
        
        for j in filtered_indices:
            sim = similarity_matrix[i, j]
            
            # Если паттерны семантически очень похожи
            if sim > similarity_threshold:
                current_tokens = set(pattern["ngram"].split())
                existing_tokens = set(patterns[j]["ngram"].split())
                
                # Если текущий паттерн короче или такого же размера
                if len(current_tokens) <= len(existing_tokens):
                    # Сравниваем качество
                    if pattern["quality_score"] <= patterns[j]["quality_score"] * 1.1:  # Уменьшили коэффициент
                        keep_pattern = False
                        break
        
        if keep_pattern:
            filtered_indices.append(i)
    
    filtered_patterns = [patterns[i] for i in filtered_indices]
    logger.info(f"Фильтрация вложенных n-грамм: {len(patterns)} -> {len(filtered_patterns)}")
    return filtered_patterns

def ensure_pattern_diversity(patterns: List[Dict],
                           model: SentenceTransformer) -> List[Dict]:
    """
    Обеспечивает разнообразие паттернов, ограничивая количество семантически похожих
    """
    max_similar = CONFIG["diversity"]["max_similar_patterns"]
    
    if len(patterns) <= max_similar:
        return patterns
    
    pattern_texts = [p["ngram"] for p in patterns]
    pattern_embeddings = model.encode(pattern_texts)
    
    # Группируем семантически похожие паттерны
    clustering = DBSCAN(eps=0.6, min_samples=1, metric='cosine').fit(pattern_embeddings)  # Увеличили eps
    labels = clustering.labels_
    
    filtered_patterns = []
    
    # Для каждого кластера берем только лучшие паттерны
    for cluster_id in set(labels):
        if cluster_id == -1:  # шумовые точки
            cluster_patterns = [p for i, p in enumerate(patterns) if labels[i] == -1]
            filtered_patterns.extend(cluster_patterns)
        else:
            cluster_patterns = [p for i, p in enumerate(patterns) if labels[i] == cluster_id]
            # Сортируем по качеству и берем лучшие
            cluster_patterns.sort(key=lambda x: x["quality_score"], reverse=True)
            filtered_patterns.extend(cluster_patterns[:max_similar])
    
    # Сортируем итоговый список по качеству
    filtered_patterns.sort(key=lambda x: x["quality_score"], reverse=True)
    
    logger.info(f"Фильтрация по разнообразию: {len(patterns)} -> {len(filtered_patterns)}")
    return filtered_patterns

def extract_high_quality_patterns(lemmatized_texts: List[str],
                                theme: str,
                                min_pmi: float,
                                min_doc_freq: int,
                                min_count: int,
                                ngram_range: tuple,
                                theme_keywords: List[str] = None) -> List[Dict]:
    """
    Комплексное извлечение качественных паттернов с продвинутой фильтрацией
    """
    # Базовое извлечение паттернов
    patterns = extract_clean_ngrams_from_theme(
        lemmatized_texts=lemmatized_texts,
        min_pmi=min_pmi,
        min_doc_freq=min_doc_freq,
        min_count=min_count,
        ngram_range=ngram_range,
        theme_keywords_lemmas=theme_keywords
    )
    
    if not patterns:
        logger.warning(f"Нет паттернов после базового извлечения для темы '{theme}'")
        return []
    
    # Инициализируем модель для семантической фильтрации
    model = SentenceTransformer(CONFIG["embeddings"]["model_name"])
    
    # 1. Семантическая оценка (не фильтрация)
    patterns = filter_patterns_by_semantic_relevance(patterns, theme, model)
    
    if not patterns:
        logger.warning(f"Нет паттернов после семантической оценки для темы '{theme}'")
        return []
    
    # 2. Вычисление композитных оценок качества
    patterns = filter_patterns_by_quality(patterns)
    
    if not patterns:
        logger.warning(f"Нет паттернов после качественной фильтрации для темы '{theme}'")
        return []
    
    # 3. Фильтрация вложенных n-грамм (только если много паттернов)
    if len(patterns) > 20:
        patterns = advanced_filter_subsumed_ngrams(patterns, model)
    
    if not patterns:
        logger.warning(f"Нет паттернов после фильтрации вложенных n-грамм для темы '{theme}'")
        return []
    
    # 4. Фильтрация по разнообразию (только если много паттернов)
    if len(patterns) > 30:
        patterns = ensure_pattern_diversity(patterns, model)
    
    logger.info(f"Итоговое количество паттернов для темы '{theme}': {len(patterns)}")
    return patterns

# ----------------------------
# 4. Построение сети паттернов и интерактивный HTML (для темы)
# ----------------------------
def build_pattern_network_for_theme(patterns: List[Dict], theme: str) -> tuple[nx.Graph, Dict]:
    """
    patterns: list of dicts {'ngram','count','doc_freq','pmi'}
    """
    if not patterns:
        logger.warning(f"Нет паттернов для темы '{theme}'")
        return nx.Graph(), {}

    ngrams = [p["ngram"] for p in patterns]
    model = SentenceTransformer(CONFIG["embeddings"]["model_name"])
    embeddings = model.encode(ngrams, batch_size=CONFIG["embeddings"]["batch_size"], show_progress_bar=True)

    G = nx.Graph()
    # Добавляем узлы
    for idx, p in enumerate(patterns):
        words = p["ngram"].split()
        # Размер узла зависит от качества и наличия ключевых слов
        base_size = 10
        size_boost = p.get("quality_score", 0) * 20
        if p.get("has_keyword", False):
            size_boost += 5
            
        G.add_node(idx,
                   label=p["ngram"],
                   words=words,
                   count=int(p["count"]),
                   doc_freq=int(p["doc_freq"]),
                   pmi=float(p["pmi"]),
                   quality_score=float(p.get("quality_score", 0)),
                   semantic_similarity=float(p.get("semantic_similarity", 0)),
                   has_keyword=bool(p.get("has_keyword", False)),
                   title=f"{p['ngram']} (quality={p.get('quality_score', 0):.2f}, pmi={p['pmi']:.2f}, keywords={p.get('has_keyword', False)})",
                   size=max(8, min(40, base_size + size_boost))
                   )

    # Эдджевая матрица
    edge_count = 0
    for i in range(len(ngrams)):
        for j in range(i + 1, len(ngrams)):
            sim = float(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0])
            if sim >= CONFIG["pattern_similarity_threshold"]:
                color_val = int(max(0, min(255, (sim - 0.5) * 2 * 255)))
                color_hex = f"rgb({255 - color_val},{color_val},120)"
                G.add_edge(i, j, weight=float(sim), title=f"sim={sim:.3f}", color=color_hex, width=1 + (sim - CONFIG["pattern_similarity_threshold"]) * 3)
                edge_count += 1

    logger.info(f"Создано {edge_count} рёбер для {len(ngrams)} узлов")

    # Вычисляем энтропию распределения степеней
    def calculate_degree_entropy(graph):
        if graph.number_of_nodes() == 0:
            return 0.0
        
        degrees = [deg for _, deg in graph.degree()]
        total_degrees = sum(degrees)
        if total_degrees == 0:
            return 0.0
            
        # Нормализуем степени для получения распределения вероятностей
        degree_probs = [deg / total_degrees for deg in degrees]
        
        # Вычисляем энтропию Шеннона
        entropy = 0.0
        for prob in degree_probs:
            if prob > 0:  # Избегаем log(0)
                entropy -= prob * math.log2(prob)
        
        return entropy

    # Нормализованная энтропия (относительно максимально возможной)
    def calculate_normalized_entropy(graph):
        entropy = calculate_degree_entropy(graph)
        max_entropy = math.log2(graph.number_of_nodes()) if graph.number_of_nodes() > 0 else 0
        return entropy / max_entropy if max_entropy > 0 else 0

    # Основные метрики сети
    degree_entropy = calculate_degree_entropy(G)
    normalized_entropy = calculate_normalized_entropy(G)
    
    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": float(nx.density(G)),
        "avg_clustering": float(nx.average_clustering(G)) if G.number_of_nodes() > 0 else 0.0,
        "degree_entropy": float(degree_entropy),
        "normalized_entropy": float(normalized_entropy),
        "avg_degree": float(sum(dict(G.degree()).values()) / G.number_of_nodes()) if G.number_of_nodes() > 0 else 0.0,
        "assortativity": float(nx.degree_assortativity_coefficient(G)) if G.number_of_nodes() > 1 else 0.0
    }
    
    logger.info(f"Сеть паттернов для '{theme}': {metrics}")

    # Приводим типы
    for n, data in list(G.nodes(data=True)):
        for k, v in list(data.items()):
            data[k] = cast_numpy_types(v)
    for u, v, data in list(G.edges(data=True)):
        for k, val in list(data.items()):
            data[k] = cast_numpy_types(val)

    # Создаём визуализацию pyvis
    net = Network(height="900px", width="100%", bgcolor="#ffffff", notebook=False)
    net.from_nx(G)
    html_path = Path(CONFIG["storage_dir"]) / f"patterns_{theme}.html"
    net.save_graph(str(html_path))

    # ИСПРАВЛЕННЫЙ JavaScript код
    js_script = f"""
    <script type="text/javascript">
    function enhanceNetwork(network, nodes, edges) {{
        // controls panel
        const panel = document.createElement('div');
        panel.style.position = 'fixed';
        panel.style.top = '10px';
        panel.style.left = '10px';
        panel.style.zIndex = 9999;
        panel.style.background = 'rgba(255,255,255,0.96)';
        panel.style.border = '1px solid #ddd';
        panel.style.padding = '10px';
        panel.style.borderRadius = '8px';
        panel.style.fontFamily = 'sans-serif';
        panel.innerHTML = `
            <b>Тема: {theme}</b><br>
            <small>Узлов: {metrics['nodes']}, Рёбер: {metrics['edges']}</small><br>
            <small>Плотность: {metrics['density']:.3f}, Кластеризация: {metrics['avg_clustering']:.3f}</small><br>
            <small>Энтропия: {metrics['degree_entropy']:.3f} (норм.: {metrics['normalized_entropy']:.3f})</small><br>
            <hr style="margin:6px 0;">
            <input type="text" id="patternSearch" placeholder="Поиск паттерна..." style="width:200px;">
            <button id="searchBtn">Найти</button>
            <button id="resetBtn">Сброс</button>
            <hr style="margin:6px 0;">
            Мин. похожесть: <input type="range" id="weightSlider" min="0" max="1" step="0.01" value="{CONFIG['pattern_similarity_threshold']}">
            <span id="weightVal">{CONFIG['pattern_similarity_threshold']:.2f}</span>
        `;
        document.body.appendChild(panel);

        const infoBox = document.createElement('div');
        infoBox.style.position = 'fixed';
        infoBox.style.bottom = '14px';
        infoBox.style.right = '14px';
        infoBox.style.zIndex = 9999;
        infoBox.style.background = 'rgba(255,255,255,0.95)';
        infoBox.style.border = '1px solid #ddd';
        infoBox.style.padding = '10px';
        infoBox.style.maxWidth = '420px';
        infoBox.style.maxHeight = '320px';
        infoBox.style.overflow = 'auto';
        infoBox.style.borderRadius = '8px';
        infoBox.style.display = 'none';
        document.body.appendChild(infoBox);

        // поиск
        document.getElementById('searchBtn').addEventListener('click', function() {{
            const q = document.getElementById('patternSearch').value.trim().toLowerCase();
            if (!q) return;
            const found = nodes.get({{filter: n => n.label.toLowerCase().includes(q) }});
            if (found.length === 0) {{
                alert('Не найдено');
                return;
            }}
            const ids = found.map(x => x.id);
            network.selectNodes(ids);
            network.fit({{nodes: ids, animation: true}});
        }});

        document.getElementById('resetBtn').addEventListener('click', function() {{
            document.getElementById('patternSearch').value = '';
            // restore nodes
            const allNodes = nodes.get();
            for (let n of allNodes) {{
                n.color = undefined;
            }}
            nodes.update(allNodes);
            // restore edges
            const allEdges = edges.get();
            for (let e of allEdges) {{
                e.hidden = false;
                e.color = undefined;
            }}
            edges.update(allEdges);
            infoBox.style.display = 'none';
            network.fit();
        }});

        // weight filter
        const slider = document.getElementById('weightSlider');
        const weightVal = document.getElementById('weightVal');
        slider.addEventListener('input', function() {{
            const minW = parseFloat(slider.value);
            weightVal.textContent = minW.toFixed(2);
            const allEdges = edges.get();
            for (let e of allEdges) {{
                e.hidden = e.weight < minW;
            }}
            edges.update(allEdges);
        }});

        // click: highlight neighbors and edges
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const id = params.nodes[0];
                const connected = network.getConnectedNodes(id);
                const allNodes = nodes.get();
                for (let n of allNodes) {{
                    n.color = (n.id === id || connected.includes(n.id)) ? '#ff9800' : '#d3d3d3';
                }}
                nodes.update(allNodes);
                // color edges: highlight edges adjacent to id
                const allEdges = edges.get();
                for (let e of allEdges) {{
                    if (e.from === id || e.to === id) {{
                        e.color = '#ff9800';
                        e.width = Math.max(1.5, e.weight*5);
                        e.hidden = false;
                    }} else {{
                        e.color = '#cccccc';
                    }}
                }}
                edges.update(allEdges);
            }} else {{
                // background click: reset colors
                const allNodes = nodes.get();
                for (let n of allNodes) n.color = undefined;
                nodes.update(allNodes);
                const allEdges = edges.get();
                for (let e of allEdges) e.color = undefined;
                edges.update(allEdges);
            }}
        }});

        // doubleClick: show constituents (words, count, doc_freq, pmi)
        network.on('doubleClick', function(params) {{
            if (params.nodes.length > 0) {{
                const id = params.nodes[0];
                const node = nodes.get(id);  // ОПРЕДЕЛЯЕМ node здесь!
                if (!node) return;
                // assemble metadata string
                let meta = `<b>Паттерн:</b> ${{node.label}}<br>`;
                if (node.words) meta += `<b>Слова:</b> ${{node.words.join(', ')}}<br>`;
                if (node.count!==undefined) meta += `<b>count:</b> ${{node.count}} `;
                if (node.doc_freq!==undefined) meta += `<b>doc_freq:</b> ${{node.doc_freq}} `;
                if (node.pmi!==undefined) meta += `<b>pmi:</b> ${{parseFloat(node.pmi).toFixed(2)}} `;
                if (node.quality_score!==undefined) meta += `<b>quality:</b> ${{parseFloat(node.quality_score).toFixed(2)}} `;
                if (node.semantic_similarity!==undefined) meta += `<b>semantic:</b> ${{parseFloat(node.semantic_similarity).toFixed(2)}} `;
                if (node.has_keyword!==undefined) meta += `<b>has keyword:</b> ${{node.has_keyword}} `;
                infoBox.innerHTML = meta;
                infoBox.style.display = 'block';
            }}
        }});

        // ensure edges are visible initially
        const allEdgesInit = edges.get();
        for (let e of allEdgesInit) e.hidden = false;
        edges.update(allEdgesInit);
    }}

    document.addEventListener('DOMContentLoaded', function() {{
        enhanceNetwork(network, nodes, edges);
    }});
    </script>
    """

    # inject JS into saved HTML
    text = html_path.read_text(encoding="utf-8")
    if "</body>" in text:
        text = text.replace("</body>", js_script + "\n</body>")
    else:
        text = text + js_script
    html_path.write_text(text, encoding="utf-8")
    logger.info(f"Интерактивный граф сохранён: {html_path}")

    # сохраняем patterns metadata (json)
    meta_path = Path(CONFIG["storage_dir"]) / f"patterns_{theme}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)

    return G, metrics

def save_theme_artifacts(theme: str, patterns: List[Dict], 
                        metrics: Dict, keywords: List[str]):
    """Сохраняет все артефакты для темы"""
    # Метрики
    metrics_path = Path(CONFIG["storage_dir"]) / f"metrics_{theme}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(cast_numpy_types(metrics), f, ensure_ascii=False, indent=2)
    
    # Ключевые слова
    keywords_path = Path(CONFIG["storage_dir"]) / f"keywords_{theme}.json"
    with open(keywords_path, "w", encoding="utf-8") as f:
        json.dump({
            "theme": theme,
            "keywords": keywords,
            "keywords_count": len(keywords)
        }, f, ensure_ascii=False, indent=2)
    
    # Паттерны с метаданными
    patterns_path = Path(CONFIG["storage_dir"]) / f"patterns_{theme}.json"
    with open(patterns_path, "w", encoding="utf-8") as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)

# ----------------------------
# 5. Главная логика
# ----------------------------
def main():
    # Собираем корпус
    corpus_by_theme = collect_corpus_by_theme(themes)
    
    # Автоматическое извлечение ключевых слов
    auto_keywords = extract_theme_keywords_automatically(corpus_by_theme)
    
    for theme in themes:
        texts = corpus_by_theme.get(theme, [])
        if not texts:
            logger.warning(f"Нет текстов для темы '{theme}' — пропускаем")
            continue

        # Лемматизация текстов
        logger.info(f"Лемматизация текстов для темы '{theme}'...")
        lemmatized = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"Лемматизировано {i}/{len(texts)} текстов")
            lemmatized.append(lemmatize_text(text, CONFIG["nlp"]["pos_filter"]))
        
        # Получаем автоматические ключевые слова для темы
        theme_keywords = auto_keywords.get(theme, [])
        logger.info(f"Тема '{theme}': {len(theme_keywords)} автоматических ключевых слов")

        # Комплексное извлечение качественных паттернов
        patterns = extract_high_quality_patterns(
            lemmatized_texts=lemmatized,
            theme=theme,
            min_pmi=CONFIG["pattern_extraction"]["min_pmi"],
            min_doc_freq=CONFIG["pattern_extraction"]["min_doc_freq"],
            min_count=CONFIG["pattern_extraction"]["min_count"],
            ngram_range=CONFIG["pattern_extraction"]["ngram_range"],
            theme_keywords=theme_keywords
        )

        if not patterns:
            logger.warning(f"Для темы '{theme}' не осталось качественных паттернов.")
            continue

        # Строим сеть
        G, metrics = build_pattern_network_for_theme(patterns, theme)
        
        # Сохраняем метрики и ключевые слова
        save_theme_artifacts(theme, patterns, metrics, theme_keywords)

if __name__ == "__main__":
    main()
