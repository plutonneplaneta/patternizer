# -*- coding: utf-8 -*-
"""
Извлечение паттернов из корпуса НКРЯ.
Парсинг файлов, лемматизация, извлечение n-грамм с PMI,
фильтрация шума, построение сети паттернов.

Версия 3.1 - с улучшенной кластеризацией (автоподбор eps)
"""

import colorsys
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import networkx as nx
import numpy as np
from natasha import (
    Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc,
    NewsNERTagger, NamesExtractor
)
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.neighbors import NearestNeighbors

# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Конфигурация программы."""
    # Пути
    input_dir: str = "./data_nkrya"
    output_dir: str = "./output"
    
    # Параметры извлечения паттернов
    min_pmi: float = 3.0
    min_doc_freq: int = 5
    min_count: int = 5
    ngram_range: Tuple[int, int] = (2, 4)
    max_patterns_per_theme: int = 1500
    filter_subsumed: bool = True
    
    # NLP
    filter_ner: bool = True
    pos_filter: List[str] = field(default_factory=lambda: ["NOUN", "ADJ", "VERB", "ADV"])
    
    # Эмбеддинги и сеть
    embedding_model: str = "cointegrated/rubert-tiny2"
    embedding_batch_size: int = 32
    similarity_threshold: float = 0.75
    
    # Кластеризация
    clustering_method: str = "dbscan"  # "dbscan" или "kmeans"
    dbscan_eps: float = 0.0  # 0 = автоподбор
    dbscan_min_samples: int = 3  # Уменьшил для лучшей кластеризации
    kmeans_n_clusters: int = 10
    min_clusters: int = 3  # Минимальное желаемое число кластеров
    max_clusters: int = 20  # Максимальное число кластеров
    
    # Кэширование
    use_cache: bool = True


CONFIG = Config()

# ═══════════════════════════════════════════════════════════════════════════════
# ЧЁРНЫЕ СПИСКИ (только метаданные блогов)
# ═══════════════════════════════════════════════════════════════════════════════

BLACKLIST_WORDS = {
    # Соцсети и блоги
    "блог", "блогер", "пост", "лайк", "репост", "подписчик", "подписка",
    "инстаграм", "телеграм", "вконтакте", "фейсбук", "ютуб", "тикток",
    "канал", "чат", "группа", "паблик", "сторис", "рилс", "хештег",
    # Профессии из подписей
    "фотограф", "стилист", "визажист", "блогер", "коуч", "психолог",
    "тренер", "эксперт", "специалист", "мастер", "автор", "фотографворонеж"
    # Города
    "москва", "петербург", "воронеж", "екатеринбург", "казань", "нижний",
    "новгород", "самара", "омск", "челябинск", "ростов", "краснодар",
    "новосибирск", "красноярск", "пермь", "волгоград", "уфа", "саратов",
    # Общие шумовые
    "реклама", "сотрудничество", "контакт", "ссылка", "профиль",
    "druzhinina", "instagram", "telegram", "youtube", "vk", "tiktok",
    # Дни недели, месяцы
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота",
    "воскресенье", "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
    # Латиница
    "ru", "com", "www", "http", "https"
}


# ═══════════════════════════════════════════════════════════════════════════════
# СТРУКТУРЫ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NKRYAMetadata:
    """Метаданные файла НКРЯ."""
    request_url: str = ""
    corpus_name: str = ""
    file_saved: str = ""
    documents_found: int = 0
    documents_downloaded: int = 0
    examples_found: int = 0
    examples_downloaded: int = 0
    sort_by: str = ""
    request: str = ""


@dataclass
class NKRYAExample:
    """Пример из корпуса НКРЯ."""
    text: str
    source: str
    source_id: str = ""
    date: str = ""
    reference: str = ""


@dataclass
class Pattern:
    """Извлечённый паттерн."""
    ngram: str
    count: int
    doc_freq: int
    pmi: float
    length: int
    words: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# ПАРСЕР ФАЙЛОВ НКРЯ
# ═══════════════════════════════════════════════════════════════════════════════

class NKRYAParser:
    """Парсер файлов формата НКРЯ с улучшенной очисткой."""
    
    META_PATTERN = re.compile(r"^([^:]+):\s*(.+)$")
    EXAMPLE_NUM_PATTERN = re.compile(r"^(\d+)\.\s+(.+)$")
    NEW_FORMAT_HEADER = re.compile(r"^(\d+)\.\s+(.+?)\s*\((\d{2}\.\d{2}\.\d{4}|\d{4})\)\s*$")
    REFERENCE_PATTERN = re.compile(r"\s*\[[^\]]+\]\s*$")
    SOURCE_HEADER_PATTERN = re.compile(r'^(.+?)\s*\((\d{2}\.\d{2}\.\d{4}|\d{4})\)\s*$')
    HASHTAG_PATTERN = re.compile(r'#\S+')
    EMOJI_PATTERN = re.compile(
        r'[\U0001F600-\U0001F64F'
        r'\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF'
        r'\U00002702-\U000027B0'
        r'\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F'
        r'\U0001FA70-\U0001FAFF'
        r'\U00002600-\U000026FF'
        r'\U00002700-\U000027BF'
        r'\U0001F000-\U0001F02F'
        r'\U0001F0A0-\U0001F0FF'
        r']+', flags=re.UNICODE
    )
    
    def parse_file(self, filepath: Path) -> Tuple[NKRYAMetadata, List[NKRYAExample]]:
        """Парсит файл НКРЯ."""
        raw = filepath.read_text(encoding="utf-8")
        return self.parse_text(raw)
    
    def parse_text(self, raw: str) -> Tuple[NKRYAMetadata, List[NKRYAExample]]:
        """Парсит текст в формате НКРЯ."""
        lines = raw.split("\n")
        metadata = NKRYAMetadata()
        examples = []
        
        in_examples = False
        current_source = ""
        current_date = ""
        pending_text_lines = []
        
        def save_pending():
            nonlocal pending_text_lines
            if pending_text_lines:
                full_text = ' '.join(pending_text_lines)
                example = self._parse_example(full_text, current_source, current_date)
                if example.text:
                    examples.append(example)
                pending_text_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            i += 1
            
            if stripped.lower() == "examples":
                in_examples = True
                continue
            
            if not in_examples:
                if stripped:
                    self._parse_metadata_line(stripped, metadata)
                continue
            
            if not stripped:
                save_pending()
                continue
            
            new_format_match = self.NEW_FORMAT_HEADER.match(stripped)
            if new_format_match:
                save_pending()
                current_source = new_format_match.group(2).strip()
                current_date = new_format_match.group(3).strip()
                continue
            
            num_match = self.EXAMPLE_NUM_PATTERN.match(stripped)
            if num_match:
                example_text = num_match.group(2).strip()
                
                source_check = self.SOURCE_HEADER_PATTERN.match(example_text)
                if source_check:
                    save_pending()
                    current_source = source_check.group(1).strip()
                    current_date = source_check.group(2).strip()
                    continue
                
                save_pending()
                
                if self.REFERENCE_PATTERN.search(example_text):
                    ref_match = self.REFERENCE_PATTERN.search(example_text)
                    if ref_match and not current_source:
                        ref_text = ref_match.group(0)
                        source_in_ref = self._extract_source_from_reference(ref_text)
                        if source_in_ref:
                            current_source, current_date = source_in_ref
                
                example = self._parse_example(example_text, current_source, current_date)
                if example.text:
                    examples.append(example)
                continue
            
            source_match = self.SOURCE_HEADER_PATTERN.match(stripped)
            if source_match:
                save_pending()
                current_source = source_match.group(1).strip()
                current_date = source_match.group(2).strip()
                continue
            
            alt_match = re.search(r'\((\d{2}\.\d{2}\.\d{4}|\d{4})\)\s*$', stripped)
            if alt_match:
                next_idx = i
                next_line = ""
                while next_idx < len(lines):
                    next_line = lines[next_idx].strip()
                    if next_line:
                        break
                    next_idx += 1
                
                if not next_line or self.EXAMPLE_NUM_PATTERN.match(next_line) or \
                   self.NEW_FORMAT_HEADER.match(next_line) or \
                   self.SOURCE_HEADER_PATTERN.match(next_line):
                    save_pending()
                    current_source = stripped[:alt_match.start()].strip()
                    current_date = alt_match.group(1)
                    continue
            
            pending_text_lines.append(stripped)
        
        save_pending()
        
        logger.info(f"Распарсено примеров: {len(examples)}")
        return metadata, examples
    
    def _extract_source_from_reference(self, ref_text: str) -> Optional[Tuple[str, str]]:
        """Извлекает источник и дату из ссылки."""
        match = re.search(r'\[(.+?)\s*\((\d{2}\.\d{2}\.\d{4}|\d{4})\)', ref_text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None
    
    def _parse_metadata_line(self, line: str, metadata: NKRYAMetadata):
        """Парсит строку метаданных."""
        match = self.META_PATTERN.match(line)
        if not match:
            return
        
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        
        key_mapping = {
            "request url": "request_url",
            "corpus name": "corpus_name",
            "file saved": "file_saved",
            "documents found": "documents_found",
            "documents with examples downloaded": "documents_downloaded",
            "examples found": "examples_found",
            "examples downloaded": "examples_downloaded",
            "sort by": "sort_by",
            "request": "request"
        }
        
        attr = key_mapping.get(key)
        if attr:
            if attr in ["documents_found", "documents_downloaded", "examples_found", "examples_downloaded"]:
                try:
                    value = int(value)
                except ValueError:
                    value = 0
            setattr(metadata, attr, value)
    
    def _parse_example(self, text: str, source: str, date: str) -> NKRYAExample:
        """Парсит и очищает текст примера."""
        reference = ""
        
        ref_match = self.REFERENCE_PATTERN.search(text)
        if ref_match:
            reference = ref_match.group(0).strip()
            text = text[:ref_match.start()].strip()
        
        text = self._clean_text(text)
        
        return NKRYAExample(
            text=text,
            source=source,
            date=date,
            reference=reference
        )
    
    def _clean_text(self, text: str) -> str:
        """Тщательная очистка текста."""
        text = self.EMOJI_PATTERN.sub(' ', text)
        text = self.HASHTAG_PATTERN.sub(' ', text)
        text = text.replace('⠀', ' ').replace('\u200b', ' ').replace('\xa0', ' ')
        text = re.sub(r'https?://\S+', ' ', text)
        text = re.sub(r'www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+\.\S+', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'\[[^\]]*\]', ' ', text)
        text = re.sub(r'\b[a-zA-Z]+\b', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text.split()) < 3:
            return ""
        
        return text


# ═══════════════════════════════════════════════════════════════════════════════
# NLP: ЛЕММАТИЗАЦИЯ С NER-ФИЛЬТРАЦИЕЙ
# ═══════════════════════════════════════════════════════════════════════════════

class Lemmatizer:
    """Лемматизатор с NER-фильтрацией."""
    
    def __init__(self):
        self._segmenter = None
        self._morph_vocab = None
        self._morph_tagger = None
        self._ner_tagger = None
        self._embeddings = None
        self._names_extractor = None
    
    def _init_components(self):
        """Ленивая инициализация."""
        if self._segmenter is None:
            logger.info("Инициализация NLP компонентов...")
            self._segmenter = Segmenter()
            self._embeddings = NewsEmbedding()
            self._morph_vocab = MorphVocab()
            self._morph_tagger = NewsMorphTagger(self._embeddings)
            self._ner_tagger = NewsNERTagger(self._embeddings)
            self._names_extractor = NamesExtractor(self._morph_vocab)
    
    def lemmatize(self, text: str, pos_filter: List[str], filter_ner: bool = True) -> str:
        """Лемматизирует текст, фильтруя NER-сущности."""
        self._init_components()
        
        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)
        
        ner_spans = set()
        if filter_ner:
            doc.tag_ner(self._ner_tagger)
            for span in doc.spans:
                if span.type in ('PER', 'LOC', 'ORG'):
                    for i in range(span.start, span.stop):
                        ner_spans.add(i)
        
        lemmas = []
        for idx, token in enumerate(doc.tokens):
            if idx in ner_spans:
                continue
            
            token.lemmatize(self._morph_vocab)
            
            if token.pos not in pos_filter:
                continue
            
            lemma = token.lemma.lower()
            
            if len(lemma) < 2:
                continue
            if lemma in BLACKLIST_WORDS:
                continue
            if re.match(r'^\d+$', lemma):
                continue
            
            lemmas.append(lemma)
        
        return " ".join(lemmas)
    
    def lemmatize_batch(self, texts: List[str], pos_filter: List[str], 
                        filter_ner: bool = True) -> List[str]:
        """Batch-лемматизация с прогрессом."""
        result = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if i > 0 and i % 100 == 0:
                logger.info(f"Лемматизировано {i}/{total} ({100*i/total:.1f}%)")
            result.append(self.lemmatize(text, pos_filter, filter_ner))
        
        logger.info(f"Лемматизация завершена: {total} текстов")
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# ИЗВЛЕЧЕНИЕ ПАТТЕРНОВ
# ═══════════════════════════════════════════════════════════════════════════════

class PatternExtractor:
    """Извлекает паттерны из лемматизированных текстов."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def extract(self, lemmatized_texts: List[str]) -> List[Pattern]:
        """Извлекает паттерны."""
        if not lemmatized_texts:
            return []
        
        lemmatized_texts = [t for t in lemmatized_texts if t.strip()]
        if not lemmatized_texts:
            return []
        
        logger.info(f"Извлечение n-грамм из {len(lemmatized_texts)} текстов...")
        
        vectorizer = CountVectorizer(
            ngram_range=self.config.ngram_range,
            token_pattern=r"(?u)\b[а-яёА-ЯЁ]{3,}\b"
        )
        
        try:
            X = vectorizer.fit_transform(lemmatized_texts)
        except ValueError as e:
            logger.warning(f"Ошибка векторизации: {e}")
            return []
        
        feature_names = vectorizer.get_feature_names_out()
        counts = X.sum(axis=0).A1
        doc_freq = (X > 0).sum(axis=0).A1
        
        all_words = [w for text in lemmatized_texts for w in text.split()]
        total_words = len(all_words)
        word_counts = Counter(all_words)
        total_ngrams = counts.sum() if counts.sum() > 0 else 1
        
        patterns = []
        
        for i, ngram in enumerate(feature_names):
            count = int(counts[i])
            df = int(doc_freq[i])
            
            if count < self.config.min_count or df < self.config.min_doc_freq:
                continue
            
            words = ngram.split()
            
            if any(len(w) < 2 for w in words):
                continue
            
            if any(w in BLACKLIST_WORDS for w in words):
                continue
            
            if len(set(words)) < len(words):
                continue
            
            pmi = self._calculate_pmi(ngram, count, total_ngrams, word_counts, total_words)
            if pmi is None or pmi < self.config.min_pmi:
                continue
            
            patterns.append(Pattern(
                ngram=ngram,
                count=count,
                doc_freq=df,
                pmi=pmi,
                length=len(words),
                words=words
            ))
        
        patterns.sort(key=lambda p: (p.pmi, p.doc_freq), reverse=True)
        
        logger.info(f"Извлечено {len(patterns)} паттернов")
        
        if self.config.filter_subsumed:
            patterns = self._filter_subsumed(patterns)
        
        if len(patterns) > self.config.max_patterns_per_theme:
            patterns = patterns[:self.config.max_patterns_per_theme]
            logger.info(f"Ограничено до {self.config.max_patterns_per_theme}")
        
        return patterns
    
    def _calculate_pmi(self, ngram: str, count: int, total_ngrams: int,
                       word_counts: Counter, total_words: int) -> Optional[float]:
        """Вычисляет PMI."""
        words = ngram.split()
        p_ngram = count / total_ngrams
        
        p_words = 1.0
        for w in words:
            wc = word_counts.get(w, 0)
            if wc == 0:
                return None
            p_words *= (wc / total_words)
        
        if p_words == 0:
            return None
        
        return math.log2(p_ngram / p_words)
    
    def _filter_subsumed(self, patterns: List[Pattern]) -> List[Pattern]:
        """Фильтрует вложенные n-граммы."""
        if len(patterns) <= 1:
            return patterns
        
        sorted_patterns = sorted(patterns, key=lambda p: (p.length, p.pmi), reverse=True)
        
        kept = []
        kept_ngrams = set()
        
        for pattern in sorted_patterns:
            words = pattern.words
            is_subsumed = False
            
            for longer_ngram in kept_ngrams:
                longer_words = longer_ngram.split()
                if len(words) >= len(longer_words):
                    continue
                
                for start in range(len(longer_words) - len(words) + 1):
                    if longer_words[start:start + len(words)] == words:
                        is_subsumed = True
                        break
                
                if is_subsumed:
                    break
            
            if not is_subsumed:
                kept.append(pattern)
                kept_ngrams.add(pattern.ngram)
        
        logger.info(f"Фильтрация вложенных: {len(patterns)} -> {len(kept)}")
        return kept


# ═══════════════════════════════════════════════════════════════════════════════
# КЛАСТЕРИЗАЦИЯ ПАТТЕРНОВ (УЛУЧШЕННАЯ)
# ═══════════════════════════════════════════════════════════════════════════════

class PatternClusterer:
    """Кластеризация паттернов с автоподбором параметров."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def cluster(self, patterns: List[Pattern], embeddings: np.ndarray) -> Dict:
        """Кластеризует паттерны с автоподбором параметров."""
        if len(patterns) < 2:
            return self._empty_result(len(patterns))
        
        method = self.config.clustering_method.lower()
        
        if method == "dbscan":
            labels = self._cluster_dbscan_auto(embeddings)
        elif method == "kmeans":
            labels = self._cluster_kmeans_auto(embeddings)
        else:
            logger.warning(f"Неизвестный метод: {method}, используем DBSCAN")
            labels = self._cluster_dbscan_auto(embeddings)
        
        return self._build_result(patterns, embeddings, labels)
    
    def _find_optimal_eps(self, embeddings: np.ndarray) -> float:
        """Находит оптимальный eps методом k-расстояний."""
        n_samples = len(embeddings)
        k = min(self.config.dbscan_min_samples, n_samples - 1)
        
        if k < 1:
            return 0.5
        
        # Вычисляем косинусные расстояния
        distances = cosine_distances(embeddings)
        
        # Для каждой точки находим k-е ближайшее расстояние
        k_distances = []
        for i in range(n_samples):
            sorted_dists = np.sort(distances[i])
            if k < len(sorted_dists):
                k_distances.append(sorted_dists[k])
        
        k_distances = np.array(sorted(k_distances))
        
        # Метод "локтя" - ищем точку максимальной кривизны
        # Упрощённый подход: берём значение на определённом перцентиле
        # Обычно хорошо работает 90-й перцентиль
        
        # Также можно использовать производную
        if len(k_distances) > 10:
            # Вычисляем вторую производную для поиска "колена"
            first_derivative = np.diff(k_distances)
            second_derivative = np.diff(first_derivative)
            
            # Точка перегиба - максимум второй производной
            if len(second_derivative) > 0:
                elbow_idx = np.argmax(second_derivative) + 1
                eps = k_distances[min(elbow_idx, len(k_distances) - 1)]
            else:
                eps = np.percentile(k_distances, 80)
        else:
            eps = np.percentile(k_distances, 80)
        
        # Ограничиваем разумными пределами для косинусного расстояния
        eps = max(0.1, min(0.8, eps))
        
        return eps
    
    def _cluster_dbscan_auto(self, embeddings: np.ndarray) -> np.ndarray:
        """DBSCAN с автоподбором eps."""
        
        if self.config.dbscan_eps > 0:
            eps = self.config.dbscan_eps
            logger.info(f"DBSCAN: eps={eps} (заданный)")
        else:
            eps = self._find_optimal_eps(embeddings)
            logger.info(f"DBSCAN: eps={eps:.3f} (автоподбор)")
        
        min_samples = self.config.dbscan_min_samples
        
        # Пробуем кластеризацию
        best_labels = None
        best_n_clusters = 0
        best_eps = eps
        
        # Попробуем несколько значений eps вокруг найденного
        eps_values = [eps * 0.7, eps * 0.85, eps, eps * 1.15, eps * 1.3]
        
        for test_eps in eps_values:
            clustering = DBSCAN(
                eps=test_eps,
                min_samples=min_samples,
                metric='cosine'
            )
            labels = clustering.fit_predict(embeddings)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels)
            
            # Критерии хорошей кластеризации:
            # 1. Достаточно кластеров
            # 2. Не слишком много шума
            if n_clusters >= self.config.min_clusters and noise_ratio < 0.5:
                if n_clusters > best_n_clusters:
                    best_labels = labels
                    best_n_clusters = n_clusters
                    best_eps = test_eps
        
        # Если не нашли хорошую кластеризацию, используем K-means как fallback
        if best_labels is None or best_n_clusters < self.config.min_clusters:
            logger.info(f"DBSCAN не дал достаточно кластеров, переключаемся на K-means")
            return self._cluster_kmeans_auto(embeddings)
        
        n_noise = list(best_labels).count(-1)
        logger.info(f"DBSCAN (eps={best_eps:.3f}): {best_n_clusters} кластеров, {n_noise} шумовых точек")
        
        return best_labels
    
    def _cluster_kmeans_auto(self, embeddings: np.ndarray) -> np.ndarray:
        """K-means с автоподбором числа кластеров через silhouette."""
        n_samples = len(embeddings)
        
        min_k = max(2, self.config.min_clusters)
        max_k = min(self.config.max_clusters, n_samples // 3, n_samples - 1)
        max_k = max(min_k, max_k)
        
        best_k = min_k
        best_score = -1
        best_labels = None
        
        logger.info(f"K-means: поиск оптимального k в диапазоне [{min_k}, {max_k}]")
        
        for k in range(min_k, max_k + 1):
            clustering = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = clustering.fit_predict(embeddings)
            
            try:
                score = silhouette_score(embeddings, labels, metric='cosine')
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
            except:
                continue
        
        if best_labels is None:
            clustering = KMeans(n_clusters=min_k, random_state=42, n_init=10)
            best_labels = clustering.fit_predict(embeddings)
            best_k = min_k
        
        logger.info(f"K-means: {best_k} кластеров (silhouette={best_score:.3f})")
        
        return best_labels
    
    def _build_result(self, patterns: List[Pattern], embeddings: np.ndarray, 
                      labels: np.ndarray) -> Dict:
        """Формирует результат кластеризации."""
        
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            clusters.setdefault(int(label), []).append(idx)
        
        cluster_info = {}
        for cluster_id, indices in clusters.items():
            cluster_patterns = [patterns[i] for i in indices]
            cluster_embeddings = embeddings[indices]
            
            centroid = cluster_embeddings.mean(axis=0)
            
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            representative_idx = indices[np.argmin(distances)]
            
            avg_pmi = np.mean([p.pmi for p in cluster_patterns])
            
            sorted_by_pmi = sorted(
                [(i, patterns[i]) for i in indices],
                key=lambda x: x[1].pmi,
                reverse=True
            )[:5]
            
            cluster_info[cluster_id] = {
                "size": len(indices),
                "pattern_indices": indices,
                "representative": patterns[representative_idx].ngram,
                "representative_idx": representative_idx,
                "avg_pmi": round(avg_pmi, 3),
                "top_patterns": [
                    {"ngram": p.ngram, "pmi": round(p.pmi, 3)}
                    for _, p in sorted_by_pmi
                ],
                "all_patterns": [patterns[i].ngram for i in indices]
            }
        
        metrics = self._calculate_clustering_metrics(embeddings, labels)
        
        return {
            "labels": labels.tolist(),
            "clusters": {k: v for k, v in clusters.items()},
            "cluster_info": cluster_info,
            "metrics": metrics,
            "n_clusters": len(clusters),
            "n_noise": int(list(labels).count(-1)),
            "method": self.config.clustering_method
        }
    
    def _calculate_clustering_metrics(self, embeddings: np.ndarray, 
                                       labels: np.ndarray) -> Dict:
        """Вычисляет метрики качества кластеризации."""
        metrics = {}
        
        mask = labels != -1
        valid_labels = labels[mask]
        valid_embeddings = embeddings[mask]
        
        n_clusters = len(set(valid_labels))
        
        if n_clusters >= 2 and len(valid_labels) >= 2:
            try:
                silhouette = silhouette_score(valid_embeddings, valid_labels, metric='cosine')
                metrics["silhouette_score"] = round(silhouette, 4)
            except Exception as e:
                logger.warning(f"Ошибка расчёта silhouette: {e}")
                metrics["silhouette_score"] = None
        else:
            metrics["silhouette_score"] = None
        
        cluster_sizes = [list(valid_labels).count(i) for i in set(valid_labels)]
        if cluster_sizes:
            metrics["cluster_size_mean"] = round(np.mean(cluster_sizes), 2)
            metrics["cluster_size_std"] = round(np.std(cluster_sizes), 2)
            metrics["cluster_size_min"] = min(cluster_sizes)
            metrics["cluster_size_max"] = max(cluster_sizes)
        
        return metrics
    
    def _empty_result(self, n_patterns: int) -> Dict:
        """Пустой результат для недостаточного количества паттернов."""
        return {
            "labels": [0] * n_patterns,
            "clusters": {0: list(range(n_patterns))} if n_patterns > 0 else {},
            "cluster_info": {},
            "metrics": {},
            "n_clusters": 1 if n_patterns > 0 else 0,
            "n_noise": 0,
            "method": self.config.clustering_method
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ПОСТРОЕНИЕ СЕТИ ПАТТЕРНОВ
# ═══════════════════════════════════════════════════════════════════════════════

class PatternNetworkBuilder:
    """Строит сеть паттернов с расширенными метриками централизации."""
    
    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self.clusterer = PatternClusterer(config)
    
    def _init_model(self):
        if self._model is None:
            logger.info(f"Загрузка модели: {self.config.embedding_model}")
            self._model = SentenceTransformer(self.config.embedding_model)
    
    def build(self, patterns: List[Pattern], theme: str) -> Tuple[nx.Graph, Dict, Dict]:
        """Строит сеть."""
        if not patterns:
            return nx.Graph(), {}, {}
        
        self._init_model()
        
        ngrams = [p.ngram for p in patterns]
        logger.info(f"Эмбеддинги для {len(ngrams)} паттернов...")
        embeddings = self._model.encode(
            ngrams,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=True
        )
        
        # Кластеризация
        clustering_result = self.clusterer.cluster(patterns, embeddings)
        
        G = nx.Graph()
        
        # Узлы с информацией о кластере
        for idx, pattern in enumerate(patterns):
            cluster_id = clustering_result["labels"][idx]
            size = max(8, min(40, 10 + pattern.pmi * 2 + pattern.doc_freq * 0.5))
            
            G.add_node(
                idx,
                label=pattern.ngram,
                words=pattern.words,
                count=pattern.count,
                doc_freq=pattern.doc_freq,
                pmi=round(pattern.pmi, 3),
                length=pattern.length,
                cluster=cluster_id,
                title=f"{pattern.ngram}\nPMI: {pattern.pmi:.2f}\nКластер: {cluster_id}",
                size=size
            )
        
        # Рёбра
        edge_count = 0
        threshold = self.config.similarity_threshold
        
        for i in range(len(ngrams)):
            for j in range(i + 1, len(ngrams)):
                sim = float(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0])
                if sim >= threshold:
                    G.add_edge(i, j, weight=round(sim, 3), title=f"Sim: {sim:.3f}",
                              width=1 + (sim - threshold) * 4)
                    edge_count += 1
        
        logger.info(f"Рёбер: {edge_count}, узлов: {len(patterns)}")
        
        # Метрики с betweenness centrality
        metrics = self._calculate_metrics(G, theme, patterns)
        
        return G, metrics, clustering_result
    
    def _calculate_metrics(self, G: nx.Graph, theme: str, patterns: List[Pattern]) -> Dict:
        """Вычисляет метрики сети включая betweenness centrality."""
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        if n_nodes == 0:
            return {"theme": theme, "nodes": 0, "edges": 0}
        
        degrees = [d for _, d in G.degree()]
        total_degree = sum(degrees)
        
        # Энтропия степеней
        if total_degree > 0:
            probs = [d / total_degree for d in degrees]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(n_nodes) if n_nodes > 1 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            entropy = normalized_entropy = 0
        
        metrics = {
            "theme": theme,
            "nodes": n_nodes,
            "edges": n_edges,
            "density": round(nx.density(G), 4),
            "avg_clustering": round(nx.average_clustering(G), 4) if n_nodes > 0 else 0,
            "avg_degree": round(sum(degrees) / n_nodes, 2) if n_nodes > 0 else 0,
            "degree_entropy": round(entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "connected_components": nx.number_connected_components(G)
        }
        
        # Centrality metrics
        if n_nodes > 0:
            degree_centrality = nx.degree_centrality(G)
            metrics["degree_centrality_avg"] = round(np.mean(list(degree_centrality.values())), 4)
            metrics["degree_centrality_max"] = round(max(degree_centrality.values()), 4)
            
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            metrics["top_degree_central"] = [
                {
                    "pattern": G.nodes[idx]["label"],
                    "degree_centrality": round(score, 4),
                    "pmi": G.nodes[idx]["pmi"]
                }
                for idx, score in top_degree
            ]
        
        if n_edges > 0 and n_nodes > 2:
            try:
                betweenness = nx.betweenness_centrality(G, normalized=True)
                
                metrics["betweenness_centrality_avg"] = round(np.mean(list(betweenness.values())), 4)
                metrics["betweenness_centrality_max"] = round(max(betweenness.values()), 4)
                metrics["betweenness_centrality_std"] = round(np.std(list(betweenness.values())), 4)
                
                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
                metrics["top_betweenness_central"] = [
                    {
                        "pattern": G.nodes[idx]["label"],
                        "betweenness_centrality": round(score, 4),
                        "pmi": G.nodes[idx]["pmi"],
                        "cluster": G.nodes[idx].get("cluster", -1)
                    }
                    for idx, score in top_betweenness
                ]
                
                max_betweenness = max(betweenness.values())
                sum_diff = sum(max_betweenness - b for b in betweenness.values())
                max_possible = (n_nodes - 1) * (n_nodes - 2) / 2
                if max_possible > 0:
                    metrics["betweenness_centralization"] = round(sum_diff / max_possible, 4)
                else:
                    metrics["betweenness_centralization"] = 0
                
                for idx, bc in betweenness.items():
                    G.nodes[idx]["betweenness"] = round(bc, 4)
                    
            except Exception as e:
                logger.warning(f"Ошибка расчёта betweenness: {e}")
                metrics["betweenness_centrality_avg"] = 0
                metrics["betweenness_centrality_max"] = 0
                metrics["top_betweenness_central"] = []
            
            try:
                closeness = nx.closeness_centrality(G)
                metrics["closeness_centrality_avg"] = round(np.mean(list(closeness.values())), 4)
            except:
                metrics["closeness_centrality_avg"] = 0
            
            try:
                metrics["assortativity"] = round(nx.degree_assortativity_coefficient(G), 4)
            except:
                metrics["assortativity"] = 0
        
        return metrics
    
    def save_visualization(self, G: nx.Graph, theme: str, metrics: Dict, 
                          clustering_result: Dict, output_dir: Path):
        """Сохраняет визуализацию с информацией о кластерах."""
        if G.number_of_nodes() == 0:
            return
        
        cluster_colors = self._generate_cluster_colors(clustering_result.get("n_clusters", 1))
        
        for node_id in G.nodes():
            cluster_id = G.nodes[node_id].get("cluster", -1)
            if cluster_id == -1:
                G.nodes[node_id]["color"] = "#999999"
            else:
                G.nodes[node_id]["color"] = cluster_colors.get(cluster_id, "#666666")
        
        net = Network(height="900px", width="100%", bgcolor="#ffffff", notebook=False)
        net.from_nx(G)
        
        html_path = output_dir / f"network_{theme}.html"
        net.save_graph(str(html_path))
        
        self._inject_interactive_js(html_path, theme, metrics, clustering_result)
        logger.info(f"Визуализация: {html_path}")
    
    def _generate_cluster_colors(self, n_clusters: int) -> Dict[int, str]:
        """Генерирует цвета для кластеров."""
        colors = {}
        for i in range(n_clusters):
            hue = i / max(n_clusters, 1)
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            colors[i] = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
        return colors
    
    def _inject_interactive_js(self, html_path: Path, theme: str, metrics: Dict,
                               clustering_result: Dict):
        """Внедряет интерактивный JS."""
        
        top_betweenness_html = ""
        if "top_betweenness_central" in metrics:
            top_betweenness_html = "<br><b>Топ по Betweenness:</b><br>"
            for i, item in enumerate(metrics["top_betweenness_central"][:5], 1):
                pattern_short = item['pattern'][:25] + "..." if len(item['pattern']) > 25 else item['pattern']
                top_betweenness_html += f"{i}. {pattern_short} ({item['betweenness_centrality']:.3f})<br>"
        
        clusters_info_html = f"Кластеров: {clustering_result.get('n_clusters', 0)}"
        if clustering_result.get('n_noise', 0) > 0:
            clusters_info_html += f" (шум: {clustering_result['n_noise']})"
        
        silhouette = clustering_result.get('metrics', {}).get('silhouette_score')
        if silhouette is not None:
            clusters_info_html += f"<br>Silhouette: {silhouette:.3f}"
        
        js_code = f'''
<script type="text/javascript">
document.addEventListener('DOMContentLoaded', function() {{
    const panel = document.createElement('div');
    panel.style.cssText = 'position:fixed;top:10px;left:10px;z-index:9999;background:rgba(255,255,255,0.95);border:1px solid #ddd;padding:12px;border-radius:8px;font-family:sans-serif;font-size:12px;box-shadow:0 2px 10px rgba(0,0,0,0.1);max-width:320px;';
    panel.innerHTML = `
        <b style="font-size:14px;">Тема: {theme}</b><br>
        <div style="margin:8px 0;color:#666;">
            Узлов: {metrics.get('nodes', 0)} | Рёбер: {metrics.get('edges', 0)}<br>
            Плотность: {metrics.get('density', 0):.3f} | Кластеризация: {metrics.get('avg_clustering', 0):.3f}<br>
            {clusters_info_html}<br>
            <hr style="margin:5px 0;border:none;border-top:1px solid #eee;">
            Betweenness avg: {metrics.get('betweenness_centrality_avg', 0):.4f}<br>
            Betweenness max: {metrics.get('betweenness_centrality_max', 0):.4f}<br>
            Централизация: {metrics.get('betweenness_centralization', 0):.4f}
            {top_betweenness_html}
        </div>
        <input type="text" id="searchInput" placeholder="Поиск..." style="width:140px;padding:4px 8px;border:1px solid #ddd;border-radius:4px;">
        <button id="searchBtn" style="padding:4px 8px;margin-left:4px;">Найти</button>
        <button id="resetBtn" style="padding:4px 8px;margin-left:4px;">Сброс</button>
    `;
    document.body.appendChild(panel);
    
    const infoBox = document.createElement('div');
    infoBox.style.cssText = 'position:fixed;bottom:14px;right:14px;z-index:9999;background:rgba(255,255,255,0.95);border:1px solid #ddd;padding:12px;max-width:350px;border-radius:8px;display:none;font-family:sans-serif;font-size:13px;';
    document.body.appendChild(infoBox);
    
    document.getElementById('searchBtn').onclick = function() {{
        const q = document.getElementById('searchInput').value.trim().toLowerCase();
        if (!q) return;
        const found = nodes.get({{filter: n => n.label.toLowerCase().includes(q)}});
        if (found.length === 0) {{ alert('Не найдено'); return; }}
        network.selectNodes(found.map(x => x.id));
        network.fit({{nodes: found.map(x => x.id), animation: true}});
    }};
    
    document.getElementById('resetBtn').onclick = function() {{
        document.getElementById('searchInput').value = '';
        nodes.update(nodes.get().map(n => ({{...n}})));
        edges.update(edges.get().map(e => ({{...e, hidden: false}})));
        infoBox.style.display = 'none';
        network.fit();
    }};
    
    network.on('click', function(params) {{
        if (params.nodes.length > 0) {{
            const id = params.nodes[0];
            const node = nodes.get(id);
            const bc = node.betweenness !== undefined ? node.betweenness.toFixed(4) : 'N/A';
            infoBox.innerHTML = `
                <b>${{node.label}}</b><br>
                PMI: ${{node.pmi}}<br>
                Doc freq: ${{node.doc_freq}}<br>
                Кластер: ${{node.cluster}}<br>
                Betweenness: ${{bc}}
            `;
            infoBox.style.display = 'block';
        }}
    }});
    
    network.on('doubleClick', function() {{
        infoBox.style.display = 'none';
    }});
}});
</script>
'''
        html = html_path.read_text(encoding="utf-8")
        html = html.replace("</body>", js_code + "\n</body>") if "</body>" in html else html + js_code
        html_path.write_text(html, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ ПАЙПЛАЙН
# ═══════════════════════════════════════════════════════════════════════════════

class PatternPipeline:
    """Главный пайплайн."""
    
    def __init__(self, config: Config = None):
        self.config = config or CONFIG
        self.parser = NKRYAParser()
        self.lemmatizer = Lemmatizer()
        self.extractor = PatternExtractor(self.config)
        self.network_builder = PatternNetworkBuilder(self.config)
    
    def process_file(self, filepath: Path) -> Dict:
        """Обрабатывает файл."""
        logger.info(f"{'='*60}")
        logger.info(f"Файл: {filepath}")
        
        theme = filepath.stem.replace("corpus_", "").replace("_", " ")
        
        metadata, examples = self.parser.parse_file(filepath)
        
        if not theme and metadata.request:
            theme = metadata.request
        
        logger.info(f"Тема: {theme}, примеров: {len(examples)}")
        
        if not examples:
            return {"theme": theme, "status": "no_examples"}
        
        texts = [ex.text for ex in examples if ex.text.strip()]
        logger.info(f"Текстов после очистки: {len(texts)}")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cache_path = output_dir / f"lemmatized_{theme}.json"
        
        if self.config.use_cache and cache_path.exists():
            logger.info("Загрузка кэша лемматизации...")
            lemmatized = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            lemmatized = self.lemmatizer.lemmatize_batch(
                texts, 
                self.config.pos_filter,
                filter_ner=self.config.filter_ner
            )
            cache_path.write_text(json.dumps(lemmatized, ensure_ascii=False, indent=2), encoding="utf-8")
        
        patterns = self.extractor.extract(lemmatized)
        
        if not patterns:
            return {"theme": theme, "status": "no_patterns"}
        
        G, metrics, clustering_result = self.network_builder.build(patterns, theme)
        
        self._save_results(theme, examples, lemmatized, patterns, metrics, 
                          clustering_result, output_dir)
        
        if G.number_of_nodes() > 0:
            self.network_builder.save_visualization(G, theme, metrics, 
                                                    clustering_result, output_dir)
        
        return {
            "theme": theme,
            "status": "success",
            "examples_count": len(examples),
            "patterns_count": len(patterns),
            "metrics": metrics,
            "clustering": {
                "method": clustering_result.get("method"),
                "n_clusters": clustering_result.get("n_clusters"),
                "n_noise": clustering_result.get("n_noise"),
                "silhouette_score": clustering_result.get("metrics", {}).get("silhouette_score")
            }
        }
    
    def _save_results(self, theme: str, examples: List[NKRYAExample],
                      lemmatized: List[str], patterns: List[Pattern],
                      metrics: Dict, clustering_result: Dict, output_dir: Path):
        
        (output_dir / f"examples_{theme}.json").write_text(
            json.dumps([{"text": ex.text, "source": ex.source, "date": ex.date} for ex in examples],
                      ensure_ascii=False, indent=2), encoding="utf-8")
        
        patterns_with_clusters = []
        for idx, p in enumerate(patterns):
            p_dict = asdict(p)
            p_dict["cluster"] = clustering_result["labels"][idx] if idx < len(clustering_result["labels"]) else -1
            patterns_with_clusters.append(p_dict)
        
        (output_dir / f"patterns_{theme}.json").write_text(
            json.dumps(patterns_with_clusters, ensure_ascii=False, indent=2), encoding="utf-8")
        
        (output_dir / f"metrics_{theme}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        
        (output_dir / f"clusters_{theme}.json").write_text(
            json.dumps({
                "method": clustering_result.get("method"),
                "n_clusters": clustering_result.get("n_clusters"),
                "n_noise": clustering_result.get("n_noise"),
                "metrics": clustering_result.get("metrics"),
                "cluster_info": clustering_result.get("cluster_info")
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        
        logger.info(f"Сохранено в {output_dir}")
    
    def process_directory(self, input_dir: Path = None) -> List[Dict]:
        """Обрабатывает директорию."""
        input_dir = input_dir or Path(self.config.input_dir)
        
        if not input_dir.exists():
            logger.error(f"Директория не существует: {input_dir}")
            return []
        
        files = list(input_dir.glob("*.txt"))
        if not files:
            logger.warning(f"Нет .txt файлов")
            return []
        
        logger.info(f"Файлов: {len(files)}")
        
        results = []
        for filepath in files:
            try:
                results.append(self.process_file(filepath))
            except Exception as e:
                logger.error(f"Ошибка {filepath}: {e}")
                import traceback
                traceback.print_exc()
                results.append({"file": str(filepath), "status": "error", "error": str(e)})
        
        (Path(self.config.output_dir) / "summary.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Извлечение паттернов из НКРЯ")
    parser.add_argument("--input", "-i", type=str, default="./files_nkry")
    parser.add_argument("--output", "-o", type=str, default="./data_nkry")
    parser.add_argument("--min-pmi", type=float, default=3.0)
    parser.add_argument("--min-doc-freq", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--similarity", type=float, default=0.75)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-ner-filter", action="store_true")
    
    # Аргументы для кластеризации
    parser.add_argument("--clustering", type=str, default="dbscan",
                        choices=["dbscan", "kmeans"])
    parser.add_argument("--dbscan-eps", type=float, default=0.0,
                        help="Порог eps для DBSCAN (0 = автоподбор)")
    parser.add_argument("--dbscan-min-samples", type=int, default=3)
    parser.add_argument("--kmeans-clusters", type=int, default=10)
    parser.add_argument("--min-clusters", type=int, default=3,
                        help="Минимальное желаемое число кластеров")
    parser.add_argument("--max-clusters", type=int, default=20,
                        help="Максимальное число кластеров")
    
    args = parser.parse_args()
    
    config = Config(
        input_dir=args.input,
        output_dir=args.output,
        min_pmi=args.min_pmi,
        min_doc_freq=args.min_doc_freq,
        min_count=args.min_count,
        similarity_threshold=args.similarity,
        use_cache=not args.no_cache,
        filter_ner=not args.no_ner_filter,
        clustering_method=args.clustering,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        kmeans_n_clusters=args.kmeans_clusters,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters
    )
    
    pipeline = PatternPipeline(config)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = pipeline.process_file(input_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        results = pipeline.process_directory(input_path)
        print(f"\nОбработано: {len(results)}")
        for r in results:
            clustering_info = r.get('clustering', {})
            silhouette = clustering_info.get('silhouette_score')
            sil_str = f", silhouette={silhouette:.3f}" if silhouette else ""
            print(f"  {r.get('theme', '?')}: {r.get('status')} "
                  f"({r.get('patterns_count', 0)} паттернов, "
                  f"{clustering_info.get('n_clusters', 0)} кластеров{sil_str})")


if __name__ == "__main__":
    main()
