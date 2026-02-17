# -*- coding: utf-8 -*-
"""
Сравнительный анализ паттернов между алгоритмическим и доалгоритмическим корпусами.

Анализирует:
- Различия в структуре сетей паттернов
- Изменения в централизации и энтропии
- Уникальные и общие паттерны
- Сдвиги в кластерной структуре
- Ключевые паттерны-посредники (betweenness)

Версия 1.0
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Для серверов без GUI

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ComparisonConfig:
    """Конфигурация сравнительного анализа."""
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ПУТИ К ФАЙЛАМ (НАСТРОЙТЕ ПОД СВОИ ДАННЫЕ)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Доалгоритмический корпус (до 2010)
    prealg_patterns_path: str = "./data_nkry/patterns_prealg любовь.json"
    prealg_metrics_path: str = "./data_nkry/metrics_prealg любовь.json"
    prealg_clusters_path: str = "./data_nkry/clusters_prealg любовь.json"
    prealg_name: str = "Доалгоритмический (до 2010)"
    
    # Алгоритмический корпус (после 2010)
    algo_patterns_path: str = "./data_nkry/patterns_algo любовь.json"
    algo_metrics_path: str = "./data_nkry/metrics_algo любовь.json"
    algo_clusters_path: str = "./data_nkry/clusters_algo любовь.json"

    algo_name: str = "Алгоритмический (после 2010)"
    
    # Выходная директория
    output_dir: str = "./comparison_results"
    
    # Параметры анализа
    embedding_model: str = "cointegrated/rubert-tiny2"
    similarity_threshold: float = 0.85  # Порог для считания паттернов "похожими"
    top_n_patterns: int = 20  # Топ паттернов для отчёта
    
    # Тема анализа (для отчёта)
    theme: str = "любовь"


CONFIG = ComparisonConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════════

def load_json(path: str) -> Optional[Dict]:
    """Загружает JSON файл."""
    filepath = Path(path)
    if not filepath.exists():
        logger.warning(f"Файл не найден: {path}")
        return None
    
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Ошибка загрузки {path}: {e}")
        return None


def load_corpus_data(patterns_path: str, metrics_path: str, clusters_path: str) -> Dict:
    """Загружает все данные корпуса."""
    patterns = load_json(patterns_path) or []
    metrics = load_json(metrics_path) or {}
    clusters = load_json(clusters_path) or {}
    
    return {
        "patterns": patterns,
        "metrics": metrics,
        "clusters": clusters,
        "pattern_set": {p["ngram"] for p in patterns},
        "pattern_dict": {p["ngram"]: p for p in patterns}
    }


# ═══════════════════════════════════════════════════════════════════════════════
# АНАЛИЗ ПАТТЕРНОВ
# ═══════════════════════════════════════════════════════════════════════════════

class PatternComparator:
    """Сравнительный анализ паттернов двух корпусов."""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self._model = None
    
    def _init_model(self):
        if self._model is None:
            logger.info(f"Загрузка модели: {self.config.embedding_model}")
            self._model = SentenceTransformer(self.config.embedding_model)
    
    def compare(self, prealg_data: Dict, algo_data: Dict) -> Dict:
        """Выполняет полный сравнительный анализ."""
        logger.info("="*60)
        logger.info("СРАВНИТЕЛЬНЫЙ АНАЛИЗ КОРПУСОВ")
        logger.info("="*60)
        
        results = {
            "config": {
                "prealg_name": self.config.prealg_name,
                "algo_name": self.config.algo_name,
                "theme": self.config.theme
            },
            "summary": {},
            "metrics_comparison": {},
            "pattern_overlap": {},
            "unique_patterns": {},
            "semantic_drift": {},
            "centrality_shift": {},
            "cluster_analysis": {},
            "key_findings": []
        }
        
        # 1. Сравнение базовых метрик
        results["metrics_comparison"] = self._compare_metrics(
            prealg_data["metrics"], 
            algo_data["metrics"]
        )
        
        # 2. Анализ пересечения паттернов
        results["pattern_overlap"] = self._analyze_overlap(
            prealg_data["pattern_set"],
            algo_data["pattern_set"],
            prealg_data["pattern_dict"],
            algo_data["pattern_dict"]
        )
        
        # 3. Уникальные паттерны
        results["unique_patterns"] = self._analyze_unique_patterns(
            prealg_data["patterns"],
            algo_data["patterns"],
            prealg_data["pattern_set"],
            algo_data["pattern_set"]
        )
        
        # 4. Семантический дрейф
        results["semantic_drift"] = self._analyze_semantic_drift(
            prealg_data["patterns"],
            algo_data["patterns"]
        )
        
        # 5. Сдвиг в централизации
        results["centrality_shift"] = self._analyze_centrality_shift(
            prealg_data["metrics"],
            algo_data["metrics"]
        )
        
        # 6. Кластерный анализ
        results["cluster_analysis"] = self._analyze_clusters(
            prealg_data["clusters"],
            algo_data["clusters"]
        )
        
        # 7. Сводка и ключевые выводы
        results["summary"] = self._generate_summary(results)
        results["key_findings"] = self._generate_findings(results)
        
        return results
    
    def _compare_metrics(self, prealg_metrics: Dict, algo_metrics: Dict) -> Dict:
        """Сравнивает метрики сетей."""
        logger.info("Сравнение метрик сетей...")
        
        metrics_to_compare = [
            ("nodes", "Узлов"),
            ("edges", "Рёбер"),
            ("density", "Плотность"),
            ("avg_clustering", "Коэф. кластеризации"),
            ("avg_degree", "Средняя степень"),
            ("normalized_entropy", "Нормализованная энтропия"),
            ("betweenness_centrality_avg", "Betweenness (среднее)"),
            ("betweenness_centrality_max", "Betweenness (макс)"),
            ("betweenness_centralization", "Централизация"),
            ("closeness_centrality_avg", "Closeness (среднее)"),
            ("connected_components", "Компонент связности"),
            ("assortativity", "Ассортативность")
        ]
        
        comparison = {}
        
        for metric_key, metric_name in metrics_to_compare:
            prealg_val = prealg_metrics.get(metric_key, 0) or 0
            algo_val = algo_metrics.get(metric_key, 0) or 0
            
            if prealg_val != 0:
                change_pct = ((algo_val - prealg_val) / abs(prealg_val)) * 100
            else:
                change_pct = 100 if algo_val > 0 else 0
            
            comparison[metric_key] = {
                "name": metric_name,
                "prealg": round(prealg_val, 4) if isinstance(prealg_val, float) else prealg_val,
                "algo": round(algo_val, 4) if isinstance(algo_val, float) else algo_val,
                "diff": round(algo_val - prealg_val, 4),
                "change_pct": round(change_pct, 2),
                "direction": "↑" if algo_val > prealg_val else ("↓" if algo_val < prealg_val else "=")
            }
        
        return comparison
    
    def _analyze_overlap(self, prealg_set: Set[str], algo_set: Set[str],
                         prealg_dict: Dict, algo_dict: Dict) -> Dict:
        """Анализирует пересечение паттернов."""
        logger.info("Анализ пересечения паттернов...")
        
        common = prealg_set & algo_set
        only_prealg = prealg_set - algo_set
        only_algo = algo_set - prealg_set
        
        # Jaccard similarity
        if len(prealg_set | algo_set) > 0:
            jaccard = len(common) / len(prealg_set | algo_set)
        else:
            jaccard = 0
        
        # Анализ общих паттернов - изменение PMI
        common_analysis = []
        for ngram in common:
            prealg_pmi = prealg_dict[ngram].get("pmi", 0)
            algo_pmi = algo_dict[ngram].get("pmi", 0)
            pmi_change = algo_pmi - prealg_pmi
            
            common_analysis.append({
                "ngram": ngram,
                "prealg_pmi": round(prealg_pmi, 3),
                "algo_pmi": round(algo_pmi, 3),
                "pmi_change": round(pmi_change, 3),
                "prealg_doc_freq": prealg_dict[ngram].get("doc_freq", 0),
                "algo_doc_freq": algo_dict[ngram].get("doc_freq", 0)
            })
        
        # Сортируем по изменению PMI
        common_analysis.sort(key=lambda x: abs(x["pmi_change"]), reverse=True)
        
        return {
            "total_prealg": len(prealg_set),
            "total_algo": len(algo_set),
            "common_count": len(common),
            "only_prealg_count": len(only_prealg),
            "only_algo_count": len(only_algo),
            "jaccard_similarity": round(jaccard, 4),
            "overlap_pct_prealg": round(len(common) / len(prealg_set) * 100, 2) if prealg_set else 0,
            "overlap_pct_algo": round(len(common) / len(algo_set) * 100, 2) if algo_set else 0,
            "common_patterns_top": common_analysis[:self.config.top_n_patterns],
            "common_patterns_all": common_analysis
        }
    
    def _analyze_unique_patterns(self, prealg_patterns: List[Dict], algo_patterns: List[Dict],
                                  prealg_set: Set[str], algo_set: Set[str]) -> Dict:
        """Анализирует уникальные паттерны каждого корпуса."""
        logger.info("Анализ уникальных паттернов...")
        
        only_prealg = prealg_set - algo_set
        only_algo = algo_set - prealg_set
        
        # Уникальные для доалгоритмического корпуса
        prealg_unique = [p for p in prealg_patterns if p["ngram"] in only_prealg]
        prealg_unique.sort(key=lambda x: x.get("pmi", 0), reverse=True)
        
        # Уникальные для алгоритмического корпуса
        algo_unique = [p for p in algo_patterns if p["ngram"] in only_algo]
        algo_unique.sort(key=lambda x: x.get("pmi", 0), reverse=True)
        
        return {
            "prealg_unique": {
                "count": len(prealg_unique),
                "top_patterns": [
                    {
                        "ngram": p["ngram"],
                        "pmi": round(p.get("pmi", 0), 3),
                        "doc_freq": p.get("doc_freq", 0),
                        "cluster": p.get("cluster", -1)
                    }
                    for p in prealg_unique[:self.config.top_n_patterns]
                ],
                "avg_pmi": round(np.mean([p.get("pmi", 0) for p in prealg_unique]), 3) if prealg_unique else 0
            },
            "algo_unique": {
                "count": len(algo_unique),
                "top_patterns": [
                    {
                        "ngram": p["ngram"],
                        "pmi": round(p.get("pmi", 0), 3),
                        "doc_freq": p.get("doc_freq", 0),
                        "cluster": p.get("cluster", -1)
                    }
                    for p in algo_unique[:self.config.top_n_patterns]
                ],
                "avg_pmi": round(np.mean([p.get("pmi", 0) for p in algo_unique]), 3) if algo_unique else 0
            },
            "interpretation": self._interpret_unique_patterns(prealg_unique, algo_unique)
        }
    
    def _interpret_unique_patterns(self, prealg_unique: List[Dict], algo_unique: List[Dict]) -> str:
        """Интерпретирует различия в уникальных паттернах."""
        interpretations = []
        
        if len(algo_unique) > len(prealg_unique):
            interpretations.append(
                f"Алгоритмический корпус содержит на {len(algo_unique) - len(prealg_unique)} "
                f"уникальных паттернов больше, что может указывать на расширение тематики или "
                f"появление новых устойчивых выражений."
            )
        elif len(prealg_unique) > len(algo_unique):
            interpretations.append(
                f"Доалгоритмический корпус содержит на {len(prealg_unique) - len(algo_unique)} "
                f"уникальных паттернов больше, что может указывать на стандартизацию и "
                f"сужение вариативности в алгоритмическую эпоху."
            )
        
        # Анализ PMI
        if prealg_unique and algo_unique:
            avg_pmi_prealg = np.mean([p.get("pmi", 0) for p in prealg_unique])
            avg_pmi_algo = np.mean([p.get("pmi", 0) for p in algo_unique])
            
            if avg_pmi_algo > avg_pmi_prealg:
                interpretations.append(
                    f"Уникальные паттерны алгоритмического корпуса имеют более высокий средний PMI "
                    f"({avg_pmi_algo:.2f} vs {avg_pmi_prealg:.2f}), что указывает на более "
                    f"устойчивые коллокации."
                )
        
        return " ".join(interpretations) if interpretations else "Нет значимых различий."
    
    def _analyze_semantic_drift(self, prealg_patterns: List[Dict], 
                                 algo_patterns: List[Dict]) -> Dict:
        """Анализирует семантический дрейф с помощью эмбеддингов."""
        logger.info("Анализ семантического дрейфа...")
        
        self._init_model()
        
        if not prealg_patterns or not algo_patterns:
            return {"error": "Недостаточно данных"}
        
        # Получаем эмбеддинги
        prealg_ngrams = [p["ngram"] for p in prealg_patterns]
        algo_ngrams = [p["ngram"] for p in algo_patterns]
        
        prealg_embeddings = self._model.encode(prealg_ngrams, show_progress_bar=False)
        algo_embeddings = self._model.encode(algo_ngrams, show_progress_bar=False)
        
        # Центроид каждого корпуса
        prealg_centroid = prealg_embeddings.mean(axis=0)
        algo_centroid = algo_embeddings.mean(axis=0)
        
        # Косинусное расстояние между центроидами
        centroid_similarity = float(cosine_similarity([prealg_centroid], [algo_centroid])[0][0])
        
        # Находим паттерны, близкие к центроиду (характерные для корпуса)
        prealg_to_centroid = cosine_similarity(prealg_embeddings, [prealg_centroid]).flatten()
        algo_to_centroid = cosine_similarity(algo_embeddings, [algo_centroid]).flatten()
        
        prealg_central_idx = np.argsort(prealg_to_centroid)[-5:][::-1]
        algo_central_idx = np.argsort(algo_to_centroid)[-5:][::-1]
        
        # Поиск "смещённых" паттернов (есть в обоих, но с разной позицией)
        common_ngrams = set(prealg_ngrams) & set(algo_ngrams)
        
        drift_analysis = []
        for ngram in common_ngrams:
            prealg_idx = prealg_ngrams.index(ngram)
            algo_idx = algo_ngrams.index(ngram)
            
            prealg_emb = prealg_embeddings[prealg_idx]
            algo_emb = algo_embeddings[algo_idx]
            
            # Как изменилось положение относительно центроида
            prealg_sim_to_center = float(cosine_similarity([prealg_emb], [prealg_centroid])[0][0])
            algo_sim_to_center = float(cosine_similarity([algo_emb], [algo_centroid])[0][0])
            
            drift_analysis.append({
                "ngram": ngram,
                "prealg_centrality": round(prealg_sim_to_center, 4),
                "algo_centrality": round(algo_sim_to_center, 4),
                "centrality_shift": round(algo_sim_to_center - prealg_sim_to_center, 4)
            })
        
        drift_analysis.sort(key=lambda x: abs(x["centrality_shift"]), reverse=True)
        
        return {
            "centroid_similarity": round(centroid_similarity, 4),
            "semantic_distance": round(1 - centroid_similarity, 4),
            "prealg_central_patterns": [prealg_ngrams[i] for i in prealg_central_idx],
            "algo_central_patterns": [algo_ngrams[i] for i in algo_central_idx],
            "top_drifted_patterns": drift_analysis[:10],
            "interpretation": self._interpret_semantic_drift(centroid_similarity, drift_analysis)
        }
    
    def _interpret_semantic_drift(self, centroid_similarity: float, 
                                   drift_analysis: List[Dict]) -> str:
        """Интерпретирует семантический дрейф."""
        interpretations = []
        
        if centroid_similarity > 0.95:
            interpretations.append(
                "Семантическое пространство корпусов практически идентично (сходство > 0.95)."
            )
        elif centroid_similarity > 0.85:
            interpretations.append(
                "Семантические пространства корпусов близки, но есть заметные различия."
            )
        else:
            interpretations.append(
                f"Обнаружен значительный семантический дрейф (сходство = {centroid_similarity:.2f}). "
                "Это может указывать на существенное изменение тематики или способов выражения."
            )
        
        if drift_analysis:
            top_shifted = [d for d in drift_analysis if abs(d["centrality_shift"]) > 0.1]
            if top_shifted:
                examples = ", ".join([d["ngram"] for d in top_shifted[:3]])
                interpretations.append(
                    f"Паттерны с наибольшим сдвигом: {examples}."
                )
        
        return " ".join(interpretations)
    
    def _analyze_centrality_shift(self, prealg_metrics: Dict, algo_metrics: Dict) -> Dict:
        """Анализирует изменение централизации сети."""
        logger.info("Анализ сдвига централизации...")
        
        prealg_top = prealg_metrics.get("top_betweenness_central", [])
        algo_top = algo_metrics.get("top_betweenness_central", [])
        
        prealg_top_set = {p["pattern"] for p in prealg_top}
        algo_top_set = {p["pattern"] for p in algo_top}
        
        stable_leaders = prealg_top_set & algo_top_set
        new_leaders = algo_top_set - prealg_top_set
        lost_leaders = prealg_top_set - algo_top_set
        
        # Изменение показателей централизации
        bc_avg_change = (algo_metrics.get("betweenness_centrality_avg", 0) - 
                         prealg_metrics.get("betweenness_centrality_avg", 0))
        bc_max_change = (algo_metrics.get("betweenness_centrality_max", 0) - 
                         prealg_metrics.get("betweenness_centrality_max", 0))
        centralization_change = (algo_metrics.get("betweenness_centralization", 0) - 
                                 prealg_metrics.get("betweenness_centralization", 0))
        
        return {
            "prealg_top_patterns": prealg_top[:10],
            "algo_top_patterns": algo_top[:10],
            "stable_leaders": list(stable_leaders),
            "new_leaders": list(new_leaders),
            "lost_leaders": list(lost_leaders),
            "bc_avg_change": round(bc_avg_change, 4),
            "bc_max_change": round(bc_max_change, 4),
            "centralization_change": round(centralization_change, 4),
            "interpretation": self._interpret_centrality_shift(
                bc_avg_change, centralization_change, 
                stable_leaders, new_leaders, lost_leaders
            )
        }
    
    def _interpret_centrality_shift(self, bc_avg_change: float, centralization_change: float,
                                     stable: Set, new: Set, lost: Set) -> str:
        """Интерпретирует изменение централизации."""
        interpretations = []
        
        if centralization_change > 0.01:
            interpretations.append(
                "Сеть алгоритмического корпуса более централизована — "
                "смыслы концентрируются вокруг меньшего числа ключевых паттернов."
            )
        elif centralization_change < -0.01:
            interpretations.append(
                "Сеть алгоритмического корпуса менее централизована — "
                "смыслы распределены более равномерно."
            )
        
        if len(new) > len(stable):
            interpretations.append(
                f"Произошла значительная смена ключевых паттернов-посредников: "
                f"{len(new)} новых лидеров, только {len(stable)} стабильных."
            )
        
        if lost:
            interpretations.append(
                f"Утратили центральную роль: {', '.join(list(lost)[:3])}."
            )
        
        return " ".join(interpretations) if interpretations else "Структура централизации стабильна."
    
    def _analyze_clusters(self, prealg_clusters: Dict, algo_clusters: Dict) -> Dict:
        """Анализирует изменение кластерной структуры."""
        logger.info("Анализ кластерной структуры...")
        
        prealg_n = prealg_clusters.get("n_clusters", 0)
        algo_n = algo_clusters.get("n_clusters", 0)
        
        prealg_silhouette = prealg_clusters.get("metrics", {}).get("silhouette_score")
        algo_silhouette = algo_clusters.get("metrics", {}).get("silhouette_score")
        
        prealg_info = prealg_clusters.get("cluster_info", {})
        algo_info = algo_clusters.get("cluster_info", {})
        
        # Размеры кластеров
        prealg_sizes = [c.get("size", 0) for c in prealg_info.values()]
        algo_sizes = [c.get("size", 0) for c in algo_info.values()]
        
        # Представители кластеров
        prealg_representatives = [c.get("representative", "") for c in prealg_info.values()]
        algo_representatives = [c.get("representative", "") for c in algo_info.values()]
        
        return {
            "prealg_n_clusters": prealg_n,
            "algo_n_clusters": algo_n,
            "cluster_change": algo_n - prealg_n,
            "prealg_silhouette": prealg_silhouette,
            "algo_silhouette": algo_silhouette,
            "silhouette_change": round((algo_silhouette or 0) - (prealg_silhouette or 0), 4),
            "prealg_cluster_sizes": prealg_sizes,
            "algo_cluster_sizes": algo_sizes,
            "prealg_size_std": round(np.std(prealg_sizes), 2) if prealg_sizes else 0,
            "algo_size_std": round(np.std(algo_sizes), 2) if algo_sizes else 0,
            "prealg_representatives": prealg_representatives,
            "algo_representatives": algo_representatives,
            "interpretation": self._interpret_clusters(
                prealg_n, algo_n, prealg_silhouette, algo_silhouette,
                prealg_sizes, algo_sizes
            )
        }
    
    def _interpret_clusters(self, prealg_n: int, algo_n: int,
                            prealg_sil: Optional[float], algo_sil: Optional[float],
                            prealg_sizes: List, algo_sizes: List) -> str:
        """Интерпретирует изменение кластерной структуры."""
        interpretations = []
        
        if algo_n < prealg_n:
            interpretations.append(
                f"Число кластеров уменьшилось ({prealg_n} → {algo_n}), "
                "что может указывать на консолидацию тем и снижение разнообразия."
            )
        elif algo_n > prealg_n:
            interpretations.append(
                f"Число кластеров увеличилось ({prealg_n} → {algo_n}), "
                "что может указывать на большую дифференциацию тематики."
            )
        
        if prealg_sil and algo_sil:
            if algo_sil > prealg_sil:
                interpretations.append(
                    "Качество кластеризации улучшилось — паттерны формируют более чёткие группы."
                )
            elif algo_sil < prealg_sil - 0.05:
                interpretations.append(
                    "Качество кластеризации ухудшилось — границы между группами паттернов размыты."
                )
        
        if prealg_sizes and algo_sizes:
            prealg_std = np.std(prealg_sizes)
            algo_std = np.std(algo_sizes)
            
            if algo_std > prealg_std * 1.5:
                interpretations.append(
                    "Размеры кластеров стали более неравномерными — "
                    "появились доминирующие смысловые группы."
                )
        
        return " ".join(interpretations) if interpretations else "Кластерная структура стабильна."
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Генерирует общую сводку."""
        metrics = results["metrics_comparison"]
        overlap = results["pattern_overlap"]
        
        return {
            "total_patterns": {
                "prealg": overlap["total_prealg"],
                "algo": overlap["total_algo"],
                "change_pct": round((overlap["total_algo"] - overlap["total_prealg"]) / 
                                   max(overlap["total_prealg"], 1) * 100, 1)
            },
            "jaccard_similarity": overlap["jaccard_similarity"],
            "entropy_change": metrics.get("normalized_entropy", {}).get("diff", 0),
            "density_change": metrics.get("density", {}).get("diff", 0),
            "centralization_change": metrics.get("betweenness_centralization", {}).get("diff", 0),
            "semantic_distance": results.get("semantic_drift", {}).get("semantic_distance", 0),
            "cluster_change": results.get("cluster_analysis", {}).get("cluster_change", 0)
        }
    
    def _generate_findings(self, results: Dict) -> List[str]:
        """Генерирует ключевые выводы."""
        findings = []
        summary = results["summary"]
        
        # 1. Изменение объёма
        pct_change = summary["total_patterns"]["change_pct"]
        if abs(pct_change) > 10:
            direction = "увеличилось" if pct_change > 0 else "уменьшилось"
            findings.append(
                f"📊 Количество паттернов {direction} на {abs(pct_change):.1f}%."
            )
        
        # 2. Пересечение
        jaccard = summary["jaccard_similarity"]
        if jaccard < 0.3:
            findings.append(
                f"⚠️ Низкое пересечение корпусов (Jaccard = {jaccard:.2f}): "
                "паттерны существенно различаются."
            )
        elif jaccard > 0.7:
            findings.append(
                f"✅ Высокое пересечение корпусов (Jaccard = {jaccard:.2f}): "
                "ядро паттернов стабильно."
            )
        
        # 3. Энтропия
        entropy_change = summary["entropy_change"]
        if entropy_change < -0.05:
            findings.append(
                "📉 Снижение энтропии: сеть стала более упорядоченной, "
                "возможно, за счёт стандартизации выражений."
            )
        elif entropy_change > 0.05:
            findings.append(
                "📈 Рост энтропии: увеличилась вариативность связей между паттернами."
            )
        
        # 4. Централизация
        centr_change = summary["centralization_change"]
        if centr_change > 0.02:
            findings.append(
                "🎯 Рост централизации: смыслы концентрируются вокруг "
                "ключевых паттернов-посредников."
            )
        elif centr_change < -0.02:
            findings.append(
                "🔄 Снижение централизации: смыслы распределены более равномерно."
            )
        
        # 5. Семантический дрейф
        sem_dist = summary["semantic_distance"]
        if sem_dist > 0.15:
            findings.append(
                f"🌊 Обнаружен семантический дрейф (расстояние = {sem_dist:.2f}): "
                "тематическое пространство изменилось."
            )
        
        # 6. Гипотеза об алгоритмическом влиянии
        if entropy_change < 0 and centr_change > 0:
            findings.append(
                "💡 ГИПОТЕЗА ПОДТВЕРЖДАЕТСЯ: снижение энтропии при росте централизации "
                "соответствует алгоритмической паттернизации речи."
            )
        
        return findings


# ═══════════════════════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

class ComparisonVisualizer:
    """Визуализация результатов сравнения."""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_all_visualizations(self, results: Dict):
        """Создаёт все визуализации."""
        logger.info("Создание визуализаций...")
        
        self._plot_metrics_comparison(results["metrics_comparison"])
        self._plot_pattern_overlap(results["pattern_overlap"])
        self._plot_cluster_comparison(results["cluster_analysis"])
        
        logger.info(f"Визуализации сохранены в {self.output_dir}")
    
    def _plot_metrics_comparison(self, metrics: Dict):
        """График сравнения метрик."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Сравнение метрик: {self.config.theme}', fontsize=14, fontweight='bold')
        
        # 1. Основные метрики сети
        ax1 = axes[0, 0]
        main_metrics = ["density", "avg_clustering", "normalized_entropy"]
        labels = [metrics[m]["name"] for m in main_metrics if m in metrics]
        prealg_vals = [metrics[m]["prealg"] for m in main_metrics if m in metrics]
        algo_vals = [metrics[m]["algo"] for m in main_metrics if m in metrics]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax1.bar(x - width/2, prealg_vals, width, label='Доалгоритмический', color='#3498db', alpha=0.8)
        ax1.bar(x + width/2, algo_vals, width, label='Алгоритмический', color='#e74c3c', alpha=0.8)
        ax1.set_ylabel('Значение')
        ax1.set_title('Структурные метрики')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Метрики централизации
        ax2 = axes[0, 1]
        centr_metrics = ["betweenness_centrality_avg", "betweenness_centrality_max", "betweenness_centralization"]
        labels2 = [metrics[m]["name"] for m in centr_metrics if m in metrics]
        prealg_vals2 = [metrics[m]["prealg"] for m in centr_metrics if m in metrics]
        algo_vals2 = [metrics[m]["algo"] for m in centr_metrics if m in metrics]
        
        x2 = np.arange(len(labels2))
        ax2.bar(x2 - width/2, prealg_vals2, width, label='Доалгоритмический', color='#3498db', alpha=0.8)
        ax2.bar(x2 + width/2, algo_vals2, width, label='Алгоритмический', color='#e74c3c', alpha=0.8)
        ax2.set_ylabel('Значение')
        ax2.set_title('Метрики централизации')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(labels2, rotation=15, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Изменение метрик (%)
        ax3 = axes[1, 0]
        all_metrics = ["density", "avg_clustering", "normalized_entropy", 
                       "betweenness_centrality_avg", "betweenness_centralization"]
        changes = []
        labels3 = []
        for m in all_metrics:
            if m in metrics:
                changes.append(metrics[m]["change_pct"])
                labels3.append(metrics[m]["name"][:15])
        
        colors = ['#27ae60' if c > 0 else '#c0392b' for c in changes]
        ax3.barh(labels3, changes, color=colors, alpha=0.8)
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Изменение (%)')
        ax3.set_title('Относительное изменение метрик')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Размер сети
        ax4 = axes[1, 1]
        size_metrics = ["nodes", "edges"]
        labels4 = [metrics[m]["name"] for m in size_metrics if m in metrics]
        prealg_vals4 = [metrics[m]["prealg"] for m in size_metrics if m in metrics]
        algo_vals4 = [metrics[m]["algo"] for m in size_metrics if m in metrics]
        
        x4 = np.arange(len(labels4))
        ax4.bar(x4 - width/2, prealg_vals4, width, label='Доалгоритмический', color='#3498db', alpha=0.8)
        ax4.bar(x4 + width/2, algo_vals4, width, label='Алгоритмический', color='#e74c3c', alpha=0.8)
        ax4.set_ylabel('Количество')
        ax4.set_title('Размер сети')
        ax4.set_xticks(x4)
        ax4.set_xticklabels(labels4)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_pattern_overlap(self, overlap: Dict):
        """Диаграмма пересечения паттернов."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Пересечение паттернов: {self.config.theme}', fontsize=14, fontweight='bold')
        
        # 1. Венн-подобная диаграмма (упрощённая)
        ax1 = axes[0]
        
        sizes = [
            overlap["only_prealg_count"],
            overlap["common_count"],
            overlap["only_algo_count"]
        ]
        labels = ['Только\nдоалгоритм.', 'Общие', 'Только\nалгоритм.']
        colors = ['#3498db', '#9b59b6', '#e74c3c']
        
        ax1.bar(labels, sizes, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        for i, (size, label) in enumerate(zip(sizes, labels)):
            ax1.text(i, size + max(sizes)*0.02, str(size), ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Количество паттернов')
        ax1.set_title(f'Jaccard similarity: {overlap["jaccard_similarity"]:.3f}')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Круговая диаграмма
        ax2 = axes[1]
        
        total = sum(sizes)
        percentages = [s/total*100 for s in sizes]
        
        wedges, texts, autotexts = ax2.pie(
            sizes, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.02, 0.02, 0.02)
        )
        ax2.set_title('Распределение паттернов')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "pattern_overlap.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_cluster_comparison(self, clusters: Dict):
        """График сравнения кластеров."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Кластерный анализ: {self.config.theme}', fontsize=14, fontweight='bold')
        
        # 1. Количество и качество кластеров
        ax1 = axes[0]
        
        metrics_names = ['Число кластеров', 'Silhouette score']
        prealg_vals = [
            clusters["prealg_n_clusters"],
            clusters["prealg_silhouette"] or 0
        ]
        algo_vals = [
            clusters["algo_n_clusters"],
            clusters["algo_silhouette"] or 0
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        ax1.bar(x - width/2, prealg_vals, width, label='Доалгоритмический', color='#3498db', alpha=0.8)
        ax1.bar(x + width/2, algo_vals, width, label='Алгоритмический', color='#e74c3c', alpha=0.8)
        ax1.set_ylabel('Значение')
        ax1.set_title('Метрики кластеризации')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Распределение размеров кластеров
        ax2 = axes[1]
        
        prealg_sizes = clusters.get("prealg_cluster_sizes", [])
        algo_sizes = clusters.get("algo_cluster_sizes", [])
        
        if prealg_sizes and algo_sizes:
            positions = [1, 2]
            bp = ax2.boxplot([prealg_sizes, algo_sizes], positions=positions, widths=0.6,
                            patch_artist=True)
            
            bp['boxes'][0].set_facecolor('#3498db')
            bp['boxes'][1].set_facecolor('#e74c3c')
            
            for box in bp['boxes']:
                box.set_alpha(0.7)
            
            ax2.set_xticklabels(['Доалгоритм.', 'Алгоритм.'])
            ax2.set_ylabel('Размер кластера')
            ax2.set_title('Распределение размеров кластеров')
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Нет данных о размерах кластеров', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "cluster_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# ГЕНЕРАТОР ОТЧЁТА
# ═══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Генератор текстового и HTML отчёта."""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_reports(self, results: Dict):
        """Генерирует все отчёты."""
        self._generate_text_report(results)
        self._generate_html_report(results)
        self._save_json_results(results)
        
        logger.info(f"Отчёты сохранены в {self.output_dir}")
    
    def _generate_text_report(self, results: Dict):
        """Генерирует текстовый отчёт."""
        lines = []
        lines.append("=" * 80)
        lines.append("СРАВНИТЕЛЬНЫЙ АНАЛИЗ ПАТТЕРНОВ")
        lines.append(f"Тема: {self.config.theme}")
        lines.append("=" * 80)
        lines.append("")
        
        # Сводка
        lines.append("КРАТКАЯ СВОДКА")
        lines.append("-" * 40)
        summary = results["summary"]
        lines.append(f"Доалгоритмический корпус: {summary['total_patterns']['prealg']} паттернов")
        lines.append(f"Алгоритмический корпус: {summary['total_patterns']['algo']} паттернов")
        lines.append(f"Изменение: {summary['total_patterns']['change_pct']:+.1f}%")
        lines.append(f"Jaccard similarity: {summary['jaccard_similarity']:.3f}")
        lines.append(f"Семантическое расстояние: {summary['semantic_distance']:.3f}")
        lines.append("")
        
        # Ключевые выводы
        lines.append("КЛЮЧЕВЫЕ ВЫВОДЫ")
        lines.append("-" * 40)
        for finding in results["key_findings"]:
            lines.append(finding)
        lines.append("")
        
        # Сравнение метрик
        lines.append("СРАВНЕНИЕ МЕТРИК")
        lines.append("-" * 40)
        lines.append(f"{'Метрика':<30} {'Доалг.':<12} {'Алг.':<12} {'Изм.':<12} {'%':<10}")
        lines.append("-" * 76)
        
        for key, data in results["metrics_comparison"].items():
            name = data["name"][:28]
            prealg = f"{data['prealg']:.4f}" if isinstance(data['prealg'], float) else str(data['prealg'])
            algo = f"{data['algo']:.4f}" if isinstance(data['algo'], float) else str(data['algo'])
            diff = f"{data['diff']:+.4f}" if isinstance(data['diff'], float) else str(data['diff'])
            pct = f"{data['change_pct']:+.1f}%"
            lines.append(f"{name:<30} {prealg:<12} {algo:<12} {diff:<12} {pct:<10}")
        lines.append("")
        
        # Уникальные паттерны
        lines.append("УНИКАЛЬНЫЕ ПАТТЕРНЫ")
        lines.append("-" * 40)
        unique = results["unique_patterns"]
        
        lines.append(f"\nТолько в доалгоритмическом ({unique['prealg_unique']['count']}):")
        for p in unique["prealg_unique"]["top_patterns"][:10]:
            lines.append(f"  • {p['ngram']} (PMI: {p['pmi']:.2f})")
        
        lines.append(f"\nТолько в алгоритмическом ({unique['algo_unique']['count']}):")
        for p in unique["algo_unique"]["top_patterns"][:10]:
            lines.append(f"  • {p['ngram']} (PMI: {p['pmi']:.2f})")
        lines.append("")
        
        # Интерпретации
        lines.append("ИНТЕРПРЕТАЦИИ")
        lines.append("-" * 40)
        lines.append(f"\nСемантика: {results['semantic_drift'].get('interpretation', 'N/A')}")
        lines.append(f"\nЦентрализация: {results['centrality_shift'].get('interpretation', 'N/A')}")
        lines.append(f"\nКластеры: {results['cluster_analysis'].get('interpretation', 'N/A')}")
        
        # Сохранение
        report_path = self.output_dir / "comparison_report.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")
    
    def _generate_html_report(self, results: Dict):
        """Генерирует HTML отчёт."""
        html = f'''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Сравнительный анализ: {self.config.theme}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; 
            color: #333; 
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ 
            color: #2c3e50; 
            margin-bottom: 10px;
            font-size: 28px;
        }}
        h2 {{ 
            color: #34495e; 
            margin: 30px 0 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }}
        h3 {{ color: #7f8c8d; margin: 20px 0 10px; }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .summary-item {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-item .value {{ font-size: 32px; font-weight: bold; }}
        .summary-item .label {{ font-size: 14px; opacity: 0.9; }}
        .finding {{
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }}
        .finding.positive {{ border-color: #27ae60; background: #e8f8e8; }}
        .finding.negative {{ border-color: #e74c3c; background: #f8e8e8; }}
        .finding.hypothesis {{ border-color: #f39c12; background: #fef8e8; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .pattern-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0;
        }}
        .pattern-tag {{
            background: #3498db;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 13px;
        }}
        .pattern-tag.prealg {{ background: #3498db; }}
        .pattern-tag.algo {{ background: #e74c3c; }}
        .pattern-tag.common {{ background: #9b59b6; }}
        .img-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .img-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .interpretation {{
            background: #fff9e6;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1> Сравнительный анализ паттернов</h1>
        <p style="color: #666; margin-bottom: 30px;">Тема: <strong>{self.config.theme}</strong></p>
        
        <!-- Сводка -->
        <div class="card">
            <h2>📋 Краткая сводка</h2>
            <div class="summary-grid">
                <div class="summary-item" style="background: linear-gradient(135deg, #3498db, #2980b9);">
                    <div class="value">{results['summary']['total_patterns']['prealg']}</div>
                    <div class="label">Паттернов (доалгоритм.)</div>
                </div>
                <div class="summary-item" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">
                    <div class="value">{results['summary']['total_patterns']['algo']}</div>
                    <div class="label">Паттернов (алгоритм.)</div>
                </div>
                <div class="summary-item" style="background: linear-gradient(135deg, #9b59b6, #8e44ad);">
                    <div class="value">{results['summary']['jaccard_similarity']:.2f}</div>
                    <div class="label">Jaccard similarity</div>
                </div>
                <div class="summary-item" style="background: linear-gradient(135deg, #1abc9c, #16a085);">
                    <div class="value">{results['summary']['semantic_distance']:.2f}</div>
                    <div class="label">Семант. расстояние</div>
                </div>
            </div>
        </div>
        
        <!-- Ключевые выводы -->
        <div class="card">
            <h2>💡 Ключевые выводы</h2>
            {''.join([self._format_finding(f) for f in results['key_findings']])}
        </div>
        
        <!-- Визуализации -->
        <div class="card">
            <h2>📈 Визуализации</h2>
            <div class="img-container">
                <img src="metrics_comparison.png" alt="Сравнение метрик">
            </div>
            <div class="img-container">
                <img src="pattern_overlap.png" alt="Пересечение паттернов">
            </div>
            <div class="img-container">
                <img src="cluster_comparison.png" alt="Кластерный анализ">
            </div>
        </div>
        
        <!-- Сравнение метрик -->
        <div class="card">
            <h2> Сравнение метрик</h2>
            <table>
                <thead>
                    <tr>
                        <th>Метрика</th>
                        <th>Доалгоритм.</th>
                        <th>Алгоритм.</th>
                        <th>Изменение</th>
                        <th>%</th>
                    </tr>
                </thead>
                <tbody>
                    {self._format_metrics_table(results['metrics_comparison'])}
                </tbody>
            </table>
        </div>
        
        <!-- Уникальные паттерны -->
        <div class="card">
            <h2> Уникальные паттерны</h2>
            
            <h3>Только в доалгоритмическом корпусе ({results['unique_patterns']['prealg_unique']['count']})</h3>
            <div class="pattern-list">
                {self._format_pattern_tags(results['unique_patterns']['prealg_unique']['top_patterns'], 'prealg')}
            </div>
            
            <h3>Только в алгоритмическом корпусе ({results['unique_patterns']['algo_unique']['count']})</h3>
            <div class="pattern-list">
                {self._format_pattern_tags(results['unique_patterns']['algo_unique']['top_patterns'], 'algo')}
            </div>
            
            <div class="interpretation">
                {results['unique_patterns'].get('interpretation', '')}
            </div>
        </div>
        
        <!-- Семантический дрейф -->
        <div class="card">
            <h2> Семантический анализ</h2>
            <p><strong>Центроидное сходство:</strong> {results['semantic_drift'].get('centroid_similarity', 'N/A')}</p>
            <p><strong>Семантическое расстояние:</strong> {results['semantic_drift'].get('semantic_distance', 'N/A')}</p>
            
            <h3>Центральные паттерны доалгоритмического корпуса</h3>
            <div class="pattern-list">
                {self._format_simple_tags(results['semantic_drift'].get('prealg_central_patterns', []), 'prealg')}
            </div>
            
            <h3>Центральные паттерны алгоритмического корпуса</h3>
            <div class="pattern-list">
                {self._format_simple_tags(results['semantic_drift'].get('algo_central_patterns', []), 'algo')}
            </div>
            
            <div class="interpretation">
                {results['semantic_drift'].get('interpretation', '')}
            </div>
        </div>
        
        <!-- Централизация -->
        <div class="card">
            <h2> Анализ централизации</h2>
            
            <h3>Топ паттернов по Betweenness (доалгоритмический)</h3>
            {self._format_centrality_list(results['centrality_shift'].get('prealg_top_patterns', []))}
            
            <h3>Топ паттернов по Betweenness (алгоритмический)</h3>
            {self._format_centrality_list(results['centrality_shift'].get('algo_top_patterns', []))}
            
            <div class="interpretation">
                {results['centrality_shift'].get('interpretation', '')}
            </div>
        </div>
        
        <!-- Кластерный анализ -->
        <div class="card">
            <h2> Кластерный анализ</h2>
            <table>
                <tr>
                    <th>Показатель</th>
                    <th>Доалгоритм.</th>
                    <th>Алгоритм.</th>
                </tr>
                <tr>
                    <td>Число кластеров</td>
                    <td>{results['cluster_analysis']['prealg_n_clusters']}</td>
                    <td>{results['cluster_analysis']['algo_n_clusters']}</td>
                </tr>
                <tr>
                    <td>Silhouette score</td>
                    <td>{results['cluster_analysis']['prealg_silhouette'] or 'N/A'}</td>
                    <td>{results['cluster_analysis']['algo_silhouette'] or 'N/A'}</td>
                </tr>
            </table>
            
            <div class="interpretation">
                {results['cluster_analysis'].get('interpretation', '')}
            </div>
        </div>
        
        <footer style="text-align: center; margin-top: 40px; color: #999; font-size: 12px;">
            Сгенерировано автоматически • Pattern Comparator v1.0
        </footer>
    </div>
</body>
</html>'''
        
        report_path = self.output_dir / "comparison_report.html"
        report_path.write_text(html, encoding="utf-8")
    
    def _format_finding(self, finding: str) -> str:
        """Форматирует вывод."""
        css_class = "finding"
        if "ПОДТВЕРЖДАЕТСЯ" in finding or "💡" in finding:
            css_class += " hypothesis"
        elif "↑" in finding or "увеличил" in finding or "✅" in finding:
            css_class += " positive"
        elif "↓" in finding or "уменьшил" in finding or "⚠️" in finding:
            css_class += " negative"
        
        return f'<div class="{css_class}">{finding}</div>'
    
    def _format_metrics_table(self, metrics: Dict) -> str:
        """Форматирует таблицу метрик."""
        rows = []
        for key, data in metrics.items():
            direction_class = "positive" if data["diff"] > 0 else "negative" if data["diff"] < 0 else ""
            
            prealg = f"{data['prealg']:.4f}" if isinstance(data['prealg'], float) else str(data['prealg'])
            algo = f"{data['algo']:.4f}" if isinstance(data['algo'], float) else str(data['algo'])
            diff = f"{data['diff']:+.4f}" if isinstance(data['diff'], float) else str(data['diff'])
            
            rows.append(f'''
                <tr>
                    <td>{data['name']}</td>
                    <td>{prealg}</td>
                    <td>{algo}</td>
                    <td class="{direction_class}">{data['direction']} {diff}</td>
                    <td class="{direction_class}">{data['change_pct']:+.1f}%</td>
                </tr>
            ''')
        return "".join(rows)
    
    def _format_pattern_tags(self, patterns: List[Dict], corpus_type: str) -> str:
        """Форматирует теги паттернов."""
        return "".join([
            f'<span class="pattern-tag {corpus_type}">{p["ngram"]} ({p["pmi"]:.2f})</span>'
            for p in patterns
        ])
    
    def _format_simple_tags(self, patterns: List[str], corpus_type: str) -> str:
        """Форматирует простые теги."""
        return "".join([
            f'<span class="pattern-tag {corpus_type}">{p}</span>'
            for p in patterns
        ])
    
    def _format_centrality_list(self, patterns: List[Dict]) -> str:
        """Форматирует список по централизации."""
        if not patterns:
            return "<p>Нет данных</p>"
        
        items = []
        for i, p in enumerate(patterns[:5], 1):
            items.append(
                f'<li><strong>{p.get("pattern", "N/A")}</strong> '
                f'(BC: {p.get("betweenness_centrality", 0):.4f})</li>'
            )
        
        return f'<ol>{"".join(items)}</ol>'
    
    def _save_json_results(self, results: Dict):
        """Сохраняет результаты в JSON."""
        # Конвертируем numpy типы
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, set):
                return list(obj)
            return obj
        
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert(obj)
        
        results_clean = deep_convert(results)
        
        json_path = self.output_dir / "comparison_results.json"
        json_path.write_text(
            json.dumps(results_clean, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ ПАЙПЛАЙН
# ═══════════════════════════════════════════════════════════════════════════════

def run_comparison(config: ComparisonConfig = None):
    """Запускает сравнительный анализ."""
    config = config or CONFIG
    
    logger.info("="*60)
    logger.info("ЗАПУСК СРАВНИТЕЛЬНОГО АНАЛИЗА")
    logger.info(f"Тема: {config.theme}")
    logger.info("="*60)
    
    # Загрузка данных
    logger.info("Загрузка доалгоритмического корпуса...")
    prealg_data = load_corpus_data(
        config.prealg_patterns_path,
        config.prealg_metrics_path,
        config.prealg_clusters_path
    )
    logger.info(f"  Загружено {len(prealg_data['patterns'])} паттернов")
    
    logger.info("Загрузка алгоритмического корпуса...")
    algo_data = load_corpus_data(
        config.algo_patterns_path,
        config.algo_metrics_path,
        config.algo_clusters_path
    )
    logger.info(f"  Загружено {len(algo_data['patterns'])} паттернов")
    
    # Проверка данных
    if not prealg_data['patterns'] or not algo_data['patterns']:
        logger.error("Ошибка: один или оба корпуса пусты!")
        return None
    
    # Сравнительный анализ
    comparator = PatternComparator(config)
    results = comparator.compare(prealg_data, algo_data)
    
    # Визуализация
    visualizer = ComparisonVisualizer(config)
    visualizer.create_all_visualizations(results)
    
    # Отчёты
    reporter = ReportGenerator(config)
    reporter.generate_all_reports(results)
    
    # Вывод ключевых выводов
    print("\n" + "="*60)
    print("КЛЮЧЕВЫЕ ВЫВОДЫ")
    print("="*60)
    for finding in results["key_findings"]:
        print(finding)
    print("="*60)
    print(f"\nОтчёты сохранены в: {config.output_dir}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Сравнительный анализ паттернов")
    parser.add_argument("--prealg-patterns", type=str, 
                        default="./data_nkry/patterns_prealg любовь.json")
    parser.add_argument("--prealg-metrics", type=str,
                        default="./data_nkry/metrics_prealg любовь.json")
    parser.add_argument("--prealg-clusters", type=str,
                        default="./data_nkry/clusters_prealg любовь.json")
    parser.add_argument("--algo-patterns", type=str,
                        default="./data_nkry/patterns_algo любовь.json")
    parser.add_argument("--algo-metrics", type=str,
                        default="./data_nkry/metrics_algo любовь.json")
    parser.add_argument("--algo-clusters", type=str,
                        default="./data_nkry/clusters_algo любовь.json")
    parser.add_argument("--output", "-o", type=str, default="./comparison_results")
    parser.add_argument("--theme", type=str, default="любовь")
    
    args = parser.parse_args()
    
    config = ComparisonConfig(
        prealg_patterns_path=args.prealg_patterns,
        prealg_metrics_path=args.prealg_metrics,
        prealg_clusters_path=args.prealg_clusters,
        algo_patterns_path=args.algo_patterns,
        algo_metrics_path=args.algo_metrics,
        algo_clusters_path=args.algo_clusters,
        output_dir=args.output,
        theme=args.theme
    )
    
    run_comparison(config)
