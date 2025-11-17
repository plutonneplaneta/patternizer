from corus import load_taiga_social
import logging
from pathlib import Path
from typing import List, Dict, Set
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math
import json

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
        "min_pmi": 3,
        "min_doc_freq": 10,  
        "min_count": 10,  
        "ngram_range": (2, 4)
    },
    "embeddings": {
        "model_name": "cointegrated/rubert-tiny2", 
        "batch_size": 32
    },
    "pattern_similarity_threshold": 0.75,
    "max_examples_per_theme": 1000,
    "nlp": {
        "pos_filter": ["NOUN", "ADJ", "VERB", "ADV", "PROPN"]
    },
    "max_patterns_per_theme": 200,
    "filter_subsumed_ngrams": True  # Включаем фильтрацию вложенных n-грамм
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

# Хард-код ключевых слов для каждой темы (в нормальной форме)
THEME_KEYWORDS = {
    'любовь': [
        'любовь', 'любить', 'влюбленный', 'романтика', 'чувство', 'сердце', 'отношение',
        'взаимность', 'признание', 'свидание', 'роман', 'нежность', 'страсть', 'обожание',
        'симпатия', 'привязанность', 'флирт', 'объятие', 'поцелуй', 'ласка', 'вожделение',
        'влечение', 'притяжение', 'эмоция', 'верность', 'преданность', 'доверие', 'понимание',
        'гармония', 'счастье', 'радость', 'восторг', 'встреча', 'расставание', 'разлука',
        'тоска', 'одиночество', 'ревность', 'измена', 'прощение', 'примирение', 'тепло',
        'забота', 'внимание', 'поддержка', 'уважение', 'самоотдача', 'подарок', 'сюрприз',
        'комплимент', 'объяснение', 'предложение', 'помолвка', 'свадьба', 'семья'
    ],
    'успех': [
        'успех', 'успешный', 'достижение', 'победа', 'триумф', 'результат', 'прогресс',
        'продвижение', 'достигать', 'преуспевать', 'побеждать', 'преуспевание', 'победный',
        'триумфальный', 'результативный', 'эффективный', 'продуктивный', 'процветание',
        'благополучие', 'развитие', 'развивать', 'рост', 'расти', 'улучшение', 'улучшать',
        'совершенствование', 'совершенствовать', 'мастерство', 'профессионализм', 'эксперт',
        'компетенция', 'квалификация', 'навык', 'умение', 'способность', 'талант', 'одаренность',
        'амбиция', 'целеустремленность', 'настойчивость', 'упорство', 'трудолюбие', 'усердие',
        'дисциплина', 'организованность', 'планирование', 'стратегия', 'тактика', 'решение',
        'проблема', 'задача', 'цель', 'миссия', 'видение', 'мотивация', 'вдохновение',
        'энтузиазм', 'энергия', 'активный', 'инициатива', 'креативность', 'инновация',
        'творчество', 'лидерство', 'руководство', 'управление', 'организация', 'координация',
        'контроль', 'оценка', 'анализ', 'оптимизация', 'эффективность', 'продуктивность',
        'качество', 'репутация', 'имидж', 'бренд', 'конкурентоспособность', 'преимущество',
        'лидерство', 'первенство', 'инновационный', 'прогрессивный', 'современный', 'актуальный',
        'востребованный', 'популярный', 'известный', 'знаменитый', 'признание', 'награда',
        'премия', 'победитель', 'чемпион', 'рекорд', 'достижение', 'результат', 'показатель'
    ],
    'здоровье': [
        'здоровье', 'здоровый', 'болезнь', 'лечение', 'медицина', 'врач', 'больница', 'выздоровление',
        'профилактика', 'диагноз', 'симптом', 'признак', 'состояние', 'самочувствие', 'настроение',
        'энергия', 'сила', 'бодрость', 'активность', 'иммунитет', 'защита', 'устойчивость', 'выносливость',
        'спорт', 'фитнес', 'тренировка', 'упражнение', 'зарядка', 'разминка', 'растяжка', 'питание',
        'диета', 'рацион', 'продукт', 'пища', 'еда', 'вода', 'витамин', 'минерал', 'белок', 'жир',
        'углевод', 'клетчатка', 'антиоксидант', 'детокс', 'очищение', 'регенерация', 'восстановление',
        'омоложение', 'долголетие', 'молодость', 'красота', 'привлекательность', 'свежесть', 'чистота',
        'гигиена', 'уход', 'забота', 'процедура', 'массаж', 'сауна', 'баня', 'бассейн', 'спа', 'отдых',
        'расслабление', 'сон', 'бессонница', 'усталость', 'стресс', 'напряжение', 'тревога', 'депрессия',
        'психология', 'психический', 'эмоциональный', 'духовный', 'физический', 'организм', 'тело',
        'орган', 'система', 'функция', 'метаболизм', 'пищеварение', 'дыхание', 'кровообращение',
        'нервный', 'сердечный', 'сосудистый', 'дыхательный', 'пищеварительный', 'мочеполовой',
        'опорно-двигательный', 'кожа', 'волосы', 'ногти', 'зубы', 'зрение', 'слух', 'обоняние'
    ],
    'семья': [
        'семья', 'семейный', 'родители', 'мать', 'отец', 'дети', 'ребенок', 'брак', 'супруг', 'супруга',
        'родственники', 'родной', 'близкий', 'любимый', 'дорогой', 'кровный', 'родительский', 'отцовский',
        'материнский', 'детский', 'брачный', 'супружеский', 'родственный', 'традиция', 'ценность',
        'принцип', 'уклад', 'основа', 'фундамент', 'опора', 'поддержка', 'помощь', 'взаимопомощь',
        'солидарность', 'единство', 'сплоченность', 'дружба', 'согласие', 'гармония', 'понимание',
        'уважение', 'доверие', 'верность', 'преданность', 'любовь', 'нежность', 'ласка', 'забота',
        'внимание', 'тепло', 'уют', 'комфорт', 'безопасность', 'надежность', 'стабильность', 'постоянство',
        'развитие', 'взросление', 'воспитание', 'образование', 'обучение', 'становление', 'формирование',
        'личность', 'коллективный', 'общий', 'совместный', 'домашний', 'бытовой', 'повседневный',
        'праздник', 'торжество', 'юбилей', 'день_рождения', 'новоселье', 'свадьба', 'помолвка',
        'венчание', 'крестины', 'поминание', 'память', 'предок', 'потомок', 'наследник', 'наследство',
        'традиция', 'преемственность', 'поколение', 'династия', 'род', 'фамилия', 'имя', 'отчество',
        'происхождение', 'корни', 'история', 'воспоминание', 'наследие', 'достояние'
    ],
    'молодёжь': [
        'молодежь', 'молодой', 'юность', 'юный', 'подросток', 'молодость', 'юноша', 'девушка',
        'парень', 'девчонка', 'тинейджер', 'подрастающий', 'поколение', 'будущее', 'перспектива',
        'потенциал', 'возможность', 'шанс', 'перспективный', 'амбициозный', 'целеустремленный',
        'активный', 'энергичный', 'динамичный', 'прогрессивный', 'современный', 'актуальный',
        'модный', 'трендовый', 'инновационный', 'креативный', 'творческий', 'талантливый',
        'одаренный', 'способный', 'умный', 'интеллектуальный', 'образованный', 'развитый',
        'культурный', 'воспитанный', 'социальный', 'общественный', 'гражданский', 'политический',
        'профессиональный', 'карьерный', 'образовательный', 'учебный', 'научный', 'исследовательский',
        'спортивный', 'художественный', 'музыкальный', 'театральный', 'танцевальный', 'вокальный',
        'инструментальный', 'изобразительный', 'литературный', 'поэтический', 'журналистский',
        'медийный', 'цифровой', 'технологический', 'инженерный', 'программистский', 'дизайнерский',
        'предпринимательский', 'бизнес', 'стартап', 'проект', 'инициатива', 'волонтерство',
        'добровольчество', 'благотворительность', 'экология', 'природа', 'окружающий', 'среда',
        'здоровье', 'спорт', 'фитнес', 'танцы', 'музыка', 'искусство', 'культура', 'образование',
        'наука', 'технологии', 'инновации', 'творчество', 'самореализация', 'саморазвитие',
        'личностный', 'рост', 'профессиональный', 'становление', 'идентичность', 'самоопределение',
        'мировоззрение', 'ценности', 'принципы', 'убеждения', 'идеалы', 'стремления', 'мечты',
        'цели', 'планы', 'будущее', 'карьера', 'профессия', 'специальность', 'квалификация',
        'навыки', 'компетенции', 'знания', 'опыт', 'практика', 'стажировка', 'работа', 'труд',
        'занятость', 'безработица', 'рынок', 'труда', 'конкуренция', 'конкурентоспособность',
        'адаптация', 'интеграция', 'социализация', 'взаимодействие', 'коммуникация', 'общение',
        'дружба', 'любовь', 'отношения', 'семья', 'брак', 'дети', 'родители', 'семейные', 'ценности'
    ]
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
# 2. Лемматизация
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

# ----------------------------
# 3. Извлечение паттернов с фильтрацией вложенных n-грамм
# ----------------------------
def extract_clean_ngrams_from_theme(lemmatized_texts: List[str],
                                    min_pmi: float,
                                    min_doc_freq: int,
                                    min_count: int,
                                    ngram_range: tuple) -> List[Dict]:
    """
    Извлекает n-граммы без фильтрации по ключевым словам
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
        meaningful_words = [w for w in words if w not in RUSSIAN_STOPWORDS and len(w) >= 2]
        if not meaningful_words:
            continue

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
            patterns.append({
                "ngram": ngram, 
                "count": c, 
                "doc_freq": df, 
                "pmi": float(pmi),
                "meaningful_words": meaningful_words,
                "length": len(words)  # Добавляем длину n-граммы
            })

    # Сортируем по PMI и doc_freq
    patterns = sorted(patterns, key=lambda x: (x["pmi"], x["doc_freq"]), reverse=True)
    logger.info(f"Извлечено {len(patterns)} паттернов для темы")
    
    # Логируем примеры найденных паттернов
    if patterns:
        sample_patterns = [p["ngram"] for p in patterns[:10]]
        logger.info(f"Топ-10 паттернов: {sample_patterns}")
    
    return patterns

def filter_subsumed_ngrams(patterns: List[Dict]) -> List[Dict]:
    """
    Фильтрует n-граммы, которые входят в состав n-грамм большей степени
    """
    if not patterns or len(patterns) <= 1:
        return patterns
    
    # Сортируем по длине (по убыванию) и затем по PMI (по убыванию)
    # Так мы сначала обрабатываем более длинные n-граммы
    patterns_sorted = sorted(patterns, key=lambda x: (x["length"], x["pmi"]), reverse=True)
    
    filtered_patterns = []
    ngram_set = set()
    
    for pattern in patterns_sorted:
        ngram = pattern["ngram"]
        words = ngram.split()
        
        # Проверяем, не является ли эта n-грамма подстрокой уже добавленной более длинной n-граммы
        is_subsumed = False
        
        for added_ngram in ngram_set:
            added_words = added_ngram.split()
            
            # Если текущая n-грамма короче уже добавленной
            if len(words) < len(added_words):
                # Проверяем, входит ли текущая n-грамма в состав добавленной
                # Ищем все возможные позиции, где текущая n-грамма может быть подпоследовательностью
                for i in range(len(added_words) - len(words) + 1):
                    if added_words[i:i+len(words)] == words:
                        # Нашли вхождение - текущая n-грамма является частью более длинной
                        is_subsumed = True
                        break
                if is_subsumed:
                    break
        
        if not is_subsumed:
            filtered_patterns.append(pattern)
            ngram_set.add(ngram)
    
    logger.info(f"Фильтрация вложенных n-грамм: {len(patterns)} -> {len(filtered_patterns)}")
    return filtered_patterns

def filter_patterns_by_quality(patterns: List[Dict]) -> List[Dict]:
    """
    Простая фильтрация паттернов по качеству
    """
    max_patterns = CONFIG["max_patterns_per_theme"]
    
    if len(patterns) > max_patterns:
        logger.info(f"Ограничиваем количество паттернов: {len(patterns)} -> {max_patterns}")
        patterns = patterns[:max_patterns]
    
    return patterns

def extract_high_quality_patterns(lemmatized_texts: List[str],
                                theme: str,
                                min_pmi: float,
                                min_doc_freq: int,
                                min_count: int,
                                ngram_range: tuple) -> List[Dict]:
    """
    Комплексное извлечение качественных паттернов
    """
    # Базовое извлечение паттернов
    patterns = extract_clean_ngrams_from_theme(
        lemmatized_texts=lemmatized_texts,
        min_pmi=min_pmi,
        min_doc_freq=min_doc_freq,
        min_count=min_count,
        ngram_range=ngram_range
    )
    
    if not patterns:
        logger.warning(f"Нет паттернов после базового извлечения для темы '{theme}'")
        return []
    
    # Фильтрация вложенных n-грамм (если включена в конфиге)
    if CONFIG.get("filter_subsumed_ngrams", True):
        patterns = filter_subsumed_ngrams(patterns)
    
    # Простая фильтрация по количеству
    patterns = filter_patterns_by_quality(patterns)
    
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
        # Размер узла зависит от PMI и doc_freq
        base_size = 10
        size_boost = p["pmi"] * 2 + p["doc_freq"] * 0.5
        
        G.add_node(idx,
                   label=p["ngram"],
                   words=words,
                   count=int(p["count"]),
                   doc_freq=int(p["doc_freq"]),
                   pmi=float(p["pmi"]),
                   length=int(p["length"]),
                   title=f"{p['ngram']} (pmi={p['pmi']:.2f}, doc_freq={p['doc_freq']}, length={p['length']})",
                   size=max(8, min(40, base_size + size_boost))
                   )

    # Эдджевая матрица
    edge_count = 0
    for i in range(len(ngrams)):
        for j in range(i + 1, len(ngrams)):
            sim = float(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0])
            if sim >= CONFIG["pattern_similarity_threshold"]:
                color_val = int(max(0, min(255, (sim - 0.3) * 3 * 255)))
                color_hex = f"rgb({255 - color_val},{color_val},120)"
                G.add_edge(i, j, weight=float(sim), title=f"sim={sim:.3f}", color=color_hex, 
                          width=1 + (sim - CONFIG["pattern_similarity_threshold"]) * 4)
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

    # JavaScript код для интерактивности
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
                const node = nodes.get(id);
                if (!node) return;
                // assemble metadata string
                let meta = `<b>Паттерн:</b> ${{node.label}}<br>`;
                if (node.words) meta += `<b>Слова:</b> ${{node.words.join(', ')}}<br>`;
                if (node.count!==undefined) meta += `<b>count:</b> ${{node.count}} `;
                if (node.doc_freq!==undefined) meta += `<b>doc_freq:</b> ${{node.doc_freq}} `;
                if (node.pmi!==undefined) meta += `<b>pmi:</b> ${{parseFloat(node.pmi).toFixed(2)}} `;
                if (node.length!==undefined) meta += `<b>length:</b> ${{node.length}} `;
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

def save_theme_artifacts(theme: str, patterns: List[Dict], metrics: Dict):
    """Сохраняет все артефакты для темы"""
    # Метрики
    metrics_path = Path(CONFIG["storage_dir"]) / f"metrics_{theme}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(cast_numpy_types(metrics), f, ensure_ascii=False, indent=2)
    
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
        
        logger.info(f"Тема '{theme}': используем {len(THEME_KEYWORDS[theme])} ключевых слов")

        # Комплексное извлечение качественных паттернов
        patterns = extract_high_quality_patterns(
            lemmatized_texts=lemmatized,
            theme=theme,
            min_pmi=CONFIG["pattern_extraction"]["min_pmi"],
            min_doc_freq=CONFIG["pattern_extraction"]["min_doc_freq"],
            min_count=CONFIG["pattern_extraction"]["min_count"],
            ngram_range=CONFIG["pattern_extraction"]["ngram_range"]
        )

        if not patterns:
            logger.warning(f"Для темы '{theme}' не осталось качественных паттернов.")
            continue

        # Строим сеть
        G, metrics = build_pattern_network_for_theme(patterns, theme)
        
        # Сохраняем метрики
        save_theme_artifacts(theme, patterns, metrics)

if __name__ == "__main__":
    main()
