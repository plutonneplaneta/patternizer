import logging
from pathlib import Path
from typing import List, Dict, Set, Optional
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math
import json
import requests
import time
from datetime import datetime, timedelta

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

# темы и соответствующие хештеги
THEMES_HASHTAGS = {
    'любовь': ['любовь', 'любить', 'отношения', 'романтика', 'чувства'],
    'успех': ['успех', 'достижения', 'победа', 'цели', 'мотивация'],
    'здоровье': ['здоровье', 'спорт', 'фитнес', 'диета', 'медицина'],
    'семья': ['семья', 'дети', 'родители', 'брак', 'домашний'],
    'молодёжь': ['молодежь', 'студенты', 'образование', 'карьера', 'развитие']
}

themes = list(THEMES_HASHTAGS.keys())

CONFIG = {
    "storage_dir": "./data_vk",
    "use_cached_data": True,  # Флаг для использования уже собранных данных
    "vk": {
        "access_token": "YOUR_TOKEN",  # Замените на ваш токен
        "api_version": "5.131",
        "request_delay": 0.34,  # Задержка между запросами (секунды)
        "max_posts_per_hashtag": 200,
        "search_period_days": 365  # За какой период искать посты (в днях)
    },
    "pattern_extraction": {
        "min_pmi": 3,
        "min_doc_freq": 5,  
        "min_count": 5,  
        "ngram_range": (2, 4)
    },
    "embeddings": {
        "model_name": "cointegrated/rubert-tiny2", 
        "batch_size": 32
    },
    "pattern_similarity_threshold": 0.75,
    "max_examples_per_theme": 500,
    "nlp": {
        "pos_filter": ["NOUN", "ADJ", "VERB", "ADV", "PROPN"]
    },
    "max_patterns_per_theme": 200,
    "filter_subsumed_ngrams": True
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
# 1. Сбор постов из ВКонтакте по хештегам
# ----------------------------
class VKAPI:
    def __init__(self, access_token: str, api_version: str = "5.131"):
        self.access_token = access_token
        self.api_version = api_version
        self.base_url = "https://api.vk.com/method"
        
    def make_request(self, method: str, params: dict) -> dict:
        """Выполняет запрос к VK API"""
        params.update({
            'access_token': self.access_token,
            'v': self.api_version
        })
        
        url = f"{self.base_url}/{method}"
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                logger.error(f"VK API Error: {data['error']}")
                return None
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
    
    def search_posts_by_hashtag(self, hashtag: str, count: int = 100, offset: int = 0, 
                               start_time: int = None, end_time: int = None) -> List[Dict]:
        """Ищет посты по хештегу"""
        params = {
            'q': f'#{hashtag}',
            'count': count,
            'offset': offset,
            'extended': 1
        }
        
        if start_time:
            params['start_time'] = start_time
        if end_time:
            params['end_time'] = end_time
            
        data = self.make_request('newsfeed.search', params)
        
        if not data or 'response' not in data:
            return []
            
        return data['response'].get('items', [])
    
    def get_user_info(self, user_ids: List[int]) -> Dict[int, Dict]:
        """Получает информацию о пользователях"""
        if not user_ids:
            return {}
            
        user_ids_str = ','.join(map(str, user_ids))
        data = self.make_request('users.get', {
            'user_ids': user_ids_str,
            'fields': 'screen_name'
        })
        
        if not data or 'response' not in data:
            return {}
            
        return {user['id']: user for user in data['response']}
    
    def get_group_info(self, group_ids: List[int]) -> Dict[int, Dict]:
        """Получает информацию о группах"""
        if not group_ids:
            return {}
            
        # Группы в VK API имеют отрицательные ID
        group_ids_str = ','.join([str(abs(gid)) for gid in group_ids])
        data = self.make_request('groups.getById', {
            'group_ids': group_ids_str,
            'fields': 'screen_name'
        })
        
        if not data or 'response' not in data:
            return {}
            
        return {group['id']: group for group in data['response']}

def fetch_posts_from_vk(themes_hashtags: Dict[str, List[str]]) -> List[Dict]:
    """Собирает посты из ВК по хештегам для каждой темы"""
    vk_config = CONFIG["vk"]
    access_token = vk_config["access_token"]
    
    if access_token == "YOUR_VK_ACCESS_TOKEN":
        logger.error("Please set your VK access token in CONFIG")
        return []
    
    vk = VKAPI(access_token, vk_config["api_version"])
    
    # Вычисляем временной диапазон
    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now() - timedelta(days=vk_config["search_period_days"])).timestamp())
    
    all_posts = []
    
    for theme, hashtags in themes_hashtags.items():
        logger.info(f"Сбор постов для темы '{theme}' с хештегами: {hashtags}")
        theme_posts_count = 0
        
        for hashtag in hashtags:
            if theme_posts_count >= CONFIG["max_examples_per_theme"]:
                break
                
            logger.info(f"Поиск по хештегу '#{hashtag}'")
            offset = 0
            hashtag_posts_count = 0
            
            while True:
                # Задержка для соблюдения лимитов API
                time.sleep(vk_config["request_delay"])
                
                posts = vk.search_posts_by_hashtag(
                    hashtag, 
                    count=100, 
                    offset=offset,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if not posts:
                    break
                
                for post in posts:
                    if theme_posts_count >= CONFIG["max_examples_per_theme"]:
                        break
                    if hashtag_posts_count >= vk_config["max_posts_per_hashtag"]:
                        break
                        
                    # Извлекаем текст поста
                    text = post.get('text', '').strip()
                    if not text or len(text) < 50:  # Фильтруем слишком короткие посты
                        continue
                    
                    # Определяем автора
                    author = "Unknown"
                    source = ""
                    
                    if post.get('from_id', 0) > 0:  # Пост от пользователя
                        author = f"user{post['from_id']}"
                    elif post.get('from_id', 0) < 0:  # Пост от группы
                        author = f"group{abs(post['from_id'])}"
                    
                    if post.get('owner_id'):
                        source = f"https://vk.com/wall{post['owner_id']}_{post['id']}"
                    
                    all_posts.append({
                        'theme': theme,
                        'text': text,
                        'source': source,
                        'author': author,
                        'date': post.get('date', None),
                        'hashtag': hashtag
                    })
                    
                    theme_posts_count += 1
                    hashtag_posts_count += 1
                
                offset += len(posts)
                
                # Если получено меньше 100 постов, значит это последняя страница
                if len(posts) < 100:
                    break
                    
                if (theme_posts_count >= CONFIG["max_examples_per_theme"] or 
                    hashtag_posts_count >= vk_config["max_posts_per_hashtag"]):
                    break
            
            logger.info(f"Для хештега '#{hashtag}' собрано {hashtag_posts_count} постов")
        
        logger.info(f"Для темы '{theme}' собрано {theme_posts_count} постов")
    
    logger.info(f"Всего собрано постов: {len(all_posts)}")
    return all_posts

def load_existing_corpus(storage_dir: str) -> Optional[Dict[str, List[str]]]:
    """Загружает уже собранный корпус из файлов"""
    storage_path = Path(storage_dir)
    corpus_by_theme = {}
    themes_loaded = 0
    
    for theme in THEMES_HASHTAGS.keys():
        corpus_file = storage_path / f"corpus_{theme}.json"
        if corpus_file.exists():
            try:
                with open(corpus_file, 'r', encoding='utf-8') as f:
                    texts = json.load(f)
                    corpus_by_theme[theme] = texts
                    themes_loaded += 1
                logger.info(f"Загружено {len(texts)} текстов для темы '{theme}'")
            except Exception as e:
                logger.error(f"Ошибка загрузки файла {corpus_file}: {e}")
        else:
            logger.warning(f"Файл {corpus_file} не найден")
    
    if themes_loaded == len(THEMES_HASHTAGS):
        logger.info(f"Успешно загружен корпус для всех {themes_loaded} тем")
        return corpus_by_theme
    else:
        logger.warning(f"Загружены данные только для {themes_loaded} из {len(THEMES_HASHTAGS)} тем")
        return None

def collect_corpus_by_theme_from_vk(themes_hashtags: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Собирает корпус текстов по темам из ВКонтакте или загружает существующий"""
    ensure_dirs(CONFIG["storage_dir"])
    
    # Пытаемся загрузить существующие данные, если включен флаг
    if CONFIG.get("use_cached_data", True):
        existing_corpus = load_existing_corpus(CONFIG["storage_dir"])
        if existing_corpus is not None:
            return existing_corpus
        else:
            logger.info("Не удалось загрузить существующий корпус, собираем новые данные...")
    
    # Собираем посты из ВК
    posts = fetch_posts_from_vk(themes_hashtags)
    
    # Группируем по темам
    corpus_by_theme = {theme: [] for theme in themes_hashtags.keys()}
    theme_stats = {theme: 0 for theme in themes_hashtags.keys()}
    
    for post in posts:
        corpus_by_theme[post['theme']].append(post['text'])
        theme_stats[post['theme']] += 1
    
    # Сохраняем корпус по темам
    for theme, texts in corpus_by_theme.items():
        path = Path(CONFIG["storage_dir"]) / f"corpus_{theme}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        logger.info(f"Сохранено {len(texts)} текстов для темы '{theme}' -> {path}")
    
    # Сохраняем метаданные постов
    metadata_path = Path(CONFIG["storage_dir"]) / "posts_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_posts': len(posts),
            'theme_stats': theme_stats,
            'collection_date': datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)
    
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
    """Извлекает n-граммы"""
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
        
        # ФИЛЬТР 1: короткие слова
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
                "length": len(words)
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
    """Фильтрует n-граммы, которые входят в состав n-грамм большей степени"""
    if not patterns or len(patterns) <= 1:
        return patterns
    
    # Сортируем по длине (по убыванию) и затем по PMI (по убыванию)
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
                for i in range(len(added_words) - len(words) + 1):
                    if added_words[i:i+len(words)] == words:
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
    """Простая фильтрация паттернов по качеству"""
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
    """Комплексное извлечение качественных паттернов"""
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
# 4. Построение сети паттернов и интерактивный HTML
# ----------------------------
def build_pattern_network_for_theme(patterns: List[Dict], theme: str) -> tuple[nx.Graph, Dict]:
    """Строит сеть паттернов для темы"""
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
            
        degree_probs = [deg / total_degrees for deg in degrees]
        
        entropy = 0.0
        for prob in degree_probs:
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy

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
    # Собираем корпус из ВКонтакте или загружаем существующий
    corpus_by_theme = collect_corpus_by_theme_from_vk(THEMES_HASHTAGS)
    
    # Проверяем, есть ли данные для обработки
    if not corpus_by_theme:
        logger.error("Нет данных для обработки. Проверьте настройки и доступность данных.")
        return
    
    total_texts = sum(len(texts) for texts in corpus_by_theme.values())
    logger.info(f"Всего текстов для обработки: {total_texts}")
    
    for theme in themes:
        texts = corpus_by_theme.get(theme, [])
        if not texts:
            logger.warning(f"Нет текстов для темы '{theme}' — пропускаем")
            continue

        # Проверяем, есть ли уже лемматизированные данные
        lemmatized_file = Path(CONFIG["storage_dir"]) / f"lemmatized_{theme}.json"
        if CONFIG.get("use_cached_data", True) and lemmatized_file.exists():
            logger.info(f"Загружаем лемматизированные данные для темы '{theme}'")
            try:
                with open(lemmatized_file, 'r', encoding='utf-8') as f:
                    lemmatized = json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки лемматизированных данных: {e}")
                lemmatized = []
        else:
            # Лемматизация текстов
            logger.info(f"Лемматизация текстов для темы '{theme}'...")
            lemmatized = []
            for i, text in enumerate(texts):
                if i % 100 == 0:
                    logger.info(f"Лемматизировано {i}/{len(texts)} текстов")
                lemmatized.append(lemmatize_text(text, CONFIG["nlp"]["pos_filter"]))
            
            # Сохраняем лемматизированные данные
            with open(lemmatized_file, 'w', encoding='utf-8') as f:
                json.dump(lemmatized, f, ensure_ascii=False, indent=2)
            logger.info(f"Сохранены лемматизированные данные для темы '{theme}'")
        
        logger.info(f"Тема '{theme}': используем {len(THEME_KEYWORDS[theme])} ключевых слов")

        # Проверяем, есть ли уже извлеченные паттерны
        patterns_file = Path(CONFIG["storage_dir"]) / f"patterns_{theme}.json"
        if CONFIG.get("use_cached_data", True) and patterns_file.exists():
            logger.info(f"Загружаем существующие паттерны для темы '{theme}'")
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    patterns = json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки паттернов: {e}")
                patterns = []
        else:
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

        # Проверяем, есть ли уже построенная сеть
        network_file = Path(CONFIG["storage_dir"]) / f"patterns_{theme}.html"
        if CONFIG.get("use_cached_data", True) and network_file.exists():
            logger.info(f"Сеть для темы '{theme}' уже существует: {network_file}")
            # Загружаем метрики, если есть
            metrics_file = Path(CONFIG["storage_dir"]) / f"metrics_{theme}.json"
            metrics = {}
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                except Exception as e:
                    logger.error(f"Ошибка загрузки метрик: {e}")
        else:
            # Строим сеть
            G, metrics = build_pattern_network_for_theme(patterns, theme)
            
            # Сохраняем метрики
            save_theme_artifacts(theme, patterns, metrics)

if __name__ == "__main__":
    main()
