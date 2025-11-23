import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Set
from dataclasses import dataclass
import argparse

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deduplication.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DeduplicationStats:
    """Статистика дедупликации"""
    theme: str
    total_before: int
    total_after: int
    duplicates_removed: int
    duplicate_ratio: float

class ExactDeduplicator:
    """Класс для удаления точных дубликатов"""
    
    def __init__(self, min_text_length: int = 30, normalize_whitespace: bool = True):
        self.min_text_length = min_text_length
        self.normalize_whitespace = normalize_whitespace
    
    def _normalize_text(self, text: str) -> str:
        """Нормализует текст для сравнения"""
        if self.normalize_whitespace:
            # Удаляем лишние пробелы и переносы строк
            text = ' '.join(text.split())
        return text.strip()
    
    def _get_text_hash(self, text: str) -> str:
        """Создает хеш нормализованного текста"""
        normalized = self._normalize_text(text)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def remove_exact_duplicates(self, texts: List[str]) -> List[str]:
        """Удаляет точные дубликаты из списка текстов"""
        seen_hashes: Set[str] = set()
        unique_texts: List[str] = []
        duplicates_count = 0
        
        for text in texts:
            if not text or len(text.strip()) < self.min_text_length:
                continue
                
            text_hash = self._get_text_hash(text)
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)  # Сохраняем оригинальный текст
            else:
                duplicates_count += 1
        
        return unique_texts, duplicates_count
    
    def process_theme_corpus(self, input_file: Path, output_file: Path = None) -> DeduplicationStats:
        """Обрабатывает корпус одной темы"""
        if not input_file.exists():
            logger.error(f"Файл {input_file} не найден")
            return None
        
        # Загружаем тексты
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                texts = json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки файла {input_file}: {e}")
            return None
        
        if not isinstance(texts, list):
            logger.error(f"Файл {input_file} должен содержать список текстов")
            return None
        
        total_before = len(texts)
        logger.info(f"Обработка темы '{input_file.stem}': {total_before} текстов")
        
        # Удаляем дубликаты
        unique_texts, duplicates_removed = self.remove_exact_duplicates(texts)
        total_after = len(unique_texts)
        
        # Вычисляем статистику
        duplicate_ratio = duplicates_removed / total_before if total_before > 0 else 0.0
        
        stats = DeduplicationStats(
            theme=input_file.stem.replace('corpus_', ''),
            total_before=total_before,
            total_after=total_after,
            duplicates_removed=duplicates_removed,
            duplicate_ratio=duplicate_ratio
        )
        
        # Сохраняем результат
        output_path = output_file or input_file
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(unique_texts, f, ensure_ascii=False, indent=2)
            logger.info(f"Сохранено: {output_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения файла {output_path}: {e}")
            return None
        
        return stats
    
    def process_directory(self, data_dir: Path, backup_original: bool = True) -> List[DeduplicationStats]:
        """Обрабатывает все корпуса в директории"""
        if not data_dir.exists():
            logger.error(f"Директория {data_dir} не найдена")
            return []
        
        # Ищем файлы корпусов
        corpus_files = list(data_dir.glob("corpus_*.json"))
        if not corpus_files:
            logger.warning(f"В директории {data_dir} не найдено файлов корпусов")
            return []
        
        logger.info(f"Найдено {len(corpus_files)} файлов корпусов для обработки")
        
        all_stats = []
        
        for corpus_file in corpus_files:
            if backup_original:
                # Создаем backup оригинального файла
                backup_file = corpus_file.with_suffix('.json.backup')
                try:
                    import shutil
                    shutil.copy2(corpus_file, backup_file)
                    logger.info(f"Создан backup: {backup_file}")
                except Exception as e:
                    logger.warning(f"Не удалось создать backup для {corpus_file}: {e}")
            
            # Обрабатываем корпус
            stats = self.process_theme_corpus(corpus_file)
            if stats:
                all_stats.append(stats)
        
        return all_stats

def print_statistics(stats_list: List[DeduplicationStats]):
    if not stats_list:
        return
    
    logger.info("=" * 80)
    logger.info("СТАТИСТИКА ДЕДУПЛИКАЦИИ")
    logger.info("=" * 80)
    
    total_before = sum(s.total_before for s in stats_list)
    total_after = sum(s.total_after for s in stats_list)
    total_duplicates = sum(s.duplicates_removed for s in stats_list)
    overall_ratio = total_duplicates / total_before if total_before > 0 else 0.0
    
    # Общая статистика
    logger.info(f"ОБЩАЯ СТАТИСТИКА:")
    logger.info(f"  Всего текстов до:    {total_before:>6}")
    logger.info(f"  Всего текстов после: {total_after:>6}")
    logger.info(f"  Удалено дубликатов:  {total_duplicates:>6}")
    logger.info(f"  Коэффициент дублей:  {overall_ratio:>7.2%}")
    logger.info("-" * 80)
    
    # Детальная статистика по темам
    logger.info("ПО ТЕМАМ:")
    for stats in stats_list:
        logger.info(f"  {stats.theme:<12} {stats.total_before:>6} → {stats.total_after:>6} "
                   f"(-{stats.duplicates_removed:>4}, {stats.duplicate_ratio:>6.2%})")
    
    logger.info("=" * 80)

def restore_backup(data_dir: Path):
    """Восстанавливает оригинальные файлы из backup"""
    backup_files = list(data_dir.glob("*.json.backup"))
    
    if not backup_files:
        logger.info("Backup файлы не найдены")
        return
    
    logger.info(f"Найдено {len(backup_files)} backup файлов для восстановления")
    
    for backup_file in backup_files:
        original_file = backup_file.with_suffix('')  # Убираем .backup
        try:
            import shutil
            shutil.copy2(backup_file, original_file)
            logger.info(f"Восстановлен: {original_file}")
        except Exception as e:
            logger.error(f"Ошибка восстановления {original_file}: {e}")

def main():
    """Основная функция скрипта"""
    parser = argparse.ArgumentParser(description='Удаление точных дубликатов из корпусов текстов')
    parser.add_argument('--data-dir', type=str, default='./data_vk',
                       help='Директория с корпусами (по умолчанию: ./data_vk)')
    parser.add_argument('--min-length', type=int, default=30,
                       help='Минимальная длина текста (по умолчанию: 30)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Не создавать backup файлы')
    parser.add_argument('--restore', action='store_true',
                       help='Восстановить оригинальные файлы из backup')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if args.restore:
        logger.info("Режим восстановления из backup")
        restore_backup(data_dir)
        return
    
    # Создаем дедупликатор
    deduplicator = ExactDeduplicator(
        min_text_length=args.min_length,
        normalize_whitespace=True
    )
    
    logger.info(f"Начало дедупликации корпусов в директории: {data_dir}")
    logger.info(f"Минимальная длина текста: {args.min_length}")
    logger.info(f"Создание backup: {'НЕТ' if args.no_backup else 'ДА'}")
    
    # Обрабатываем все корпуса
    stats = deduplicator.process_directory(
        data_dir=data_dir,
        backup_original=not args.no_backup
    )
    
    # Выводим статистику
    print_statistics(stats)
    
    if stats:
        logger.info("Дедупликация завершена успешно!")
    else:
        logger.error("Дедупликация не выполнена. Проверьте логи для деталей.")

if __name__ == "__main__":
    main()
