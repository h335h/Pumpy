import os
import sys
import logging
from config import Config
from db import Database
from semantic import SemanticFilter
from collectors.telegram_collector import TelegramCollector
from collectors.rss_collector import RssCollector
from sender import DigestSender

# ------------------------------------------------------------
# Настройка логирования: в файл и в консоль
# ------------------------------------------------------------
# Создаём папку logs, если её нет
os.makedirs('logs', exist_ok=True)

# Создаём корневой логгер
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Обработчик для файла
file_handler = logging.FileHandler('logs/bot.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Обработчик для консоли
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Добавляем обработчики к корневому логгеру
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# ------------------------------------------------------------
logger.info("=== Бот запущен, логирование настроено ===")


def main():
    try:
        Config.validate()
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)

    # Создаём папку для логов, если её нет
    os.makedirs('logs', exist_ok=True)

    # Инициализация зависимостей
    db = Database(Config.DB_PATH)
    semantic_filter = SemanticFilter(
        model_name=Config.MODEL_NAME,
        interest_vector_path=Config.INTEREST_VECTOR_PATH,
        threshold=Config.INTEREST_THRESHOLD
    )

    # Коллекторы
    telegram = TelegramCollector(
        api_id=Config.TG_API_ID,
        api_hash=Config.TG_API_HASH,
        channels=['@htech_plus', '@nplusone', '@qwerty_live', '@confsci', '@sci_career', '@biotehno'],
        db=db,
        filter=semantic_filter
    )
    rss = RssCollector(
        feeds=[
            'https://connect.biorxiv.org/biorxiv_xml.php?subject=biochemistry+bioinformatics+genomics+genetics+molecular_biology+plant_biology',
            'https://www.nature.com/nature.rss',
            'https://www.nature.com/ng.rss',
            'https://www.nature.com/nmeth.rss',
            'https://www.nature.com/nplants.rss',
            'https://apsjournals.apsnet.org/action/showFeed?type=etoc&feed=rss&jc=mpmi',
            'https://apsjournals.apsnet.org/action/showFeed?type=etoc&feed=rss&jc=pbiomes',
            'https://apsjournals.apsnet.org/action/showFeed?type=etoc&feed=rss&jc=pdis',
            'https://apsjournals.apsnet.org/action/showFeed?type=etoc&feed=rss&jc=phyto',
            'https://journals.asm.org/action/showFeed?type=etoc&feed=rss&jc=aem',
            'https://journals.asm.org/action/showFeed?type=etoc&feed=rss&jc=spectrum'
        ],
        db=db,
        filter=semantic_filter
    )

    sender = DigestSender(
        db=db,
        filter=semantic_filter,
        token=Config.VK_GROUP_TOKEN,
        recipients=Config.VK_RECIPIENTS,
        top_n=Config.TOP_N,
        alpha=Config.FRESHNESS_ALPHA,
        beta=Config.FRESHNESS_BETA
    )

    # Выполнение
    #telegram.collect()
    rss.collect()
    sender.send_digest()

if __name__ == '__main__':
    main()
