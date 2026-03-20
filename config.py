import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Хранит все настройки приложения."""
    # Telegram
    TG_API_ID = int(os.getenv('TG_API_ID', 0))
    TG_API_HASH = os.getenv('TG_API_HASH', '')

    # VK
    VK_GROUP_TOKEN = os.getenv('VK_GROUP_TOKEN', '')
    VK_RECIPIENTS = [int(x.strip()) for x in os.getenv('VK_RECIPIENTS', '').split(',') if x.strip()]

    # Фильтрация
    INTEREST_THRESHOLD = float(os.getenv('INTEREST_THRESHOLD', 0.1))

    # Отправка
    TOP_N = int(os.getenv('TOP_N', 10))
    FRESHNESS_ALPHA = float(os.getenv('FRESHNESS_ALPHA', 0.2))
    FRESHNESS_BETA = float(os.getenv('FRESHNESS_BETA', 30))

    # Пути
    DB_PATH = 'articles.db'
    INTEREST_VECTOR_PATH = 'interest_vector.npy'
    POSITIVE_EXAMPLES_PATH = 'positive_examples.txt'

    # Модель
    MODEL_NAME = 'BAAI/bge-small-en-v1.5'

    @classmethod
    def validate(cls):
        """Проверка обязательных настроек."""
        if not cls.TG_API_ID or not cls.TG_API_HASH:
            raise ValueError("TG_API_ID и TG_API_HASH должны быть заданы в .env")
        if not cls.VK_GROUP_TOKEN:
            raise ValueError("VK_GROUP_TOKEN должен быть задан в .env")
        if not cls.VK_RECIPIENTS:
            raise ValueError("VK_RECIPIENTS должен быть задан в .env")
