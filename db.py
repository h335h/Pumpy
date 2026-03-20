import sqlite3
import logging
from typing import List, Tuple, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connection(self):
        """Контекстный менеджер для соединения с БД."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Создаёт таблицу, если её нет."""
        with self._connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    url TEXT PRIMARY KEY,
                    title TEXT,
                    text TEXT,
                    source TEXT,
                    date TIMESTAMP,
                    sent INTEGER DEFAULT 0
                )
            ''')
            conn.commit()

    def save_article(self, url: str, title: str, text: str, source: str, date: str) -> None:
        """Сохраняет статью, игнорируя дубликаты."""
        with self._connection() as conn:
            try:
                conn.execute(
                    'INSERT INTO articles (url, title, text, source, date) VALUES (?, ?, ?, ?, ?)',
                    (url, title, text, source, date)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                logger.info(f"Duplicate article: {url}")

    def get_unsent_articles(self) -> List[Tuple[str, str, str, str, str]]:
        """Возвращает все неотправленные статьи."""
        with self._connection() as conn:
            cursor = conn.execute('SELECT url, title, text, source, date FROM articles WHERE sent=0')
            return cursor.fetchall()

    def mark_as_sent(self, urls: List[str]) -> None:
        """Помечает статьи как отправленные."""
        if not urls:
            return
        with self._connection() as conn:
            conn.executemany('UPDATE articles SET sent=1 WHERE url=?', [(url,) for url in urls])
            conn.commit()
