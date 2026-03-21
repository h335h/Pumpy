import sqlite3
import os

class Database:
    def __init__(self, database_url):
        if database_url.startswith('sqlite:///'):
            self.db_path = database_url.replace('sqlite:///', '')
        else:
            self.db_path = database_url
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _column_exists(self, cursor, table, column):
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [info[1] for info in cursor.fetchall()]
        return column in columns

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    title TEXT,
                    text TEXT,
                    source TEXT,
                    date TIMESTAMP,
                    sent BOOLEAN DEFAULT 0,
                    embedding TEXT,
                    similarity REAL
                )
            ''')
            if not self._column_exists(cursor, 'articles', 'similarity'):
                cursor.execute('ALTER TABLE articles ADD COLUMN similarity REAL')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    email TEXT,
                    password TEXT,
                    lab_id INTEGER,
                    vector_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    article_id INTEGER,
                    rating INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, article_id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positive_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS labs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    invite_code TEXT
                )
            ''')
            conn.commit()

    def save_article(self, url, title, text, source, date, similarity):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO articles (url, title, text, source, date, similarity)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (url, title, text, source, date, similarity))
                conn.commit()
            except sqlite3.IntegrityError:
                pass

    def update_article_similarity(self, article_id, similarity):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE articles SET similarity = ? WHERE id = ?', (similarity, article_id))
            conn.commit()

    def get_unsent_articles(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM articles WHERE sent = 0')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def mark_as_sent(self, article_ids):
        if not article_ids:
            return
        placeholders = ','.join(['?'] * len(article_ids))
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'UPDATE articles SET sent = 1 WHERE id IN ({placeholders})', article_ids)
            conn.commit()

    def get_article(self, article_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM articles WHERE id = ?', (article_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_user(self, username, email, password):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO users (username, email, password)
                    VALUES (?, ?, ?)
                ''', (username, email, password))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def get_user_by_username(self, username):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def add_rating(self, user_id, article_id, rating):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ratings (user_id, article_id, rating)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, article_id) DO UPDATE SET rating = excluded.rating, created_at = CURRENT_TIMESTAMP
            ''', (user_id, article_id, rating))
            conn.commit()

    def add_positive_example(self, user_id, text):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO positive_examples (user_id, text)
                VALUES (?, ?)
            ''', (user_id, text))
            conn.commit()

    def get_top_articles_for_user(self, user_id, limit=10):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM articles WHERE sent = 0
                ORDER BY date DESC LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]