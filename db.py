import os
import re
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor

class Database:
    def __init__(self, database_url):
        self.database_url = database_url
        self.is_sqlite = database_url.startswith('sqlite:///')
        self.init_db()

    def _get_sqlite_conn(self):
        db_path = self.database_url.replace('sqlite:///', '')
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_postgres_conn(self):
        return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)

    def get_connection(self):
        if self.is_sqlite:
            return self._get_sqlite_conn()
        else:
            return self._get_postgres_conn()

    def _column_exists(self, conn, table, column):
        if self.is_sqlite:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [info[1] for info in cursor.fetchall()]
            return column in columns
        else:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = %s AND column_name = %s
            """, (table, column))
            return cursor.fetchone() is not None

    def init_db(self):
        with self.get_connection() as conn:
            if self.is_sqlite:
                self._init_sqlite(conn)
            else:
                self._init_postgres(conn)

    def _init_sqlite(self, conn):
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
                embedding BLOB,
                similarity REAL
            )
        ''')
        if not self._column_exists(conn, 'articles', 'similarity'):
            cursor.execute('ALTER TABLE articles ADD COLUMN similarity REAL')
        if not self._column_exists(conn, 'articles', 'embedding'):
            cursor.execute('ALTER TABLE articles ADD COLUMN embedding BLOB')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT,
                password TEXT,
                lab_id INTEGER,
                vector_path TEXT,
                is_admin BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        if not self._column_exists(conn, 'users', 'is_admin'):
            cursor.execute('ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0')

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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_vectors (
                user_id INTEGER PRIMARY KEY,
                vector BLOB NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rss_feeds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                name TEXT,
                active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # ---------- Новая таблица для обратной связи ----------
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        conn.commit()

    def _init_postgres(self, conn):
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id SERIAL PRIMARY KEY,
                url TEXT UNIQUE,
                title TEXT,
                text TEXT,
                source TEXT,
                date TIMESTAMP,
                sent BOOLEAN DEFAULT FALSE,
                embedding BYTEA,
                similarity REAL
            )
        ''')
        if not self._column_exists(conn, 'articles', 'similarity'):
            cursor.execute('ALTER TABLE articles ADD COLUMN similarity REAL')
        if not self._column_exists(conn, 'articles', 'embedding'):
            cursor.execute('ALTER TABLE articles ADD COLUMN embedding BYTEA')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT,
                password TEXT,
                lab_id INTEGER,
                vector_path TEXT,
                is_admin BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW()
            )
        ''')
        if not self._column_exists(conn, 'users', 'is_admin'):
            cursor.execute('ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ratings (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
                rating INTEGER,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(user_id, article_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positive_examples (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                text TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS labs (
                id SERIAL PRIMARY KEY,
                name TEXT,
                invite_code TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_vectors (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                vector BYTEA NOT NULL,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rss_feeds (
                id SERIAL PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                name TEXT,
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW()
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                text TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        ''')
        conn.commit()

    # --- Методы работы со статьями (без изменений) ---
    def save_article(self, url, title, text, source, date, similarity, embedding=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('''
                    INSERT OR IGNORE INTO articles (url, title, text, source, date, similarity, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (url, title, text, source, date, similarity, embedding))
            else:
                cursor.execute('''
                    INSERT INTO articles (url, title, text, source, date, similarity, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING
                ''', (url, title, text, source, date, similarity, embedding))
            conn.commit()

    def update_article_similarity(self, article_id, similarity):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('UPDATE articles SET similarity = ? WHERE id = ?', (similarity, article_id))
            else:
                cursor.execute('UPDATE articles SET similarity = %s WHERE id = %s', (similarity, article_id))
            conn.commit()

    def update_article_embedding(self, article_id, embedding_bytes):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('UPDATE articles SET embedding = ? WHERE id = ?', (embedding_bytes, article_id))
            else:
                cursor.execute('UPDATE articles SET embedding = %s WHERE id = %s', (embedding_bytes, article_id))
            conn.commit()

    def get_unsent_articles(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('SELECT id, url, title, text, source, date, similarity, embedding FROM articles WHERE sent = 0')
            else:
                cursor.execute('SELECT id, url, title, text, source, date, similarity, embedding FROM articles WHERE sent = FALSE')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def mark_as_sent(self, article_ids):
        if not article_ids:
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                placeholders = ','.join(['?'] * len(article_ids))
                cursor.execute(f'UPDATE articles SET sent = 1 WHERE id IN ({placeholders})', article_ids)
            else:
                placeholders = ','.join(['%s'] * len(article_ids))
                cursor.execute(f'UPDATE articles SET sent = TRUE WHERE id IN ({placeholders})', article_ids)
            conn.commit()

    def get_article(self, article_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('SELECT * FROM articles WHERE id = ?', (article_id,))
            else:
                cursor.execute('SELECT * FROM articles WHERE id = %s', (article_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # --- Методы работы с пользователями ---
    def create_user(self, username, email, password):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                if self.is_sqlite:
                    cursor.execute('''
                        INSERT INTO users (username, email, password)
                        VALUES (?, ?, ?)
                    ''', (username, email, password))
                    conn.commit()
                    return cursor.lastrowid
                else:
                    cursor.execute('''
                        INSERT INTO users (username, email, password)
                        VALUES (%s, %s, %s) RETURNING id
                    ''', (username, email, password))
                    user_id = cursor.fetchone()[0]
                    conn.commit()
                    return user_id
            except (sqlite3.IntegrityError, psycopg2.IntegrityError):
                return None

    def get_user_by_username(self, username):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            else:
                cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            else:
                cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def add_rating(self, user_id, article_id, rating):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('''
                    INSERT INTO ratings (user_id, article_id, rating)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id, article_id) DO UPDATE SET rating = excluded.rating, created_at = CURRENT_TIMESTAMP
                ''', (user_id, article_id, rating))
            else:
                cursor.execute('''
                    INSERT INTO ratings (user_id, article_id, rating)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, article_id) DO UPDATE SET rating = EXCLUDED.rating, created_at = NOW()
                ''', (user_id, article_id, rating))
            conn.commit()

    def add_positive_example(self, user_id, text):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('''
                    INSERT INTO positive_examples (user_id, text)
                    VALUES (?, ?)
                ''', (user_id, text))
            else:
                cursor.execute('''
                    INSERT INTO positive_examples (user_id, text)
                    VALUES (%s, %s)
                ''', (user_id, text))
            conn.commit()

    # --- Методы для user_vectors ---
    def save_user_vector(self, user_id, vector_bytes):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('''
                    INSERT INTO user_vectors (user_id, vector, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET vector = excluded.vector, updated_at = CURRENT_TIMESTAMP
                ''', (user_id, vector_bytes))
            else:
                cursor.execute('''
                    INSERT INTO user_vectors (user_id, vector, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (user_id) DO UPDATE SET vector = EXCLUDED.vector, updated_at = NOW()
                ''', (user_id, vector_bytes))
            conn.commit()

    def get_user_vector(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('SELECT vector FROM user_vectors WHERE user_id = ?', (user_id,))
            else:
                cursor.execute('SELECT vector FROM user_vectors WHERE user_id = %s', (user_id,))
            row = cursor.fetchone()
            if row:
                return row[0]
            return None

    def delete_rating(self, user_id, article_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('DELETE FROM ratings WHERE user_id = ? AND article_id = ?', (user_id, article_id))
            else:
                cursor.execute('DELETE FROM ratings WHERE user_id = %s AND article_id = %s', (user_id, article_id))
            conn.commit()

    # --- Методы для RSS-источников ---
    def get_all_feeds(self, active_only=True):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                if active_only:
                    cursor.execute('SELECT * FROM rss_feeds WHERE active = 1 ORDER BY id')
                else:
                    cursor.execute('SELECT * FROM rss_feeds ORDER BY id')
            else:
                if active_only:
                    cursor.execute('SELECT * FROM rss_feeds WHERE active = TRUE ORDER BY id')
                else:
                    cursor.execute('SELECT * FROM rss_feeds ORDER BY id')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def add_feed(self, url, name=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('INSERT INTO rss_feeds (url, name) VALUES (?, ?)', (url, name))
            else:
                cursor.execute('INSERT INTO rss_feeds (url, name) VALUES (%s, %s)', (url, name))
            conn.commit()

    def delete_feed(self, feed_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('DELETE FROM rss_feeds WHERE id = ?', (feed_id,))
            else:
                cursor.execute('DELETE FROM rss_feeds WHERE id = %s', (feed_id,))
            conn.commit()

    def toggle_feed_active(self, feed_id, active):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('UPDATE rss_feeds SET active = ? WHERE id = ?', (active, feed_id))
            else:
                cursor.execute('UPDATE rss_feeds SET active = %s WHERE id = %s', (active, feed_id))
            conn.commit()

    # --- Методы для настроек ---
    def get_setting(self, key, default=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
            else:
                cursor.execute('SELECT value FROM settings WHERE key = %s', (key,))
            row = cursor.fetchone()
            if row:
                return row[0]
            return default

    def set_setting(self, key, value):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('''
                    INSERT OR REPLACE INTO settings (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (key, value))
            else:
                cursor.execute('''
                    INSERT INTO settings (key, value, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                ''', (key, value))
            conn.commit()

    # --- Методы для обратной связи ---
    def add_feedback(self, user_id, text):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('INSERT INTO feedback (user_id, text) VALUES (?, ?)', (user_id, text))
            else:
                cursor.execute('INSERT INTO feedback (user_id, text) VALUES (%s, %s)', (user_id, text))
            conn.commit()

    def get_all_feedback(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_sqlite:
                cursor.execute('''
                    SELECT f.id, f.user_id, u.username, f.text, f.created_at
                    FROM feedback f
                    LEFT JOIN users u ON f.user_id = u.id
                    ORDER BY f.created_at DESC
                ''')
            else:
                cursor.execute('''
                    SELECT f.id, f.user_id, u.username, f.text, f.created_at
                    FROM feedback f
                    LEFT JOIN users u ON f.user_id = u.id
                    ORDER BY f.created_at DESC
                ''')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]