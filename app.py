import os
import logging
import math
import numpy as np
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from dotenv import load_dotenv
from db import Database
from semantic import SemanticFilter
from bm25_indexer import BM25Indexer

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-prod')
app.config['DATABASE_URL'] = os.getenv('DATABASE_URL', 'sqlite:///articles.db')

db = Database(app.config['DATABASE_URL'])

MODEL_NAME = os.getenv('MODEL_NAME', 'BAAI/bge-small-en-v1.5')
INTEREST_VECTOR_PATH = os.getenv('INTEREST_VECTOR_PATH', 'interest_vector.npy')
THRESHOLD = float(os.getenv('INTEREST_THRESHOLD', '0.1'))

# Генерация глобального вектора, если отсутствует
if not os.path.exists(INTEREST_VECTOR_PATH):
    if os.path.exists('positive_examples.txt'):
        logging.info("Generating interest vector from positive_examples.txt...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        with open('positive_examples.txt', 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        if texts:
            embeddings = model.encode(texts, convert_to_tensor=True)
            interest_vector = embeddings.mean(axis=0).cpu().numpy()
            np.save(INTEREST_VECTOR_PATH, interest_vector)
            logging.info(f"Interest vector saved to {INTEREST_VECTOR_PATH}")
        else:
            raise FileNotFoundError("positive_examples.txt is empty or missing")
    else:
        raise FileNotFoundError("positive_examples.txt not found, cannot generate interest vector")

semantic = SemanticFilter(MODEL_NAME, INTEREST_VECTOR_PATH, THRESHOLD)

# Инициализация BM25 индексатора
bm25_indexer = BM25Indexer(db)

TOP_N = int(os.getenv('TOP_N', 10))
FRESHNESS_ALPHA = float(os.getenv('FRESHNESS_ALPHA', '0.2'))
FRESHNESS_BETA = float(os.getenv('FRESHNESS_BETA', '30'))

# MMR параметр: λ — баланс между релевантностью и разнообразием (0.5 = средний)
MMR_LAMBDA = 0.5

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_id, username, email, is_admin=False, lab_id=None):
        self.id = user_id
        self.username = username
        self.email = email
        self.is_admin = is_admin
        self.lab_id = lab_id

@login_manager.user_loader
def load_user(user_id):
    user_data = db.get_user_by_id(user_id)
    if user_data:
        return User(user_data['id'], user_data['username'], user_data['email'],
                    user_data.get('is_admin', False), user_data.get('lab_id'))
    return None

# ---------- Вспомогательные функции ----------
def parse_date(date_val):
    if isinstance(date_val, datetime):
        return date_val
    if not date_val:
        return datetime.now()
    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%a, %d %b %Y %H:%M:%S %z', '%a, %d %b %Y %H:%M:%S +0000'):
        try:
            return datetime.strptime(date_val, fmt)
        except (ValueError, TypeError):
            continue
    return datetime.now()

def freshness_factor(pub_date):
    days = (datetime.now() - pub_date).days
    if days < 0:
        days = 0
    return 1 + FRESHNESS_ALPHA * math.exp(-days / FRESHNESS_BETA)

# ---------- MMR диверсификация ----------
def mmr_selection(candidates, lambda_val=0.5, top_n=10):
    if not candidates:
        return []
    selected = []
    remaining = list(candidates)

    first = max(remaining, key=lambda x: x['score'])
    selected.append(first)
    remaining.remove(first)

    for _ in range(min(top_n - 1, len(remaining))):
        best = None
        best_mmr = -1
        for cand in remaining:
            relevance = cand['score']
            max_sim = max(np.dot(cand['embedding'], sel['embedding']) / (np.linalg.norm(cand['embedding']) * np.linalg.norm(sel['embedding']))
                         for sel in selected)
            mmr = lambda_val * relevance - (1 - lambda_val) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best = cand
        selected.append(best)
        remaining.remove(best)

    return selected

# ---------- Персонализация ----------
def compute_user_vector(user_id):
    with db.get_connection() as conn:
        cursor = conn.cursor()
        if db.is_sqlite:
            cursor.execute('''
                SELECT a.embedding, r.rating
                FROM articles a
                JOIN ratings r ON a.id = r.article_id
                WHERE r.user_id = ?
            ''', (user_id,))
        else:
            cursor.execute('''
                SELECT a.embedding, r.rating
                FROM articles a
                JOIN ratings r ON a.id = r.article_id
                WHERE r.user_id = %s
            ''', (user_id,))
        rows = cursor.fetchall()

    if not rows:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            if db.is_sqlite:
                cursor.execute('DELETE FROM user_vectors WHERE user_id = ?', (user_id,))
            else:
                cursor.execute('DELETE FROM user_vectors WHERE user_id = %s', (user_id,))
            conn.commit()
        return None

    weighted_vectors = []
    for row in rows:
        emb = row[0]
        if emb is None:
            continue
        vec = np.frombuffer(emb, dtype=np.float32)
        weight = 1 if row[1] == 1 else -1
        weighted_vectors.append(vec * weight)

    if not weighted_vectors:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            if db.is_sqlite:
                cursor.execute('DELETE FROM user_vectors WHERE user_id = ?', (user_id,))
            else:
                cursor.execute('DELETE FROM user_vectors WHERE user_id = %s', (user_id,))
            conn.commit()
        return None

    user_vec = np.mean(weighted_vectors, axis=0)
    db.save_user_vector(user_id, user_vec.tobytes())
    return user_vec

# ---------- Декоратор для админ-доступа ----------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# ---------- Маршруты ----------
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if db.get_user_by_username(username):
            flash('Username already exists')
            return redirect(url_for('register'))
        user_id = db.create_user(username, email, password)
        if user_id:
            # Сделать первого пользователя администратором (если нет других)
            with db.get_connection() as conn:
                cursor = conn.cursor()
                if db.is_sqlite:
                    cursor.execute('SELECT COUNT(*) FROM users')
                    count = cursor.fetchone()[0]
                    if count == 1:
                        cursor.execute('UPDATE users SET is_admin = 1 WHERE id = ?', (user_id,))
                else:
                    cursor.execute('SELECT COUNT(*) FROM users')
                    count = cursor.fetchone()[0]
                    if count == 1:
                        cursor.execute('UPDATE users SET is_admin = TRUE WHERE id = %s', (user_id,))
                conn.commit()
            flash('Registration successful, please login')
            return redirect(url_for('login'))
        else:
            flash('Registration failed')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = db.get_user_by_username(username)
        if user and user['password'] == password:
            user_obj = User(user['id'], user['username'], user['email'],
                            user.get('is_admin', False), user.get('lab_id'))
            login_user(user_obj)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_vec_bytes = db.get_user_vector(current_user.id)
    if user_vec_bytes is None:
        interest_vector = np.load(INTEREST_VECTOR_PATH)
    else:
        interest_vector = np.frombuffer(user_vec_bytes, dtype=np.float32)

    articles = db.get_unsent_articles()
    article_ids = [a['id'] for a in articles]

    # --- BM25: строим запрос из интересов пользователя (лайкнутые статьи) ---
    query = ""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        if db.is_sqlite:
            cursor.execute('''
                SELECT a.title, a.text FROM articles a
                JOIN ratings r ON a.id = r.article_id
                WHERE r.user_id = ? AND r.rating = 1
            ''', (current_user.id,))
        else:
            cursor.execute('''
                SELECT a.title, a.text FROM articles a
                JOIN ratings r ON a.id = r.article_id
                WHERE r.user_id = %s AND r.rating = 1
            ''', (current_user.id,))
        liked_texts = cursor.fetchall()
        if liked_texts:
            query = " ".join([f"{row[0]} {row[1] or ''}" for row in liked_texts])

    bm25_scores = None
    if query:
        bm25_scores = bm25_indexer.get_scores(query)

    # Получаем текущие рейтинги пользователя для подсветки кнопок
    user_ratings = {}
    if article_ids:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            if db.is_sqlite:
                placeholders = ','.join(['?'] * len(article_ids))
                cursor.execute(f'''
                    SELECT article_id, rating FROM ratings
                    WHERE user_id = ? AND article_id IN ({placeholders})
                ''', (current_user.id, *article_ids))
            else:
                placeholders = ','.join(['%s'] * len(article_ids))
                cursor.execute(f'''
                    SELECT article_id, rating FROM ratings
                    WHERE user_id = %s AND article_id IN ({placeholders})
                ''', (current_user.id, *article_ids))
            for row in cursor.fetchall():
                user_ratings[row[0]] = row[1]

    # Ранжирование
    scored = []
    for idx, article in enumerate(articles):
        emb_bytes = article.get('embedding')
        if emb_bytes is None:
            full_text = article['title'] + ' ' + (article['text'] or '')
            emb = semantic.get_embedding(full_text)
            article_embedding = emb
            db.update_article_embedding(article['id'], emb.tobytes())
        else:
            article_embedding = np.frombuffer(emb_bytes, dtype=np.float32)

        sim = np.dot(interest_vector, article_embedding) / (np.linalg.norm(interest_vector) * np.linalg.norm(article_embedding))

        bm25 = 0.0
        if bm25_scores is not None and idx < len(bm25_scores):
            bm25 = bm25_scores[idx]

        if bm25_scores is not None and len(bm25_scores) > 0:
            max_bm25 = np.max(bm25_scores) if bm25_scores.size > 0 else 1.0
            if max_bm25 > 0:
                bm25_norm = bm25 / max_bm25
            else:
                bm25_norm = 0.0
        else:
            bm25_norm = 0.0

        hybrid_score = 0.3 * bm25_norm + 0.7 * sim
        pub_date = parse_date(article['date'])
        fresh = freshness_factor(pub_date)
        final_score = hybrid_score * fresh

        scored.append({
            'id': article['id'],
            'score': final_score,
            'embedding': article_embedding,
            'article': article
        })

    selected = mmr_selection(scored, lambda_val=MMR_LAMBDA, top_n=TOP_N)
    top_articles = [item['article'] for item in selected]

    for article in top_articles:
        article['user_rating'] = user_ratings.get(article['id'], None)

    return render_template('dashboard.html', articles=top_articles)

@app.route('/article/<int:article_id>')
@login_required
def get_article(article_id):
    article = db.get_article(article_id)
    if not article:
        return jsonify({'error': 'Article not found'}), 404
    return jsonify({
        'id': article['id'],
        'title': article['title'],
        'text': article['text'],
        'source': article['source'],
        'date': article['date'],
        'url': article['url']
    })

@app.route('/rate/<int:article_id>', methods=['POST'])
@login_required
def rate_article(article_id):
    data = request.get_json()
    rating = data.get('rating')
    if rating is None:
        db.delete_rating(current_user.id, article_id)
    else:
        db.add_rating(current_user.id, article_id, rating)
    compute_user_vector(current_user.id)
    return jsonify({'status': 'ok'})

@app.route('/collect', methods=['POST'])
def collect():
    token = request.args.get('token')
    if token != os.getenv('COLLECT_TOKEN', ''):
        return 'Unauthorized', 401
    from collectors.rss_collector import RssCollector
    # Загружаем активные фиды из БД
    feeds = db.get_all_feeds(active_only=True)
    RSS_FEEDS = [feed['url'] for feed in feeds]
    collector = RssCollector(RSS_FEEDS, db, semantic)
    collector.collect()
    bm25_indexer.refresh()
    return 'OK', 200

# ---------- Административные маршруты ----------
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    return render_template('admin/index.html')

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    with db.get_connection() as conn:
        cursor = conn.cursor()
        if db.is_sqlite:
            cursor.execute('''
                SELECT u.id, u.username, u.email, u.is_admin,
                       COUNT(r.id) as rating_count,
                       SUM(CASE WHEN r.rating=1 THEN 1 ELSE 0 END) as likes,
                       SUM(CASE WHEN r.rating=0 THEN 1 ELSE 0 END) as dislikes
                FROM users u
                LEFT JOIN ratings r ON u.id = r.user_id
                GROUP BY u.id
                ORDER BY u.id
            ''')
        else:
            cursor.execute('''
                SELECT u.id, u.username, u.email, u.is_admin,
                       COUNT(r.id) as rating_count,
                       SUM(CASE WHEN r.rating=1 THEN 1 ELSE 0 END) as likes,
                       SUM(CASE WHEN r.rating=0 THEN 1 ELSE 0 END) as dislikes
                FROM users u
                LEFT JOIN ratings r ON u.id = r.user_id
                GROUP BY u.id
                ORDER BY u.id
            ''')
        users = cursor.fetchall()
    return render_template('admin/users.html', users=users)

@app.route('/admin/user/<int:user_id>')
@login_required
@admin_required
def admin_user_details(user_id):
    user_data = db.get_user_by_id(user_id)
    if not user_data:
        flash('User not found', 'danger')
        return redirect(url_for('admin_users'))

    with db.get_connection() as conn:
        cursor = conn.cursor()
        if db.is_sqlite:
            cursor.execute('''
                SELECT a.id, a.title, a.url, r.rating, r.created_at
                FROM ratings r
                JOIN articles a ON r.article_id = a.id
                WHERE r.user_id = ?
                ORDER BY r.created_at DESC
            ''', (user_id,))
        else:
            cursor.execute('''
                SELECT a.id, a.title, a.url, r.rating, r.created_at
                FROM ratings r
                JOIN articles a ON r.article_id = a.id
                WHERE r.user_id = %s
                ORDER BY r.created_at DESC
            ''', (user_id,))
        ratings = cursor.fetchall()
    return render_template('admin/user.html', user=user_data, ratings=ratings)

@app.route('/admin/feeds')
@login_required
@admin_required
def admin_feeds():
    feeds = db.get_all_feeds(active_only=False)
    return render_template('admin/feeds.html', feeds=feeds)

@app.route('/admin/feeds/add', methods=['POST'])
@login_required
@admin_required
def add_feed():
    url = request.form.get('url')
    name = request.form.get('name')
    if url:
        db.add_feed(url, name)
        flash('Feed added successfully', 'success')
    else:
        flash('URL is required', 'danger')
    return redirect(url_for('admin_feeds'))

@app.route('/admin/feeds/delete/<int:feed_id>')
@login_required
@admin_required
def delete_feed(feed_id):
    db.delete_feed(feed_id)
    flash('Feed deleted', 'success')
    return redirect(url_for('admin_feeds'))

@app.route('/admin/feeds/toggle/<int:feed_id>')
@login_required
@admin_required
def toggle_feed(feed_id):
    feeds = db.get_all_feeds(active_only=False)
    feed = next((f for f in feeds if f['id'] == feed_id), None)
    if feed:
        new_active = 0 if feed['active'] else 1
        db.toggle_feed_active(feed_id, new_active)
        flash('Feed status updated', 'success')
    else:
        flash('Feed not found', 'danger')
    return redirect(url_for('admin_feeds'))

if __name__ == '__main__':
    app.run(debug=True)