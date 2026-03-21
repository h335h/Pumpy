import os
import logging
import math
import numpy as np
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, make_response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
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

def freshness_factor(pub_date, alpha, beta):
    days = (datetime.now() - pub_date).days
    if days < 0:
        days = 0
    return 1 + alpha * math.exp(-days / beta)

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

# ---------- Функция для получения ранжированных статей ----------
def get_ranked_articles(user_id, query=None):
    # Загружаем параметры из БД
    bm25_weight = float(db.get_setting('bm25_weight', '0.3'))
    mmr_lambda = float(db.get_setting('mmr_lambda', '0.5'))
    freshness_alpha = float(db.get_setting('freshness_alpha', '0.2'))
    freshness_beta = float(db.get_setting('freshness_beta', '30'))
    top_n = int(os.getenv('TOP_N', 10))

    user_vec_bytes = db.get_user_vector(user_id)
    if user_vec_bytes is None:
        interest_vector = np.load(INTEREST_VECTOR_PATH)
    else:
        interest_vector = np.frombuffer(user_vec_bytes, dtype=np.float32)

    articles = db.get_unsent_articles()
    article_ids = [a['id'] for a in articles]

    # Если передан поисковый запрос, используем его вместо лайков
    if query:
        bm25_query = query
    else:
        # Строим запрос из лайкнутых статей
        with db.get_connection() as conn:
            cursor = conn.cursor()
            if db.is_sqlite:
                cursor.execute('''
                    SELECT a.title, a.text FROM articles a
                    JOIN ratings r ON a.id = r.article_id
                    WHERE r.user_id = ? AND r.rating = 1
                ''', (user_id,))
            else:
                cursor.execute('''
                    SELECT a.title, a.text FROM articles a
                    JOIN ratings r ON a.id = r.article_id
                    WHERE r.user_id = %s AND r.rating = 1
                ''', (user_id,))
            liked_texts = cursor.fetchall()
            if liked_texts:
                bm25_query = " ".join([f"{row[0]} {row[1] or ''}" for row in liked_texts])
            else:
                bm25_query = None

    bm25_scores = {}
    if bm25_query:
        bm25_scores = bm25_indexer.get_scores_dict(bm25_query)

    # Получаем рейтинги пользователя для подсветки кнопок
    user_ratings = {}
    if article_ids:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            if db.is_sqlite:
                placeholders = ','.join(['?'] * len(article_ids))
                cursor.execute(f'''
                    SELECT article_id, rating FROM ratings
                    WHERE user_id = ? AND article_id IN ({placeholders})
                ''', (user_id, *article_ids))
            else:
                placeholders = ','.join(['%s'] * len(article_ids))
                cursor.execute(f'''
                    SELECT article_id, rating FROM ratings
                    WHERE user_id = %s AND article_id IN ({placeholders})
                ''', (user_id, *article_ids))
            for row in cursor.fetchall():
                user_ratings[row[0]] = row[1]

    # Ранжирование
    scored = []
    for article in articles:
        emb_bytes = article.get('embedding')
        if emb_bytes is None:
            full_text = article['title'] + ' ' + (article['text'] or '')
            emb = semantic.get_embedding(full_text)
            article_embedding = emb
            db.update_article_embedding(article['id'], emb.tobytes())
        else:
            article_embedding = np.frombuffer(emb_bytes, dtype=np.float32)

        sim = np.dot(interest_vector, article_embedding) / (np.linalg.norm(interest_vector) * np.linalg.norm(article_embedding))

        bm25 = bm25_scores.get(article['id'], 0.0)

        # Нормализуем BM25 по максимуму
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        if max_bm25 > 0:
            bm25_norm = bm25 / max_bm25
        else:
            bm25_norm = 0.0

        hybrid_score = bm25_weight * bm25_norm + (1 - bm25_weight) * sim
        pub_date = parse_date(article['date'])
        fresh = freshness_factor(pub_date, freshness_alpha, freshness_beta)
        final_score = hybrid_score * fresh

        scored.append({
            'id': article['id'],
            'score': final_score,
            'embedding': article_embedding,
            'article': article
        })

    selected = mmr_selection(scored, lambda_val=mmr_lambda, top_n=top_n)
    top_articles = [item['article'] for item in selected]

    for article in top_articles:
        article['user_rating'] = user_ratings.get(article['id'], None)

    return top_articles

# ---------- Маршруты ----------
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if db.get_user_by_username(username):
            flash('Username already exists')
            return redirect(url_for('register'))
        hashed = generate_password_hash(password)
        user_id = db.create_user(username, email, hashed)
        if user_id:
            # Сделать первого пользователя администратором
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
        if user and check_password_hash(user['password'], password):
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
    top_articles = get_ranked_articles(current_user.id)
    return render_template('dashboard.html', articles=top_articles)

@app.route('/api/search', methods=['POST'])
@login_required
def search():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Empty query'}), 400
    articles = get_ranked_articles(current_user.id, query=query)
    html = render_template('_article_list.html', articles=articles)
    return jsonify({'html': html})

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
    feeds = db.get_all_feeds(active_only=True)
    RSS_FEEDS = [feed['url'] for feed in feeds]
    collector = RssCollector(RSS_FEEDS, db, semantic)
    collector.collect()
    bm25_indexer.refresh()
    return 'OK', 200

# ---------- API для интерфейса (упрощённые) ----------
@app.route('/api/user_context')
@login_required
def user_context():
    with db.get_connection() as conn:
        cursor = conn.cursor()
        if db.is_sqlite:
            cursor.execute('SELECT COUNT(*) FROM ratings WHERE user_id = ? AND rating = 1', (current_user.id,))
        else:
            cursor.execute('SELECT COUNT(*) FROM ratings WHERE user_id = %s AND rating = 1', (current_user.id,))
        likes_count = cursor.fetchone()[0]
    if likes_count == 0:
        return jsonify({
            'show_tip': True,
            'message': '👋 Welcome! Like articles you find interesting — we will learn your preferences and show more relevant papers.'
        })
    else:
        return jsonify({'show_tip': False})

# Удалён маршрут /api/command (или можно оставить заглушку)
# Оставляем его, но возвращаем сообщение "Coming soon" без ложных обещаний
@app.route('/api/command', methods=['POST'])
@login_required
def process_command():
    return jsonify({'message': 'Natural language commands are coming soon! Use the search bar above to refine your digest.'})

@app.route('/export/bibtex')
@login_required
def export_bibtex():
    articles = db.get_unsent_articles()
    bibtex_entries = []
    for art in articles:
        bibtex = f"""@article{{{art['id']},
  title = {{{art['title']}}},
  journal = {{{art['source']}}},
  year = {{{art['date'][:4]}}},
  url = {{{art['url']}}}
}}"""
        bibtex_entries.append(bibtex)
    response = make_response('\n\n'.join(bibtex_entries))
    response.headers['Content-Type'] = 'application/x-bibtex'
    response.headers['Content-Disposition'] = 'attachment; filename=digest.bib'
    return response

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

@app.route('/admin/settings', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_settings():
    if request.method == 'POST':
        db.set_setting('bm25_weight', request.form.get('bm25_weight', '0.3'))
        db.set_setting('mmr_lambda', request.form.get('mmr_lambda', '0.5'))
        db.set_setting('freshness_alpha', request.form.get('freshness_alpha', '0.2'))
        db.set_setting('freshness_beta', request.form.get('freshness_beta', '30'))
        db.set_setting('interest_threshold', request.form.get('interest_threshold', '0.6'))
        flash('Settings saved', 'success')
        return redirect(url_for('admin_settings'))
    settings = {
        'bm25_weight': float(db.get_setting('bm25_weight', '0.3')),
        'mmr_lambda': float(db.get_setting('mmr_lambda', '0.5')),
        'freshness_alpha': float(db.get_setting('freshness_alpha', '0.2')),
        'freshness_beta': float(db.get_setting('freshness_beta', '30')),
        'interest_threshold': float(db.get_setting('interest_threshold', '0.6')),
    }
    return render_template('admin/settings.html', settings=settings)

if __name__ == '__main__':
    app.run(debug=True)