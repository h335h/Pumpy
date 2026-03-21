import os
import logging
import math
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from dotenv import load_dotenv
from db import Database
from semantic import SemanticFilter

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.makedirs('logs', exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-prod')
app.config['DATABASE_URL'] = os.getenv('DATABASE_URL', 'sqlite:///articles.db')

# Инициализация БД (SQLite-версия)
db = Database(app.config['DATABASE_URL'])

# Инициализация семантического модуля (загружаем модель и вектор один раз)
MODEL_NAME = os.getenv('MODEL_NAME', 'BAAI/bge-small-en-v1.5')
INTEREST_VECTOR_PATH = os.getenv('INTEREST_VECTOR_PATH', 'interest_vector.npy')
THRESHOLD = float(os.getenv('INTEREST_THRESHOLD', '0.1'))
semantic = SemanticFilter(MODEL_NAME, INTEREST_VECTOR_PATH, THRESHOLD)

# Генерация interest_vector.npy, если отсутствует
if not os.path.exists(INTEREST_VECTOR_PATH):
    if os.path.exists('positive_examples.txt'):
        logging.info("Generating interest vector from positive_examples.txt...")
        from sentence_transformers import SentenceTransformer
        import numpy as np
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


# Константы для ранжирования
TOP_N = int(os.getenv('TOP_N', 10))
FRESHNESS_ALPHA = float(os.getenv('FRESHNESS_ALPHA', '0.2'))
FRESHNESS_BETA = float(os.getenv('FRESHNESS_BETA', '30'))

# Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_id, username, email, lab_id=None):
        self.id = user_id
        self.username = username
        self.email = email
        self.lab_id = lab_id

@login_manager.user_loader
def load_user(user_id):
    user_data = db.get_user_by_id(user_id)
    if user_data:
        return User(user_data['id'], user_data['username'], user_data['email'], user_data.get('lab_id'))
    return None

# ---------- Вспомогательные функции для ранжирования ----------
def parse_date(date_val):
    """Пытается преобразовать строку даты из RSS в datetime."""
    if isinstance(date_val, datetime):
        return date_val
    if not date_val:
        return datetime.now()
    # Пробуем разные форматы
    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%a, %d %b %Y %H:%M:%S %z', '%a, %d %b %Y %H:%M:%S +0000'):
        try:
            return datetime.strptime(date_val, fmt)
        except (ValueError, TypeError):
            continue
    # Если ничего не помогло, считаем сегодня
    return datetime.now()

def freshness_factor(pub_date):
    """Коэффициент свежести: чем новее, тем выше."""
    days = (datetime.now() - pub_date).days
    if days < 0:
        days = 0
    return 1 + FRESHNESS_ALPHA * math.exp(-days / FRESHNESS_BETA)

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
            user_obj = User(user['id'], user['username'], user['email'], user.get('lab_id'))
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
    articles = db.get_unsent_articles()
    if not articles:
        return render_template('dashboard.html', articles=[])

    scored = []
    for article in articles:
        sim = article.get('similarity')
        if sim is None:
            # Для старых статей (до добавления колонки) вычисляем один раз и сохраняем
            full_text = article['title'] + ' ' + (article['text'] or '')
            sim = semantic.get_similarity(full_text)
            db.update_article_similarity(article['id'], sim)
        pub_date = parse_date(article['date'])
        fresh = freshness_factor(pub_date)
        score = sim * fresh
        scored.append((score, article))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_articles = [item[1] for item in scored[:TOP_N]]
    return render_template('dashboard.html', articles=top_articles)

@app.route('/rate/<int:article_id>', methods=['POST'])
@login_required
def rate_article(article_id):
    data = request.get_json()
    rating = data.get('rating')
    if rating is None:
        return jsonify({'error': 'No rating provided'}), 400
    db.add_rating(current_user.id, article_id, rating)
    if rating == 1:
        article = db.get_article(article_id)
        if article and article.get('text'):
            db.add_positive_example(current_user.id, article['text'])
            # TODO: пересчёт вектора интересов пользователя
    return jsonify({'status': 'ok'})

@app.route('/collect', methods=['POST'])
def collect():
    token = request.args.get('token')
    if token != os.getenv('COLLECT_TOKEN', ''):
        return 'Unauthorized', 401
    from collectors.rss_collector import RssCollector
    RSS_FEEDS = [
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
    ]
    collector = RssCollector(RSS_FEEDS, db, semantic)
    collector.collect()
    return 'OK', 200

if __name__ == '__main__':
    app.run(debug=True)