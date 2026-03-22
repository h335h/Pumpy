import numpy as np
import faiss
from db import Database
from dotenv import load_dotenv
import os

load_dotenv()
db = Database(os.getenv('DATABASE_URL'))

# Получаем все статьи (можно использовать get_unsent_articles, если нужно только неотправленные,
# но для индекса лучше взять все статьи, чтобы не перестраивать при каждом сборе)
articles = db.get_unsent_articles()
embeddings = []
ids = []
for art in articles:
    emb = art.get('embedding')
    if emb is not None:
        embeddings.append(np.frombuffer(emb, dtype=np.float32))
        ids.append(art['id'])

if not embeddings:
    print("No embeddings found")
    exit()

embeddings = np.vstack(embeddings).astype('float32')
faiss.normalize_L2(embeddings)  # нормализуем для косинусного сходства

index = faiss.IndexFlatIP(384)  # размерность эмбеддингов (384)
index.add(embeddings)
faiss.write_index(index, 'articles.faiss')

with open('article_ids.txt', 'w') as f:
    for i in ids:
        f.write(f"{i}\n")

print(f"FAISS index built with {len(ids)} articles")
