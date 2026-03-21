from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)

class BM25Indexer:
    def __init__(self, db):
        self.db = db
        self.index = None
        self.article_ids = []
        self.refresh()

    def refresh(self):
        """Перестраивает индекс по всем неотправленным статьям."""
        articles = self.db.get_unsent_articles()
        if not articles:
            self.index = None
            self.article_ids = []
            return
        self.article_ids = [a['id'] for a in articles]
        tokenized_texts = []
        for article in articles:
            text = (article['title'] + ' ' + (article['text'] or '')).lower()
            tokenized_texts.append(text.split())
        self.index = BM25Okapi(tokenized_texts)
        logger.info(f"BM25 index refreshed with {len(articles)} articles")

    def get_scores(self, query):
        """Возвращает массив BM25-оценок для всех статей в индексе."""
        if self.index is None:
            return []
        tokenized_query = query.lower().split()
        scores = self.index.get_scores(tokenized_query)
        return scores