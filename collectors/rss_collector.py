import logging
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List
from .base import Collector
from db import Database
from semantic import SemanticFilter

logger = logging.getLogger(__name__)

class RssCollector(Collector):
    def __init__(self, feeds: List[str], db: Database, filter: SemanticFilter):
        self.feeds = feeds
        self.db = db
        self.filter = filter

    def _fetch_full_text(self, url: str) -> str:
        """Загружает полный текст статьи (первые параграфы)."""
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'lxml')
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text() for p in paragraphs)
            return text[:2000]
        except Exception as e:
            logger.error(f"Error fetching full text from {url}: {e}")
            return ''

    def _process_feed(self, feed_url: str):
        try:
            feed = feedparser.parse(feed_url)
            logger.info(f"RSS feed {feed_url} has {len(feed.entries)} entries")
            for entry in feed.entries:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                link = entry.get('link', '')
                published = entry.get('published', datetime.now().isoformat())
                # Если summary короткий, пробуем загрузить полный текст
                if len(summary) < 200:
                    full = self._fetch_full_text(link)
                    text = title + ' ' + full
                else:
                    text = title + ' ' + summary
                if self.filter.is_relevant(text):
                    self.db.save_article(link, title, text[:1000], feed.feed.get('title', 'RSS'), published)
                    logger.info(f"Saved RSS: {link}")
        except Exception as e:
            logger.error(f"Error in RSS feed {feed_url}: {e}")

    def collect(self):
        for feed_url in self.feeds:
            self._process_feed(feed_url)
