import logging
import math
from datetime import datetime
from typing import List, Tuple
import vk_api
from vk_api.utils import get_random_id
from db import Database
from semantic import SemanticFilter
from config import Config

logger = logging.getLogger(__name__)

class DigestSender:
    def __init__(self, db: Database, filter: SemanticFilter, token: str, recipients: List[int], top_n: int, alpha: float, beta: float):
        self.db = db
        self.filter = filter
        self.token = token
        self.recipients = recipients
        self.top_n = top_n
        self.alpha = alpha
        self.beta = beta

    def _truncate_text(self, text: str, max_len: int = 100) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + '...'

    def _parse_date(self, date_val: str) -> datetime:
        """Преобразует дату из строки в datetime."""
        try:
            return datetime.fromisoformat(date_val)
        except (ValueError, TypeError):
            return datetime.now()

    def _freshness_factor(self, pub_date: datetime) -> float:
        """Вычисляет коэффициент свежести."""
        days = (datetime.now() - pub_date).days
        if days < 0:
            days = 0
        return 1 + self.alpha * math.exp(-days / self.beta)

    def _format_message(self, top_articles: List[Tuple]) -> str:
        """Формирует текст сообщения."""
        lines = ["Ваш дайджест готов :)\n"]
        for idx, (score, sim, fresh, url, title, text, source, pub_date) in enumerate(top_articles, 1):
            description = title.strip() if title else self._truncate_text(text, 100)
            clean_url = self._clean_telegram_url(url)
            lines.append(f"{idx}. {description}\n{clean_url}\n")
        return '\n'.join(lines)

    def _compute_scores(self, articles: List[Tuple]) -> List[Tuple]:
        """Вычисляет итоговый score для каждой статьи."""
        result = []
        for url, title, text, source, date_str in articles:
            full_text = title + ' ' + text
            try:
                sim = self.filter.get_similarity(full_text)
                pub_date = self._parse_date(date_str)
                fresh = self._freshness_factor(pub_date)
                score = sim * fresh
                result.append((score, sim, fresh, url, title, text, source, pub_date))
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                result.append((0.0, 0.0, 1.0, url, title, text, source, datetime.now()))
        return result

    def _send_vk(self, message: str) -> bool:
        """Отправляет сообщение через VK API всем получателям."""
        vk_session = vk_api.VkApi(token=self.token)
        vk = vk_session.get_api()
        success = False
        for peer_id in self.recipients:
            try:
                vk.messages.send(
                    peer_id=peer_id,
                    random_id=get_random_id(),
                    message=message,
                    disable_mentions=1
                )
                logger.info(f"Digest sent successfully to peer {peer_id}")
                success = True
            except Exception as e:
                logger.error(f"Failed to send digest to peer {peer_id}: {e}")
        return success

    def send_digest(self):
        """Основной метод отправки дайджеста."""
        articles = self.db.get_unsent_articles()
        if not articles:
            logger.info("No new articles to send.")
            return

        scored = self._compute_scores(articles)
        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:self.top_n]

        if not top:
            logger.info("No articles in top after filtering.")
            return

        message = self._format_message(top)
        if len(message) > 4000:
            message = message[:4000] + '...'

        if self._send_vk(message):
            urls_to_mark = [a[3] for a in top]  # a[3] – url
            self.db.mark_as_sent(urls_to_mark)
            logger.info(f"Marked {len(urls_to_mark)} articles as sent.")
        else:
            logger.error("No messages were sent successfully. Articles remain unsent.")
