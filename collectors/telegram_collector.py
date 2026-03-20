import asyncio
import logging
from telethon import TelegramClient
from datetime import datetime
from typing import List
from .base import Collector
from db import Database
from semantic import SemanticFilter

logger = logging.getLogger(__name__)

class TelegramCollector(Collector):
    def __init__(self, api_id: int, api_hash: str, channels: List[str], db: Database, filter: SemanticFilter):
        self.api_id = api_id
        self.api_hash = api_hash
        self.channels = channels
        self.db = db
        self.filter = filter
        self._client = None

    async def _fetch_channel(self, channel: str):
        try:
            entity = await self._client.get_entity(channel)
            messages = await self._client.get_messages(entity, limit=50)
            for msg in messages:
                if msg.text and self.filter.is_relevant(msg.text):
                    url = f'https://t.me/{channel}/{msg.id}'
                    self.db.save_article(url, '', msg.text, channel, msg.date.isoformat())
                    logger.info(f"Saved relevant post: {url}")
        except Exception as e:
            logger.error(f"Error in {channel}: {e}")

    async def _run(self):
        self._client = TelegramClient('session_name', self.api_id, self.api_hash)
        await self._client.start()
        for channel in self.channels:
            await self._fetch_channel(channel)
        await self._client.disconnect()

    def collect(self):
        asyncio.run(self._run())
