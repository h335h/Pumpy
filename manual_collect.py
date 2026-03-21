import os
from dotenv import load_dotenv
from db import Database
from semantic import SemanticFilter
from collectors.rss_collector import RssCollector
from bm25_indexer import BM25Indexer
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()

db = Database(os.getenv('DATABASE_URL', 'sqlite:///articles.db'))
semantic = SemanticFilter(
    model_name=os.getenv('MODEL_NAME', 'BAAI/bge-small-en-v1.5'),
    interest_vector_path=os.getenv('INTEREST_VECTOR_PATH', 'interest_vector.npy'),
    threshold=float(os.getenv('INTEREST_THRESHOLD', '0.1'))
)

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

bm25_indexer = BM25Indexer(db)
bm25_indexer.refresh()

print("Сбор завершён, BM25 индекс обновлён.")
