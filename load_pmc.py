import requests
import time
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from db import Database
from semantic import SemanticFilter
import os

load_dotenv()

db = Database(os.getenv('DATABASE_URL'))
semantic = SemanticFilter(
    model_name=os.getenv('MODEL_NAME', 'BAAI/bge-small-en-v1.5'),
    interest_vector_path=os.getenv('INTEREST_VECTOR_PATH', 'interest_vector.npy'),
    threshold=float(os.getenv('INTEREST_THRESHOLD', '0.1'))
)

TARGET = 5000
# Ключевые слова для поиска (можно расширить)
queries = [
    'genomics',
    'plant biology',
    'microbiology',
    'bioinformatics',
    'molecular biology',
    'biochemistry',
    'genetics',
    'CRISPR'
]

# Базовые URL для E-utilities
base_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
base_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}

total_loaded = 0
failed_articles = []

for query in queries:
    if total_loaded >= TARGET:
        break
    # Поиск по ключевым словам
    search_params = {
        'db': 'pmc',
        'term': query,
        'retmax': 100,          # максимум на запрос
        'retstart': 0,
        'usehistory': 'y',
        'sort': 'relevance'     # можно изменить на 'pub date' для свежих
    }
    while True:
        if total_loaded >= TARGET:
            break
        print(f"Searching for: {query} (loaded: {total_loaded}/{TARGET})")
        try:
            resp = requests.get(base_search, params=search_params, headers=headers, timeout=30)
            if resp.status_code != 200:
                print(f"Error {resp.status_code} for query {query}")
                break
            root = ET.fromstring(resp.text)
            id_list = root.find('IdList')
            if id_list is None:
                break
            pmids = [id_elem.text for id_elem in id_list.findall('Id')]
            if not pmids:
                break

            for pmid in pmids:
                if total_loaded >= TARGET:
                    break
                # Получаем метаданные для каждого PMID
                fetch_params = {
                    'db': 'pmc',
                    'id': pmid,
                    'retmode': 'xml'
                }
                fetch_resp = requests.get(base_fetch, params=fetch_params, headers=headers, timeout=30)
                if fetch_resp.status_code != 200:
                    print(f"Error fetching {pmid}: {fetch_resp.status_code}")
                    failed_articles.append(pmid)
                    continue

                article_xml = fetch_resp.text
                root_art = ET.fromstring(article_xml)
                # Извлекаем title, abstract, date, journal
                # Структура XML может варьироваться, используем простой поиск
                title_elem = root_art.find('.//article-title')
                title = title_elem.text if title_elem is not None else ''
                abstract_elem = root_art.find('.//abstract')
                abstract = ''
                if abstract_elem is not None:
                    # Собираем текст из всех <p> внутри абстракта
                    for p in abstract_elem.findall('.//p'):
                        if p.text:
                            abstract += p.text + ' '
                date_elem = root_art.find('.//pub-date')
                year = ''
                if date_elem is not None:
                    year_elem = date_elem.find('year')
                    if year_elem is not None:
                        year = year_elem.text
                journal_elem = root_art.find('.//journal-title')
                journal = journal_elem.text if journal_elem is not None else 'PMC'
                url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmid}/"

                full_text = title + ' ' + abstract
                if not full_text.strip():
                    print(f"Skipping {pmid}: no text")
                    continue

                try:
                    embedding = semantic.get_embedding(full_text).tobytes()
                    db.save_article(
                        url=url,
                        title=title,
                        text=abstract[:1000],
                        source=journal,
                        date=year,
                        similarity=0.0,
                        embedding=embedding
                    )
                    total_loaded += 1
                    if total_loaded % 100 == 0:
                        print(f"Loaded {total_loaded} articles so far")
                except Exception as e:
                    print(f"Error processing {pmid}: {e}")
                    failed_articles.append(pmid)

                time.sleep(0.5)  # пауза между запросами, чтобы не превысить лимит

            # Переход к следующей странице
            search_params['retstart'] += len(pmids)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error in query {query}: {e}")
            break

print("\n" + "="*50)
print(f"Collection finished. Total articles loaded: {total_loaded}")
if failed_articles:
    print(f"Failed to load {len(failed_articles)} articles:")
    for u in failed_articles[:10]:
        print(u)
    if len(failed_articles) > 10:
        print(f"... and {len(failed_articles)-10} more")
else:
    print("All articles were loaded successfully.")
print("="*50)