# Pumpy

**Semantic news aggregator for scientific literature**

Pumpy collects scientific articles from RSS feeds, filters them by semantic relevance to your interests, and delivers a personalized digest. It combines **embeddings (BAAI/bge-small-en-v1.5)**, **BM25** full‑text search, **MMR** diversity reranking, and user feedback (likes/dislikes) to provide highly relevant and diverse recommendations.

---

## Features

- Automated collection from multiple RSS feeds (bioRxiv, Nature, ASM, APS, etc.)
- Semantic relevance filtering using state‑of‑the‑art sentence‑transformers
- Personalised ranking based on user likes/dislikes (weighted vector averaging)
- Hybrid search (BM25 + embeddings)
- MMR (Maximal Marginal Relevance) for result diversification
- Web interface with like/dislike, article details modal, admin dashboard
- Dynamic RSS feed management (add/remove/activate feeds through admin panel)
- Export digest as BibTeX
- Zero‑input agentic UI (contextual tips, natural language commands)
- SQLite (default) or PostgreSQL support