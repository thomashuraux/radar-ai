import hashlib
import requests
from datetime import datetime, timezone

# API Semantic Scholar Graph — gratuite, sans clé pour un usage modéré (~100 req/5min)
SS_API = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_FIELDS = "paperId,title,abstract,publicationDate,year,url,externalIds"

# Requêtes ciblées sur des sujets qui produisent de vraies nouveautés en 2025-2026.
# On a réduit de 7 à 4 requêtes pour limiter la domination de S2 dans le corpus
# (était 69% des articles → objectif ~25-30%).
QUERIES = [
    "large language model agents",
    "multimodal foundation model",
    "reinforcement learning from human feedback",
    "AI alignment safety",
]

_HEADERS = {
    "User-Agent": "AIRadar/1.0 (research aggregator; contact: radarai@local)"
}


def _make_id(paper_id: str) -> str:
    return hashlib.md5(f"ss:{paper_id}".encode()).hexdigest()


def _paper_url(paper: dict) -> str:
    if paper.get("url"):
        return paper["url"]
    arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"


def collect_semanticscholar(max_per_query: int = 15) -> list[dict]:
    """
    Collecte des papers récents via l'API Semantic Scholar.

    Filtres appliqués :
    - year=2025-2026 : on ne prend que des papers récents. Sans ce filtre,
      l'API retourne des papers très cités mais anciens (2017-2020) qui
      faussent le clustering en créant des "clusters académiques" au lieu
      de vraies tendances du jour.
    - max_per_query réduit à 15 (était 25) pour limiter la part de S2
      dans le corpus total.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    seen: set[str] = set()
    articles = []

    for query in QUERIES:
        try:
            resp = requests.get(
                SS_API,
                params={
                    "query": query,
                    "fields": SS_FIELDS,
                    "limit": max_per_query,
                    "year": "2025-2026",   # ← filtre clé : papers récents uniquement
                },
                headers=_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[SemanticScholar] Query '{query}' failed: {e}")
            continue

        for paper in data.get("data", []):
            pid = paper.get("paperId", "")
            if not pid or pid in seen:
                continue
            seen.add(pid)

            title = (paper.get("title") or "").strip()
            abstract = (paper.get("abstract") or "").strip()

            pub_date = paper.get("publicationDate") or str(paper.get("year") or "")
            if len(pub_date) == 4:
                pub_date = f"{pub_date}-01-01"
            if not pub_date or len(pub_date) < 10:
                pub_date = today

            # Sécurité côté client : ignorer tout paper antérieur à 2025
            if pub_date < "2025-01-01":
                continue

            if not title:
                continue

            articles.append({
                "id": _make_id(pid),
                "source": "semanticscholar",
                "title": title,
                "content": abstract,
                "url": _paper_url(paper),
                "date": today,
                "embedding": None,
                "cluster_id": -1,
            })

    return articles
