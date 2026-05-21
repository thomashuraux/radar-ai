import hashlib
import requests
from datetime import datetime, timezone

# API Semantic Scholar Graph — gratuite, sans clé pour un usage modéré (~100 req/5min)
SS_API = "https://api.semanticscholar.org/graph/v1/paper/search"

# Champs demandés à l'API : on ne récupère que ce dont on a besoin
# pour éviter des réponses trop lourdes
SS_FIELDS = "paperId,title,abstract,publicationDate,year,url,externalIds"

# On lance plusieurs requêtes thématiques plutôt qu'une seule requête large.
# Cela donne plus de diversité dans les papers collectés.
QUERIES = [
    "large language model",
    "deep learning",
    "multi-agent",
    "transformer",
    "foundation model",
    "reinforcement learning agent",
    "multimodal AI vision",
]

_HEADERS = {
    "User-Agent": "AIRadar/1.0 (research aggregator; contact: radarai@local)"
}


def _make_id(paper_id: str) -> str:
    return hashlib.md5(f"ss:{paper_id}".encode()).hexdigest()


def _paper_url(paper: dict) -> str:
    # Priorité : URL directe S2, sinon lien arXiv si le paper y est référencé,
    # sinon URL de la page S2 construite depuis le paperId
    if paper.get("url"):
        return paper["url"]
    arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"


def collect_semanticscholar(max_per_query: int = 25) -> list[dict]:
    """
    Collecte des papers via l'API Semantic Scholar.

    On déduplique par paperId pour éviter qu'un paper apparaisse
    dans plusieurs requêtes thématiques.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    seen: set[str] = set()  # paperId déjà vus dans ce run
    articles = []

    for query in QUERIES:
        try:
            resp = requests.get(
                SS_API,
                params={"query": query, "fields": SS_FIELDS, "limit": max_per_query},
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

            # publicationDate = "YYYY-MM-DD" ou None ; year = entier ex: 2024
            pub_date = paper.get("publicationDate") or str(paper.get("year") or "")
            if len(pub_date) == 4:
                # Seule l'année est connue → on prend le 1er janvier
                pub_date = f"{pub_date}-01-01"
            if not pub_date or len(pub_date) < 10:
                pub_date = today

            if not title:
                continue

            articles.append({
                "id": _make_id(pid),
                "source": "semanticscholar",
                "title": title,
                "content": abstract,
                "url": _paper_url(paper),
                "date": pub_date,
                "embedding": None,
                "cluster_id": -1,
            })

    return articles
