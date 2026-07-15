import hashlib
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

# API publique arXiv — pas de clé requise, rate limit ~3 req/s
ARXIV_API = "http://export.arxiv.org/api/query"

# Catégories arXiv surveillées :
#   cs.AI  — Intelligence artificielle générale
#   cs.LG  — Machine learning
#   cs.CL  — Traitement du langage naturel (NLP)
#   cs.CV  — Vision par ordinateur
#   cs.RO  — Robotique
CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.RO"]

# Namespace XML Atom utilisé par l'API arXiv
NS = {"atom": "http://www.w3.org/2005/Atom"}


def _make_id(arxiv_id: str) -> str:
    # On hash l'ID arXiv pour avoir un identifiant court et uniforme
    # avec les IDs des autres sources (qui sont aussi des MD5)
    return hashlib.md5(f"arxiv:{arxiv_id}".encode()).hexdigest()


def collect_arxiv(max_results: int = 50) -> list[dict]:
    """
    Collecte les derniers papers arXiv pour les catégories IA.

    L'API retourne du XML Atom. On construit une requête OR sur toutes
    les catégories, triée par date de soumission décroissante.
    """
    # Requête : cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR ...
    query = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    try:
        resp = requests.get(ARXIV_API, params=params, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[arXiv] Request failed: {e}")
        return []

    root = ET.fromstring(resp.text)
    articles = []

    for entry in root.findall("atom:entry", NS):
        title_el = entry.find("atom:title", NS)
        summary_el = entry.find("atom:summary", NS)
        id_el = entry.find("atom:id", NS)
        # On cherche le lien "alternate" = page HTML du paper (pas le PDF)
        link_el = entry.find("atom:link[@rel='alternate']", NS)

        title = (title_el.text or "").strip().replace("\n", " ")
        summary = (summary_el.text or "").strip()
        arxiv_id = (id_el.text or "").strip()
        link = (link_el.attrib.get("href", "") if link_el is not None else arxiv_id)

        # On utilise la date d'aujourd'hui (UTC) plutôt que la date de soumission
        # arXiv réelle (<published>) : le cycle d'annonce d'arXiv a environ un jour
        # de décalage, donc utiliser la date de soumission ferait apparaître la
        # plupart des papers datés d'hier (ou avant), et ils disparaîtraient
        # de la vue "aujourd'hui".
        published = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if not title:
            continue

        articles.append({
            "id": _make_id(arxiv_id),
            "source": "arxiv",
            "title": title,
            "content": summary,   # le résumé arXiv = notre "contenu"
            "url": link,
            "date": published,
            "embedding": None,
            "cluster_id": -1,
        })

    return articles
