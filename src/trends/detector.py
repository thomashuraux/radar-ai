import json
import numpy as np
from datetime import date, timedelta
from collections import defaultdict
from src.storage import db
from src.nlp.keywords import extract_keywords, get_cluster_name
from src.nlp.clusterer import MIN_CLUSTER_FIT


def _cluster_centroid(arts: list[dict]) -> np.ndarray | None:
    vecs = []
    for a in arts:
        emb = a.get("embedding")
        if isinstance(emb, str):
            emb = json.loads(emb)
        if emb:
            vecs.append(np.array(emb, dtype=np.float32))
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def _centroid_title(arts: list[dict]) -> str:
    """Article le plus proche du centroïde — utilisé en fallback si les mots-clés sont vides."""
    vecs = [(a, np.array(json.loads(a["embedding"]) if isinstance(a.get("embedding"), str) else a.get("embedding", []), dtype=np.float32))
            for a in arts if a.get("embedding")]
    if not vecs:
        return ""
    centroid = np.mean([v for _, v in vecs], axis=0)
    closest = min(vecs, key=lambda x: np.linalg.norm(x[1] - centroid))
    return closest[0]["title"]


def _match_yesterday_counts(
    today_keywords: dict[int, list[str]],
    yesterday_clusters: list[dict],
) -> dict[int, int]:
    """
    Mappe chaque cluster d'aujourd'hui vers son équivalent d'hier par recouvrement de mots-clés.

    Les cluster_id sont réassignés à zéro chaque jour (KMeans/HDBSCAN repart de zéro),
    donc comparer les ID directement est faux. Comparer les embeddings ne marche pas non plus :
    le TF-IDF+SVD est refit indépendamment chaque jour (voir src/nlp/embedder.py), donc les
    vecteurs d'aujourd'hui et d'hier vivent dans des bases latentes différentes et non alignées —
    une similarité cosinus entre les deux n'a pas de sens mathématique. On compare à la place les
    ensembles de mots-clés persistés, qui sont indépendants de tout espace vectoriel.

    Seuil de 0.25 sur le coefficient de chevauchement (overlap / min(len_a, len_b)) : environ
    "2 mots-clés communs sur 8" suffisent à considérer que c'est le même sujet. Ajustable.
    """
    result: dict[int, int] = {}
    for cid, keywords in today_keywords.items():
        today_set = set(keywords)
        if not today_set:
            result[cid] = 0
            continue

        best_overlap = 0.0
        best_count = 0
        for yc in yesterday_clusters:
            yest_set = set(yc.get("keywords", []))
            if not yest_set:
                continue
            # Coefficient de chevauchement : robuste à l'asymétrie de longueur des
            # listes de mots-clés entre aujourd'hui et hier.
            overlap = len(today_set & yest_set) / min(len(today_set), len(yest_set))
            if overlap > best_overlap:
                best_overlap = overlap
                best_count = yc.get("article_count", 0)

        result[cid] = best_count if best_overlap >= 0.25 else 0

    return result


def compute_trend_score(count_today: int, count_yesterday: int) -> float:
    """
    Score de tendance : combine volume absolu et croissance relative.

    Formule : count * 0.6 + growth_rate * 0.4

    - count * 0.6 : un sujet avec 50 articles est intrinsèquement plus important
      qu'un sujet avec 3 articles, même si les deux ont doublé.
    - growth_rate * 0.4 : un sujet qui explose aujourd'hui remonte dans le classement
      même s'il était petit hier.
    """
    growth_rate = (count_today - count_yesterday) / max(1, count_yesterday)
    return round(count_today * 0.6 + growth_rate * 0.4, 4)


def build_clusters(articles: list[dict], target_date: str) -> list[dict]:
    yesterday = (date.fromisoformat(target_date) - timedelta(days=1)).isoformat()
    yesterday_clusters = db.get_clusters_by_date(yesterday)

    cluster_articles: dict[int, list[dict]] = defaultdict(list)
    for a in articles:
        cid = a.get("cluster_id", -1)
        if cid == -1:
            continue
        cluster_articles[cid].append(a)

    # Corpus global du jour pour l'IDF : tous les articles, pas seulement ceux du cluster.
    # Permet à TF-IDF de pénaliser "large language model" qui apparaît partout.
    all_texts = [f"{a['title']} {a.get('content', '')[:200]}" for a in articles]

    # Mots-clés calculés en premier pour chaque cluster, nécessaires au matching cross-jour
    today_keywords: dict[int, list[str]] = {}
    cluster_names: dict[int, str] = {}
    for cid, arts in cluster_articles.items():
        texts = [f"{a['title']} {a.get('content', '')[:200]}" for a in arts]
        keywords = extract_keywords(texts, corpus=all_texts)
        today_keywords[cid] = keywords

        # Mots-clés en priorité : ils sont extraits de TOUS les articles du cluster
        # et représentent le sujet commun. Le titre centroïde est un seul article —
        # il peut être trompeur si le cluster est hétérogène.
        cluster_names[cid] = get_cluster_name(keywords) or _centroid_title(arts)

    # Matching cross-jour par recouvrement de mots-clés (les ID et les embeddings
    # ne sont pas stables/comparables d'un jour à l'autre, cf. _match_yesterday_counts)
    yesterday_counts = _match_yesterday_counts(today_keywords, yesterday_clusters)

    duplicate_sources = db.get_duplicate_sources_by_date(target_date)

    clusters = []
    for cid, arts in cluster_articles.items():
        keywords = today_keywords[cid]
        name = cluster_names[cid]

        count_today = len(arts)
        count_yest = yesterday_counts.get(cid, 0)
        score = compute_trend_score(count_today, count_yest)

        top_arts = sorted(arts, key=lambda x: len(x.get("content", "")), reverse=True)[:3]
        top_titles = [{"title": a["title"], "url": a.get("url", ""), "source": a.get("source", "")} for a in top_arts]

        # Cohésion : similarité moyenne des membres à leur centroïde (cf. clusterer.py).
        # None pour les articles dont le cluster n'a jamais été "gaté" (cas <5 articles/jour).
        fits = [a["cluster_fit"] for a in arts if a.get("cluster_fit") is not None]
        cohesion = round(float(np.mean(fits)), 4) if fits else 0.0

        # Diversité des sources : réunit les sources des articles du cluster ET celles
        # des doublons fusionnés (exclus du clustering mais toujours une corroboration
        # réelle — cf. src/processor/deduplicator.py).
        sources = {a.get("source", "") for a in arts}
        for a in arts:
            sources.update(duplicate_sources.get(a["id"], []))
        sources.discard("")
        sources = sorted(sources)

        clusters.append({
            "id": cid,
            "name": name,
            "keywords": keywords,
            "article_count": count_today,
            "yesterday_count": count_yest,
            "trend_score": score,
            "top_titles": top_titles,
            "articles": arts,
            "cohesion": cohesion,
            "sources": sources,
            "source_count": len(sources),
            "low_confidence": cohesion < MIN_CLUSTER_FIT,
        })

    clusters.sort(key=lambda c: -c["trend_score"])
    return clusters
