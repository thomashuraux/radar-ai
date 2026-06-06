import json
import numpy as np
from datetime import date, timedelta
from collections import defaultdict
from src.storage import db
from src.nlp.keywords import extract_keywords, get_cluster_name


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
    today_clusters: dict[int, list[dict]],
    yesterday_articles: list[dict],
) -> dict[int, int]:
    """
    Mappe chaque cluster d'aujourd'hui vers son équivalent d'hier par similarité cosinus.

    Les cluster_id sont réassignés à zéro chaque jour (KMeans/HDBSCAN repart de zéro),
    donc comparer les ID directement est faux. On compare les centroïdes à la place :
    le cluster d'hier le plus proche du cluster d'aujourd'hui est son "ancêtre" thématique.

    Un seuil de 0.5 évite de matcher des clusters sans rapport.
    """
    yest_by_cid: dict[int, list[dict]] = defaultdict(list)
    for a in yesterday_articles:
        cid = a.get("cluster_id", -1)
        if cid >= 0:
            yest_by_cid[cid].append(a)

    # Pré-calcul des centroïdes d'hier
    yest_centroids: list[tuple[np.ndarray, int]] = []
    for arts in yest_by_cid.values():
        c = _cluster_centroid(arts)
        if c is not None:
            yest_centroids.append((c, len(arts)))

    if not yest_centroids:
        return {}

    result: dict[int, int] = {}
    for cid, arts in today_clusters.items():
        today_c = _cluster_centroid(arts)
        if today_c is None:
            result[cid] = 0
            continue

        today_norm = np.linalg.norm(today_c)
        if today_norm == 0:
            result[cid] = 0
            continue

        best_sim = 0.0
        best_count = 0
        for yest_c, count in yest_centroids:
            # Les embeddings peuvent avoir des dimensions différentes si le corpus
            # d'hier était plus petit (SVD tronqué à min(128, n-1)). On tronque au minimum.
            dim = min(len(today_c), len(yest_c))
            tc, yc = today_c[:dim], yest_c[:dim]
            yest_norm = np.linalg.norm(yc)
            today_norm_d = np.linalg.norm(tc)
            if yest_norm == 0 or today_norm_d == 0:
                continue
            sim = float(np.dot(tc, yc) / (today_norm_d * yest_norm))
            if sim > best_sim:
                best_sim = sim
                best_count = count

        result[cid] = best_count if best_sim >= 0.5 else 0

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
    yesterday_articles = db.get_articles_by_date(yesterday)

    cluster_articles: dict[int, list[dict]] = defaultdict(list)
    for a in articles:
        cid = a.get("cluster_id", -1)
        if cid == -1:
            continue
        cluster_articles[cid].append(a)

    # Matching cross-jour par similarité cosinus des centroïdes (les ID ne sont pas stables)
    yesterday_counts = _match_yesterday_counts(cluster_articles, yesterday_articles)

    # Corpus global du jour pour l'IDF : tous les articles, pas seulement ceux du cluster.
    # Permet à TF-IDF de pénaliser "large language model" qui apparaît partout.
    all_texts = [f"{a['title']} {a.get('content', '')[:200]}" for a in articles]

    clusters = []
    for cid, arts in cluster_articles.items():
        texts = [f"{a['title']} {a.get('content', '')[:200]}" for a in arts]
        keywords = extract_keywords(texts, corpus=all_texts)

        # Mots-clés en priorité : ils sont extraits de TOUS les articles du cluster
        # et représentent le sujet commun. Le titre centroïde est un seul article —
        # il peut être trompeur si le cluster est hétérogène.
        name = get_cluster_name(keywords) or _centroid_title(arts)

        count_today = len(arts)
        count_yest = yesterday_counts.get(cid, 0)
        score = compute_trend_score(count_today, count_yest)

        top_arts = sorted(arts, key=lambda x: len(x.get("content", "")), reverse=True)[:3]
        top_titles = [{"title": a["title"], "url": a.get("url", ""), "source": a.get("source", "")} for a in top_arts]

        clusters.append({
            "id": cid,
            "name": name,
            "keywords": keywords,
            "article_count": count_today,
            "yesterday_count": count_yest,
            "trend_score": score,
            "top_titles": top_titles,
            "articles": arts,
        })

    clusters.sort(key=lambda c: -c["trend_score"])
    return clusters
