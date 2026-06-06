import numpy as np

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def cluster_articles(articles: list[dict], embeddings: np.ndarray) -> list[dict]:
    """
    Groupe les articles par sujet via clustering dans l'espace d'embeddings.

    Stratégie : HDBSCAN en priorité, KMeans en fallback.

    HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) :
      - Ne nécessite pas de spécifier le nombre de clusters à l'avance.
      - Détecte des clusters de forme arbitraire (pas forcément sphériques).
      - Marque les articles trop isolés comme "bruit" (label = -1).
      - Inconvénient : avec des embeddings TF-IDF peu denses, il peut tout
        classifier en bruit. D'où le fallback KMeans.

    KMeans :
      - Requiert un k fixe, mais garantit que chaque article est assigné.
      - Moins fin que HDBSCAN mais robuste sur les espaces TF-IDF.
    """
    # Pas la peine de clusterer 4 articles — un seul groupe suffit
    if len(articles) < 5:
        for a in articles:
            a["cluster_id"] = 0
        return articles

    # Renormaliser pour garantir des vecteurs unitaires
    # (HDBSCAN euclidean ≡ cosine similarity sur vecteurs normalisés)
    embs = normalize(embeddings)

    if HAS_HDBSCAN and len(articles) >= 10:
        clusterer = hdbscan.HDBSCAN(
            # Un cluster doit contenir au moins N/15 articles (légèrement plus strict
            # que N/20) pour éviter les micro-clusters trop spécifiques.
            min_cluster_size=max(3, len(articles) // 15),
            min_samples=1,
            metric="euclidean",
            # epsilon=0.0 : pas de fusion forcée entre clusters distincts.
            # Valeur 0.1 fusionnait des sujets trop éloignés → clusters incohérents.
            cluster_selection_epsilon=0.0,
        )
        labels = clusterer.fit_predict(embs)

        # Si HDBSCAN classe moins de 30% des articles (reste = bruit),
        # ses résultats ne sont pas exploitables → on bascule sur KMeans.
        valid = sum(1 for l in labels if l >= 0)
        if valid < len(articles) * 0.3:
            n_clusters = min(12, max(4, len(articles) // 6))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embs)
    else:
        # Moins de 10 articles ou HDBSCAN absent → KMeans directement
        n_clusters = min(12, max(4, len(articles) // 6))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embs)

    for article, label in zip(articles, labels):
        article["cluster_id"] = int(label)

    return articles
