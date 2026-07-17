import numpy as np

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Similarité cosinus minimale d'un article à son centroïde de cluster pour y rester.
# En dessous, l'article est rétrogradé en bruit (-1) plutôt que forcé dans un groupe
# auquel il ne ressemble pas — c'est le garde-fou qui manquait à KMeans (qui autrement
# assigne TOUJOURS chaque article à son centroïde le plus proche, aussi mauvais soit-il).
#
# Calibré sur des embeddings nomic-embed-text (via Ollama) réels, sur deux journées
# de production (cf. `main.py inspect` sur 2026-07-15/16) : les similarités cosinus de
# ce modèle vivent dans une bande beaucoup plus resserrée et plus haute que l'ancien
# espace TF-IDF+SVD (0.15-0.42) — même un cluster hétéroclite mélangeant des sujets
# sans rapport n'était jamais descendu sous ~0.69 de similarité à son propre centroïde.
# La séparation coherent/incohérent ne se fait donc plus via ce seuil seul : c'est
# `cluster_selection_method="leaf"` ci-dessous qui empêche HDBSCAN de fusionner des
# sous-sujets distincts en un seul gros cluster ; ce seuil de 0.55 reste un filet de
# sécurité sous le plancher observé des clusters sains (~0.74-0.88), pas le mécanisme
# principal de séparation. À recalibrer si le backend d'embeddings change à nouveau.
MIN_CLUSTER_FIT = 0.55

# Taille minimale d'un cluster après rétrogradation des membres mal ajustés — un
# "sujet" porté par un seul article restant n'en est plus un.
MIN_CLUSTER_SIZE_AFTER_GATING = 2


def _gate_by_cohesion(embs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Rétrograde en bruit (-1) tout article trop éloigné du centroïde de son cluster,
    puis dissout les clusters devenus trop petits une fois ces membres retirés.
    """
    labels = labels.copy()
    unique_labels = {l for l in labels.tolist() if l != -1}

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        centroid = normalize(embs[idx].mean(axis=0, keepdims=True))[0]
        sims = embs[idx] @ centroid
        for i, sim in zip(idx, sims):
            if sim < MIN_CLUSTER_FIT:
                labels[i] = -1

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if 0 < len(idx) < MIN_CLUSTER_SIZE_AFTER_GATING:
            labels[idx] = -1

    return labels


def _attach_cluster_fit(articles: list[dict], embs: np.ndarray, labels: np.ndarray) -> None:
    """Attache à chaque article sa similarité cosinus au centroïde final (None si bruit)."""
    unique_labels = {l for l in labels.tolist() if l != -1}
    centroids = {
        label: normalize(embs[np.where(labels == label)[0]].mean(axis=0, keepdims=True))[0]
        for label in unique_labels
    }

    for article, emb, label in zip(articles, embs, labels):
        article["cluster_fit"] = float(emb @ centroids[label]) if label in centroids else None


def cluster_articles(articles: list[dict], embeddings: np.ndarray) -> list[dict]:
    """
    Groupe les articles par sujet via clustering dans l'espace d'embeddings.

    Stratégie : HDBSCAN en priorité, KMeans en fallback, puis un filtre de cohésion
    commun aux deux (voir _gate_by_cohesion) qui rejette les membres mal ajustés
    au lieu de les forcer dans un groupe auquel ils ne ressemblent pas.

    HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) :
      - Ne nécessite pas de spécifier le nombre de clusters à l'avance.
      - Détecte des clusters de forme arbitraire (pas forcément sphériques).
      - Marque les articles trop isolés comme "bruit" (label = -1).
      - Inconvénient : sur un espace peu discriminant (TF-IDF creux, ou même des
        embeddings neuronaux denses où "actualité IA" domine la variance), il peut
        soit tout classifier en bruit, soit à l'inverse fusionner plusieurs sujets
        distincts en un seul gros cluster stable. D'où le fallback KMeans pour le
        premier cas, et `cluster_selection_method="leaf"` pour le second (voir plus
        bas).

    KMeans :
      - Requiert un k fixe, mais garantit que chaque article est assigné.
      - Moins fin que HDBSCAN mais robuste sur les espaces TF-IDF.
      - Sans le filtre de cohésion, un article isolé thématiquement serait quand
        même assigné à son centroïde le plus proche, aussi éloigné soit-il.
    """
    # Pas la peine de clusterer 4 articles — un seul groupe suffit
    if len(articles) < 5:
        for a in articles:
            a["cluster_id"] = 0
            a["cluster_fit"] = None
        return articles

    # Renormaliser pour garantir des vecteurs unitaires
    # (HDBSCAN euclidean ≡ cosine similarity sur vecteurs normalisés)
    embs = normalize(embeddings)

    if HAS_HDBSCAN and len(articles) >= 10:
        clusterer = hdbscan.HDBSCAN(
            # Calibré sur des embeddings nomic-embed-text réels (cf. commentaire de
            # MIN_CLUSTER_FIT ci-dessus) : un plancher fixe de 4 (avec une légère
            # croissance pour de plus gros corpus) produit des clusters étroits et
            # thématiquement homogènes ; l'ancienne formule N/15 (~7-8 pour un corpus
            # quotidien type) était trop permissive et laissait `cluster_selection_method`
            # par défaut ("eom") fusionner des sujets distincts en un seul gros cluster.
            min_cluster_size=max(4, len(articles) // 30),
            # min_samples=2 (au lieu de 1, le réglage le plus permissif de HDBSCAN) :
            # limite l'effet de chaînage single-link qui laissait un cluster s'étendre
            # de proche en proche jusqu'à absorber des articles sans rapport entre eux.
            min_samples=2,
            metric="euclidean",
            # epsilon=0.0 : pas de fusion forcée entre clusters distincts.
            # Valeur 0.1 fusionnait des sujets trop éloignés → clusters incohérents.
            cluster_selection_epsilon=0.0,
            # "leaf" plutôt que le défaut "eom" (excess of mass) : eom préfère le
            # cluster le plus "stable" dans la hiérarchie, ce qui sur des embeddings
            # nomic-embed-text tend à choisir un unique gros cluster parent regroupant
            # plusieurs sous-sujets (ex. observé : un cluster de 42 articles mélangeant
            # jeux Roblox, sécurité d'agents IA et modèles de conduite autonome).
            # "leaf" sélectionne les clusters les plus fins de l'arbre, ce qui donne des
            # groupes plus petits mais réellement homogènes (vérifié sur données réelles
            # 2026-07-15/16 — cf. `main.py inspect`).
            cluster_selection_method="leaf",
        )
        labels = clusterer.fit_predict(embs)

        # Avec cluster_selection_method="leaf", un taux de bruit élevé (60-70%) est
        # normal et attendu (la plupart des articles d'une journée n'ont pas 4+
        # équivalents thématiques ce jour-là) — ce n'est plus un signe d'échec de
        # HDBSCAN. On ne bascule sur KMeans que si la clusterisation est vraiment
        # dégénérée (quasi aucune structure trouvée), pas juste sélective.
        valid = sum(1 for l in labels if l >= 0)
        if valid < len(articles) * 0.15:
            n_clusters = min(12, max(4, len(articles) // 6))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embs)
    else:
        # Moins de 10 articles ou HDBSCAN absent → KMeans directement
        n_clusters = min(12, max(4, len(articles) // 6))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embs)

    labels = _gate_by_cohesion(embs, labels)
    _attach_cluster_fit(articles, embs, labels)

    for article, label in zip(articles, labels):
        article["cluster_id"] = int(label)

    return articles
