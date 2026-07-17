import json

import numpy as np
from sklearn.preprocessing import normalize

from src.nlp import ollama_client

EMBED_MODEL = "nomic-embed-text"


def embed_articles(articles: list[dict]) -> list[dict]:
    """
    Calcule et attache un embedding sémantique à chaque article via Ollama
    (nomic-embed-text). Un seul appel batch pour tout le lot — pas de modèle
    fitté à mettre en cache/refit d'un run à l'autre, contrairement à
    l'ancienne approche TF-IDF+SVD.

    Lève OllamaUnavailableError si Ollama est injoignable : à l'appelant de
    décider (abandonner le run plutôt que sauver des clusters à moitié à jour,
    cf. main.py:cmd_analyze et src/api/app.py:_pipeline).
    """
    if not articles:
        return articles

    # On concatène titre + début du contenu pour représenter l'article
    texts = [f"{a['title']}. {a.get('content', '')[:512]}" for a in articles]

    embeddings = ollama_client.embed(texts, model=EMBED_MODEL)
    embeddings = normalize(embeddings)

    for article, emb in zip(articles, embeddings):
        article["embedding"] = emb.tolist()

    return articles


def get_embeddings_matrix(articles: list[dict]) -> np.ndarray:
    """
    Reconstruit la matrice numpy des embeddings depuis les articles.
    Les embeddings sont stockés en JSON string dans SQLite — on les désérialise ici.
    """
    vecs = []
    for a in articles:
        emb = a.get("embedding")
        if isinstance(emb, str):
            emb = json.loads(emb)
        vecs.append(emb)
    return np.array(vecs, dtype=np.float32)
