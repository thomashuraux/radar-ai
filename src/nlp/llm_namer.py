import hashlib

import numpy as np

from src.storage import db
from src.nlp import ollama_client

NAME_MODEL = "llama3.1:8b"

_TITLE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
}

_SYSTEM_PROMPT = (
    "You are a terse tech news editor. Given headlines that all belong to the "
    "same news topic cluster, produce ONE short, specific headline (max 8 words) "
    "naming the shared topic — not a generic category. No quotation marks, no "
    "trailing punctuation. Respond with JSON only."
)

_MAX_TITLE_LEN = 100
_MAX_ARTICLES_IN_PROMPT = 8


def compute_articles_hash(article_ids: list[str]) -> str:
    """
    Clé de cache basée sur la composition du cluster (IDs triés), pas sur
    cluster_id : les IDs de cluster sont ré-assignés arbitrairement à chaque
    run (HDBSCAN/KMeans repartent de zéro), donc seule la composition réelle
    identifie stablement "le même cluster" d'un run à l'autre.
    """
    return hashlib.sha256("|".join(sorted(article_ids)).encode()).hexdigest()


def _build_prompt(arts: list[dict]) -> str:
    # Les articles les plus représentatifs (meilleur cluster_fit) en premier ;
    # ceux sans cluster_fit (cas <5 articles/jour, jamais gaté) gardent leur
    # ordre d'origine et passent en dernier.
    ranked = sorted(
        arts,
        key=lambda a: (a.get("cluster_fit") is None, -(a.get("cluster_fit") or 0)),
    )
    lines = [f"- {a['title']}" for a in ranked[:_MAX_ARTICLES_IN_PROMPT]]
    return "Article titles:\n" + "\n".join(lines)


def _cluster_cohesion(arts: list[dict]) -> float:
    fits = [a["cluster_fit"] for a in arts if a.get("cluster_fit") is not None]
    return float(np.mean(fits)) if fits else 0.0


def generate_cluster_title(
    arts: list[dict], fallback_name: str, ollama_available: bool
) -> tuple[str, str]:
    """
    Génère un titre de cluster via llama3.1:8b, avec dégradation gracieuse
    vers le nom heuristique (mots-clés TF-IDF) si Ollama est indisponible ou
    si l'appel échoue. Un échec de nommage n'interrompt jamais le run — il ne
    dégrade que ce cluster.
    """
    if not ollama_available:
        return fallback_name, "heuristic-keywords"

    article_hash = compute_articles_hash([a["id"] for a in arts])

    # On ne réutilise le cache que si le hit précédent était un vrai résultat LLM :
    # un cluster tombé en fallback pendant une panne ne doit pas rester bloqué
    # dessus indéfiniment une fois Ollama de nouveau disponible.
    cached = db.get_cached_label(article_hash)
    if cached and cached.get("labeling_method") == "llm" and cached.get("name"):
        return cached["name"], "llm"

    try:
        result = ollama_client.chat_json(
            system=_SYSTEM_PROMPT,
            user=_build_prompt(arts),
            schema=_TITLE_SCHEMA,
            model=NAME_MODEL,
        )
        title = result["title"].strip().strip('"')
        if not title or len(title) > _MAX_TITLE_LEN:
            raise ValueError(f"empty or oversized title: {title!r}")
    except Exception:
        return fallback_name, "heuristic-keywords"

    db.save_cached_label(article_hash, title, "", "llm", _cluster_cohesion(arts))
    return title, "llm"
