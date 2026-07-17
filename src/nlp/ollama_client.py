import json
import os

import requests

# Instance Ollama locale — surchargable via variable d'env pour un déploiement
# où Ollama tournerait sur une autre machine/port.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaUnavailableError(Exception):
    """Levée quand Ollama ne répond pas, timeout, ou renvoie une réponse inattendue."""


def is_available(timeout: float = 3.0) -> bool:
    """Health check bon marché — ne lève jamais, retourne juste True/False."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/version", timeout=timeout)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def embed(texts: list[str], model: str = "nomic-embed-text", timeout: float = 60.0) -> list[list[float]]:
    """
    Calcule les embeddings d'une liste de textes en un seul appel batch.

    Utilise /api/embed (pluriel) et non /api/embeddings (singulier, legacy) :
    ce dernier n'accepte qu'un texte à la fois et renvoie des vecteurs non
    normalisés, alors que /api/embed accepte une liste et renvoie des vecteurs
    déjà normalisés L2 (vérifié : norme ≈ 1.0).
    """
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": model, "input": texts},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        embeddings = data["embeddings"]
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        raise OllamaUnavailableError(f"Ollama embed call failed: {e}") from e

    if len(embeddings) != len(texts):
        raise OllamaUnavailableError(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
        )
    return embeddings


def chat_json(
    system: str,
    user: str,
    schema: dict,
    model: str,
    timeout: float = 30.0,
    temperature: float = 0.2,
) -> dict:
    """
    Appelle le modèle de chat avec un schéma JSON structuré (paramètre `format`).

    Le mode JSON structuré est nécessaire, pas optionnel : sans lui, le modèle
    ajoute du préambule/de la numérotation autour de la réponse (fragile à
    parser) et répond ~5x plus lentement (mesuré : 5.1s sans format structuré
    contre 0.9s avec, sur ce modèle/cette machine).
    """
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": schema,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return json.loads(content)
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        raise OllamaUnavailableError(f"Ollama chat call failed: {e}") from e
