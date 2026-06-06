import re
from bs4 import BeautifulSoup

_REDDIT_SOURCES = {"reddit_ml", "reddit_localllama", "reddit_artificial"}

def clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ", strip=True)


def normalize(text: str) -> str:
    text = clean_html(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Reddit publie des pseudo-articles dont le titre est juste l'interface UI
_REDDIT_NOISE = {"link", "submitted", "comments", "score", "by", "posted"}

# Patterns de titres Reddit sans valeur informative pour l'analyse de tendances
_REDDIT_TITLE_NOISE = re.compile(
    r"^\s*("
    r"what (laptop|gpu|pc|computer|setup|model|tool|software|library|framework)"  # achats
    r"|help.{0,30}\?$"           # demandes d'aide
    r"|question about"           # questions génériques
    r"|anyone (know|tried|using|have)"
    r"|how (do|can|should|did) (i|you|we)"
    r"|\[d\]|\[p\]|\[n\]"        # tags Reddit r/ML sans contenu
    r")",
    re.IGNORECASE,
)


def is_valid_article(article: dict) -> bool:
    title = article.get("title", "")
    content = article.get("content", "")
    source = article.get("source", "")

    # Titre trop court
    if len(title) < 15:
        return False

    # Bruit Reddit pur (titres = métadonnées UI)
    words = set(title.lower().split())
    if words and words.issubset(_REDDIT_NOISE):
        return False

    # Filtres spécifiques aux sources Reddit
    if source in _REDDIT_SOURCES:
        # Contenu trop court pour être un vrai article (question, meta-post)
        if len(content) < 80:
            return False
        # Patterns de titres non informatifs pour le clustering
        if _REDDIT_TITLE_NOISE.match(title):
            return False

    return True


def clean_article(article: dict) -> dict:
    article["title"] = normalize(article.get("title", ""))
    article["content"] = normalize(article.get("content", ""))
    return article
