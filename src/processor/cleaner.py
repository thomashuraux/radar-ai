import re
from bs4 import BeautifulSoup

_REDDIT_SOURCES = {"reddit_ml", "reddit_localllama", "reddit_artificial"}

# Sources qui publient un flux RSS général (non-AI) : on ne garde que les articles
# dont le titre contient un terme clairement lié à l'IA/ML.
_GENERAL_SOURCES = {"the_verge_ai", "wired_ai"}

_AI_TITLE_TERMS = re.compile(
    r"\b(ai\b|artificial intelligence|machine learning|llm|gpt|chatgpt|claude|gemini|openai|anthropic|"
    r"neural|robot|robotics|automation|algorithm|generative|language model|siri|alexa|copilot|"
    r"gpu|nvidia|training|inference|transformer|diffusion|computer vision|nlp|deep learning|"
    r"large model|foundation model|multimodal|agi|chatbot|prompt|embedding|rag|deepfake|deep fake)\b",
    re.IGNORECASE,
)

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
    r"|\[d\]|\[p\]|\[n\]|\[r\]"  # tags Reddit r/ML sans contenu (début de titre)
    r")",
    re.IGNORECASE,
)

# Tags Reddit en fin de titre — [D], [R], [P], [N] peuvent aussi apparaître à la fin
_REDDIT_TAG_SUFFIX = re.compile(r"\s*\[([dpnr])\]\s*$", re.IGNORECASE)


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
        # Tags [D]/[R]/[P]/[N] en fin de titre
        if _REDDIT_TAG_SUFFIX.search(title):
            return False
        # Titres entièrement en minuscules = posts personnels/discussions informelles
        if title == title.lower() and len(title) > 10:
            return False

    # Sources à flux général : garder uniquement les articles AI-pertinents
    if source in _GENERAL_SOURCES:
        if not _AI_TITLE_TERMS.search(title):
            return False

    return True


def clean_article(article: dict) -> dict:
    article["title"] = normalize(article.get("title", ""))
    article["content"] = normalize(article.get("content", ""))
    return article
