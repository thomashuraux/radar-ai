from sklearn.feature_extraction.text import TfidfVectorizer
import re


# Mots à exclure du calcul des mots-clés.
# sklearn fournit une liste "english" standard, mais elle ne couvre pas
# les termes génériques du domaine IA ni le bruit des interfaces web (Reddit).
_AI_STOPWORDS = {
    "said", "says", "also", "new", "one", "like", "just", "use",
    "using", "used", "year", "years", "way", "make", "making",
    "made", "get", "got", "work", "working", "need", "needs",
    "even", "still", "first", "last", "week", "day", "time",
    "researchers", "research", "company", "companies", "team",
    "according", "report", "model", "models",
    # Bruit Reddit
    "comments", "comment", "link", "submitted", "score", "points",
    "posted", "upvotes", "reddit", "subreddit", "thread",
    # Termes génériques
    "article", "read", "post", "posts", "people", "thing", "things",
    "help", "ceo", "user", "users", "data", "many", "much",
    "know", "think", "want", "really", "actually", "would", "could",
    "different", "better", "good", "bad", "big", "small", "high",
    "well", "may", "will", "can", "let", "take", "give", "right",
    # Termes omniprésents dans les articles IA — aucune valeur discriminante
    # pour nommer un cluster spécifique (apparaissent dans ~100% des articles)
    "large", "language", "llm", "llms", "ai", "artificial", "intelligence",
    "neural", "deep", "learning", "network", "networks",
    "training", "trained",
    "benchmark", "performance", "state", "art", "approach", "method",
    "results", "result", "experiments", "evaluation", "dataset", "task", "tasks",
    "based", "proposed", "paper", "show", "shows", "achieve", "achieves",
    "improved", "improve", "ability", "capabilities", "capability",
    "output", "input", "human", "number", "type", "types", "level",
    "open", "source", "release", "released", "version", "latest",
    "say", "including", "within", "across", "without", "existing",
    "current", "general", "specific", "multiple", "various", "novel",
    "enable", "enables", "enabled", "allow", "allows", "allowed",
    "generate", "generated", "generation", "response", "responses",
    "query", "queries", "context", "information", "knowledge",
    # Noms propres politiques — jamais pertinents comme nom de trend IA
    "trump", "krishnan", "sriram", "biden", "administration",
    # Termes vagues qui polluent les noms de clusters
    "meet", "world", "domains", "domain", "significant", "diverse",
    "comprehensive", "addresses", "addressing", "increasingly", "ensuring",
    "shaping", "understanding", "survey", "review", "overview", "introduction",
    "early", "long", "role", "opinion", "playing", "image", "images",
    "thought", "thoughts", "fun", "coding", "code", "design", "development",
    "complex", "powerful", "efficient", "effective", "accurate", "robust",
    "challenging", "difficult", "simple", "flexible", "scalable",
}


def extract_keywords(texts: list[str], top_n: int = 8, corpus: list[str] | None = None) -> list[str]:
    """
    Extrait les mots-clés les plus représentatifs d'un ensemble de textes.

    Si `corpus` est fourni, l'IDF est calculé sur ce corpus global (tous les articles
    du jour) plutôt que sur les seuls articles du cluster. Cela pénalise les termes
    omniprésents comme "large language model" qui apparaissent dans tous les clusters
    et n'ont aucune valeur discriminante.
    """
    if not texts:
        return []

    clean = [re.sub(r"[^a-zA-Z0-9 ]", " ", t).lower() for t in texts]
    corpus_clean = [re.sub(r"[^a-zA-Z0-9 ]", " ", t).lower() for t in corpus] if corpus else clean

    try:
        # min_df=2 quand le cluster a plus de 3 articles : un terme qui n'apparaît
        # que dans 1 article (nom propre, hapax) ne doit pas nommer tout le cluster.
        # Ex : "Benn" (nom d'un auteur dans 1 seul titre) était remonté en #1.
        min_df = 2 if len(texts) > 3 else 1

        vec = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            stop_words="english",
            min_df=min_df,
        )
        # Fitter l'IDF sur le corpus global pour que les termes omniprésents
        # (ex: "large language") soient correctement pénalisés
        vec.fit(corpus_clean)
        matrix = vec.transform(clean)

        scores = matrix.sum(axis=0).A1
        vocab = vec.get_feature_names_out()

        ranked = sorted(zip(vocab, scores), key=lambda x: -x[1])

        keywords = []
        for term, _ in ranked:
            words = term.split()
            if any(w in _AI_STOPWORDS for w in words):
                continue
            if any(len(w) <= 2 for w in words):
                continue
            keywords.append(term)
            if len(keywords) >= top_n:
                break

        return keywords
    except Exception:
        return []


def get_cluster_name(keywords: list[str]) -> str:
    # Fallback quand le titre centroïde n'est pas disponible :
    # on forge un nom lisible à partir des 3 meilleurs mots-clés
    if not keywords:
        return "Unknown topic"
    top = keywords[:3]
    return " · ".join(w.title() for w in top)
