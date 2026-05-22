from datetime import datetime
from src.nlp.keywords import get_cluster_name


def _cluster_name(cluster: dict) -> str:
    # Priorité : nom sauvegardé en base (titre centroïde calculé lors de l'analyse).
    # Fallback : nom forgé à partir des mots-clés TF-IDF (moins précis mais lisible).
    name = cluster.get("name")
    if name:
        return name
    return get_cluster_name(cluster.get("keywords", []))


# sumy est optionnel — si absent, on bascule sur le fallback textuel
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    HAS_SUMY = True
except ImportError:
    HAS_SUMY = False


def extractive_summary(text: str, sentences: int = 2) -> str:
    """
    Résumé extractif : sélectionne les phrases les plus représentatives du texte.

    TextRank (algorithme de sumy) est inspiré de PageRank :
    - Chaque phrase est un "nœud"
    - Deux phrases similaires se "votent" mutuellement
    - Les phrases avec le plus de votes (= les plus centrales) sont retenues

    C'est une approche 100% locale, sans LLM, et souvent suffisante pour
    donner une idée du sujet d'un cluster.
    """
    if not text or not HAS_SUMY:
        return _fallback_summary(text)

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        result = summarizer(parser.document, sentences)
        summary = " ".join(str(s) for s in result)
        return summary if summary.strip() else _fallback_summary(text)
    except Exception:
        return _fallback_summary(text)


def _fallback_summary(text: str) -> str:
    # Si sumy n'est pas disponible ou échoue, on prend simplement les 2 premières
    # phrases du texte (heuristique simple mais fiable)
    if not text:
        return ""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 30]
    return ". ".join(sentences[:2]) + "." if sentences else text[:200]


def generate_digest(clusters: list[dict], target_date: str, top_n: int = 5) -> str:
    """Génère le digest en texte brut (pour la CLI)."""
    top = clusters[:top_n]
    date_fmt = datetime.fromisoformat(target_date).strftime("%d %B %Y")

    lines = [
        f"AI Radar — {date_fmt}",
        "=" * 50,
        "",
        f"Top tendances du jour ({len(top)} sujets detectes) :",
        "",
    ]

    for i, cluster in enumerate(top, 1):
        name = _cluster_name(cluster)
        count = cluster["article_count"]
        yest = cluster.get("yesterday_count", 0)
        score = cluster["trend_score"]
        keywords = cluster.get("keywords", [])[:5]
        top_titles = cluster.get("top_titles", [])

        # Calcul de la croissance affichée
        if yest > 0:
            growth_pct = int((count - yest) / yest * 100)
            growth_str = f"+{growth_pct}%" if growth_pct >= 0 else f"{growth_pct}%"
        else:
            growth_str = "nouveau"

        # Résumé extractif sur les 5 articles les plus longs du cluster
        all_text = " ".join(
            f"{a['title']} {a.get('content','')}"
            for a in cluster.get("articles", [])[:5]
        )
        summary = extractive_summary(all_text, sentences=2)

        lines.append(f"{'—'*50}")
        lines.append(f"#{i} {name.upper()}")
        lines.append(f"Signal : {count} articles ({growth_str} vs hier) | score: {score:.2f}")
        lines.append("")
        if summary:
            lines.append(f"  {summary}")
            lines.append("")
        if keywords:
            lines.append(f"  Mots-cles : {', '.join(keywords[:5])}")
        if top_titles:
            lines.append("")
            lines.append("  Sources cles :")
            for t in top_titles[:3]:
                label = t["title"] if isinstance(t, dict) else t
                url = t.get("url", "") if isinstance(t, dict) else ""
                lines.append(f"    - {label}" + (f" ({url})" if url else ""))
        lines.append("")

    lines.append("=" * 50)
    lines.append(f"Total articles collectes aujourd'hui : {sum(c['article_count'] for c in clusters)}")
    lines.append(f"Clusters detectes : {len(clusters)}")
    lines.append("=" * 50)

    return "\n".join(lines)


def generate_digest_html(clusters: list[dict], target_date: str, top_n: int = 5) -> dict:
    """
    Génère le digest sous forme de dict Python consommé par le template Jinja2.
    Retourne les top_n clusters enrichis avec résumé, croissance, mots-clés et sources.
    """
    top = clusters[:top_n]
    date_fmt = datetime.fromisoformat(target_date).strftime("%d %B %Y")

    topics = []
    for i, cluster in enumerate(top, 1):
        name = _cluster_name(cluster)
        count = cluster["article_count"]
        yest = cluster.get("yesterday_count", 0)
        keywords = cluster.get("keywords", [])[:6]
        top_titles = cluster.get("top_titles", [])

        if yest > 0:
            growth_pct = int((count - yest) / yest * 100)
            growth_str = f"+{growth_pct}%" if growth_pct >= 0 else f"{growth_pct}%"
            is_new = False
        else:
            growth_str = "new"
            is_new = True

        all_text = " ".join(
            f"{a['title']} {a.get('content','')}"
            for a in cluster.get("articles", [])[:5]
        )
        summary = extractive_summary(all_text, sentences=2)

        # Normalisation des top_titles : on accepte des strings (legacy) ou des dicts
        normalized = []
        for t in top_titles[:3]:
            if isinstance(t, dict):
                normalized.append({
                    "title": t.get("title", ""),
                    "url": t.get("url", ""),
                    "source": t.get("source", ""),
                })
            else:
                normalized.append({"title": t, "url": "", "source": ""})

        topics.append({
            "rank": i,
            "name": name,
            "count": count,
            "growth": growth_str,
            "is_new": is_new,
            "score": round(cluster["trend_score"], 2),
            "keywords": keywords,
            "summary": summary,
            "top_titles": normalized,
            "articles": cluster.get("articles", []),
        })

    return {
        "date": date_fmt,
        "total_articles": sum(c["article_count"] for c in clusters),
        "total_clusters": len(clusters),
        "topics": topics,
    }
