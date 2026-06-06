import hashlib
import io
import feedparser
import requests
from datetime import datetime, timezone
from dateutil import parser as dateparser

# feedparser seul est souvent bloqué (User-Agent "python-requests" filtré par les sites).
# On fetch le contenu avec requests (User-Agent réaliste) puis on le passe à feedparser.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AIRadar/1.0; +https://github.com/local/radarai)"
}

RSS_FEEDS = {
    "techcrunch":      "https://techcrunch.com/category/artificial-intelligence/feed/",
    "venturebeat":     "https://venturebeat.com/category/ai/feed/",
    "mit_tech_review": "https://www.technologyreview.com/feed/",
    "the_verge_ai":    "https://www.theverge.com/rss/index.xml",
    "wired_ai":        "https://www.wired.com/feed/tag/ai/latest/rss",
    "reddit_ml":       "https://www.reddit.com/r/MachineLearning/.rss",
    "reddit_localllama": "https://www.reddit.com/r/LocalLLaMA/.rss",
    "reddit_artificial": "https://www.reddit.com/r/artificial/.rss",
    # HN filtré par mots-clés AI/LLM via hnrss.org (service tiers gratuit)
    "hackernews":      "https://hnrss.org/frontpage?q=AI+LLM+machine+learning",
    # Newsletters techniques
    "latent_space":    "https://www.latent.space/feed",
    "import_ai":       "https://jack-clark.net/feed/",
    "tldr_ai":         "https://tldr.tech/api/rss/ai",
}

# Reddit représente 3 feeds × 50 = 150 articles potentiels, ce qui noie
# les sources de qualité. On le limite à ~40 articles au total.
_SOURCE_LIMITS: dict[str, int] = {
    "reddit_ml":         15,
    "reddit_localllama": 15,
    "reddit_artificial": 10,
    "tldr_ai":           20,  # résumés très courts, on prend les 20 plus récents
}


def _parse_date(entry) -> str:
    # Les flux RSS utilisent des formats de date hétérogènes (RFC 2822, ISO 8601...).
    # dateutil.parser gère tous ces formats. On essaie plusieurs champs dans l'ordre.
    for field in ("published", "updated", "created"):
        val = entry.get(field)
        if val:
            try:
                dt = dateparser.parse(val)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    # Fallback : si aucune date trouvée, on prend aujourd'hui
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _make_id(source: str, url: str, title: str) -> str:
    # L'URL est le meilleur identifiant unique d'un article.
    # Si elle manque, on utilise le titre comme fallback.
    raw = f"{source}:{url or title}"
    return hashlib.md5(raw.encode()).hexdigest()


def collect_rss(max_per_feed: int = 50) -> list[dict]:
    articles = []
    for source, url in RSS_FEEDS.items():
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            # On passe resp.content (bytes) à feedparser pour éviter les problèmes
            # d'encodage que feedparser rencontrerait en fetching lui-même
            feed = feedparser.parse(resp.content)

            # Appliquer la limite par source (Reddit réduit, autres = max_per_feed)
            limit = _SOURCE_LIMITS.get(source, max_per_feed)

            for entry in feed.entries[:limit]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                summary = entry.get("summary", "") or entry.get("description", "")

                # Certains flux fournissent le contenu complet dans "content" (liste)
                # et un résumé court dans "summary". On prend le plus complet.
                content_list = entry.get("content", [])
                content = content_list[0].get("value", "") if content_list else summary

                if not title:
                    continue

                articles.append({
                    "id": _make_id(source, link, title),
                    "source": source,
                    "title": title,
                    "content": content or summary,
                    "url": link,
                    "date": _parse_date(entry),
                    "embedding": None,
                    "cluster_id": -1,
                })
        except Exception as e:
            print(f"[RSS] Error fetching {source}: {e}")

    return articles
