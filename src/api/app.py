import sys
import threading
from pathlib import Path
from datetime import date
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from src.storage import db
from src.digest.generator import generate_digest_html

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

app = FastAPI(title="AI Radar")

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"


def render_template(name: str, context: dict) -> str:
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    tmpl = env.get_template(name)
    return tmpl.render(**context)


NEWSLETTER_SOURCES = {"latent_space", "import_ai", "tldr_ai"}


def _latest_newsletter_articles(target_date: str) -> list[dict]:
    """
    Retourne le dernier article disponible par source newsletter sur les 7 jours
    précédant target_date. Les newsletters hebdomadaires (Import AI) ou décalées
    d'un jour (TLDR AI) n'ont pas d'article daté exactement aujourd'hui.
    """
    from datetime import date, timedelta
    cutoff = (date.fromisoformat(target_date) - timedelta(days=7)).isoformat()

    import sqlite3
    from pathlib import Path
    db_path = Path(__file__).parent.parent.parent / "data" / "radar.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    results = []
    for source in NEWSLETTER_SOURCES:
        rows = conn.execute(
            "SELECT * FROM articles WHERE source = ? AND date <= ? AND date >= ? "
            "ORDER BY date DESC LIMIT 1",
            (source, target_date, cutoff),
        ).fetchall()
        results.extend([dict(r) for r in rows])

    conn.close()
    # Tri par source puis date décroissante pour un affichage cohérent
    results.sort(key=lambda a: (a["source"], a["date"]), reverse=True)
    return results


def _pipeline(today: str):
    from src.collector.rss_collector import collect_rss
    from src.collector.arxiv_collector import collect_arxiv
    from src.collector.semanticscholar_collector import collect_semanticscholar
    from src.collector.huggingface_collector import collect_huggingface
    from src.processor.cleaner import clean_article, is_valid_article
    from src.processor.deduplicator import deduplicate
    from src.nlp.embedder import embed_articles, get_embeddings_matrix
    from src.nlp.clusterer import cluster_articles
    from src.trends.detector import build_clusters

    articles = collect_rss() + collect_arxiv() + collect_semanticscholar() + collect_huggingface()
    articles = [clean_article(a) for a in articles]
    articles = [a for a in articles if a["title"] and is_valid_article(a)]
    articles = deduplicate(articles)

    saved = 0
    for a in articles:
        if not db.article_exists(a["id"], a["date"]):
            db.upsert_article(a)
            saved += 1

    all_today = db.get_articles_by_date(today)
    # Newsletters exclues du clustering : digest multi-sujets → clusters incohérents.
    today_articles = [a for a in all_today if a["source"] not in NEWSLETTER_SOURCES]
    if not today_articles:
        print("[auto-refresh] No articles collected for today.")
        return

    today_articles = embed_articles(today_articles)
    for a in today_articles:
        db.upsert_article(a)

    embeddings = get_embeddings_matrix(today_articles)
    today_articles = cluster_articles(today_articles, embeddings)
    for a in today_articles:
        db.update_article_cluster(a["id"], a["cluster_id"])

    clusters = build_clusters(today_articles, today)
    db.save_clusters(clusters, today)
    print(f"[auto-refresh] Done — {saved} articles saved, {len(clusters)} clusters.")


def _daily_refresh_loop():
    import time
    RUN_INTERVAL = 3600  # run every hour
    while True:
        today = date.today().isoformat()
        print(f"[auto-refresh] Running pipeline for {today}...")
        try:
            _pipeline(today)
        except Exception as e:
            print(f"[auto-refresh] Pipeline error: {e}")
        time.sleep(RUN_INTERVAL)


@app.on_event("startup")
def startup():
    db.init_db()
    threading.Thread(target=_daily_refresh_loop, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index(request: Request, d: str = None):
    target_date = d or date.today().isoformat()
    clusters = db.get_clusters_by_date(target_date)
    total_articles = db.count_articles_by_date(target_date)
    source_counts = db.count_articles_by_source(target_date)

    all_articles = db.get_articles_by_date(target_date)
    # Les newsletters paraissent souvent la veille ou de façon hebdomadaire.
    # On prend le dernier article disponible par source sur les 7 derniers jours.
    newsletter_articles = _latest_newsletter_articles(target_date)

    if clusters:
        articles_by_cluster = {}
        for a in all_articles:
            if a["source"] not in NEWSLETTER_SOURCES:
                articles_by_cluster.setdefault(a["cluster_id"], []).append(a)

        full_clusters = []
        for c in clusters:
            c["articles"] = articles_by_cluster.get(c["id"], [])
            c.setdefault("yesterday_count", 0)
            full_clusters.append(c)

        digest = generate_digest_html(full_clusters, target_date)
    else:
        digest = {"date": target_date, "total_articles": total_articles, "total_clusters": 0, "topics": []}

    html = render_template("index.html", {
        "digest": digest,
        "selected_date": target_date,
        "today": date.today().isoformat(),
        "source_counts": source_counts,
        "newsletter_articles": newsletter_articles,
    })
    return HTMLResponse(html)


@app.get("/api/digest")
def api_digest(d: str = None):
    target_date = d or date.today().isoformat()
    clusters = db.get_clusters_by_date(target_date)

    if not clusters:
        return {"date": target_date, "topics": [], "total_articles": 0}

    full_clusters = []
    articles_today = db.get_articles_by_date(target_date)
    articles_by_cluster: dict[int, list] = {}
    for a in articles_today:
        cid = a["cluster_id"]
        articles_by_cluster.setdefault(cid, []).append(a)

    for c in clusters:
        c["articles"] = articles_by_cluster.get(c["id"], [])
        c["yesterday_count"] = 0
        full_clusters.append(c)

    return generate_digest_html(full_clusters, target_date)


@app.post("/api/refresh")
def refresh():
    today = date.today().isoformat()
    try:
        from src.collector.rss_collector import collect_rss
        from src.collector.arxiv_collector import collect_arxiv
        from src.processor.cleaner import clean_article, is_valid_article
        from src.processor.deduplicator import deduplicate
        from src.nlp.embedder import embed_articles, get_embeddings_matrix
        from src.nlp.clusterer import cluster_articles
        from src.trends.detector import build_clusters

        from src.collector.semanticscholar_collector import collect_semanticscholar
        from src.collector.huggingface_collector import collect_huggingface
        articles = collect_rss() + collect_arxiv() + collect_semanticscholar() + collect_huggingface()
        articles = [clean_article(a) for a in articles]
        articles = [a for a in articles if a["title"] and is_valid_article(a)]
        articles = deduplicate(articles)

        saved = 0
        for a in articles:
            if not db.article_exists(a["id"], a["date"]):
                db.upsert_article(a)
                saved += 1

        today_articles = db.get_articles_by_date(today)
        if not today_articles:
            return JSONResponse({"ok": False, "error": "No articles for today"})

        today_articles = embed_articles(today_articles)
        for a in today_articles:
            db.upsert_article(a)

        embeddings = get_embeddings_matrix(today_articles)
        today_articles = cluster_articles(today_articles, embeddings)
        for a in today_articles:
            db.update_article_cluster(a["id"], a["cluster_id"])

        clusters = build_clusters(today_articles, today)
        db.save_clusters(clusters, today)

        return {"ok": True, "articles_saved": saved, "clusters": len(clusters)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.get("/api/status")
def status():
    today = date.today().isoformat()
    return {
        "status": "ok",
        "today": today,
        "articles_today": db.count_articles_by_date(today),
        "clusters_today": len(db.get_clusters_by_date(today)),
    }
