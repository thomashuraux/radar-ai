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


def _pipeline(today: str):
    from src.collector.rss_collector import collect_rss
    from src.collector.arxiv_collector import collect_arxiv
    from src.collector.semanticscholar_collector import collect_semanticscholar
    from src.processor.cleaner import clean_article, is_valid_article
    from src.processor.deduplicator import deduplicate
    from src.nlp.embedder import embed_articles, get_embeddings_matrix
    from src.nlp.clusterer import cluster_articles
    from src.trends.detector import build_clusters

    articles = collect_rss() + collect_arxiv() + collect_semanticscholar()
    articles = [clean_article(a) for a in articles]
    articles = [a for a in articles if a["title"] and is_valid_article(a)]
    articles = deduplicate(articles)

    saved = 0
    for a in articles:
        if not db.article_exists(a["id"]):
            db.upsert_article(a)
            saved += 1

    today_articles = db.get_articles_by_date(today)
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

    if clusters:
        full_clusters = []
        for c in clusters:
            articles = [
                a for a in db.get_articles_by_date(target_date)
                if a["cluster_id"] == c["id"]
            ]
            c["articles"] = articles
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
        articles = collect_rss() + collect_arxiv() + collect_semanticscholar()
        articles = [clean_article(a) for a in articles]
        articles = [a for a in articles if a["title"] and is_valid_article(a)]
        articles = deduplicate(articles)

        saved = 0
        for a in articles:
            if not db.article_exists(a["id"]):
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
