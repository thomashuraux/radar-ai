#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Radar — CLI entry point

Usage:
  python main.py collect          # collect articles from all sources
  python main.py analyze          # embed + cluster + detect trends
  python main.py digest           # print today's digest to terminal
  python main.py run              # full pipeline (collect + analyze + digest)
  python main.py serve            # start web UI on http://localhost:8000
  python main.py collect --date 2026-04-15   # collect for a specific date
"""
import sys
import argparse
from datetime import date


def cmd_collect(target_date: str, verbose: bool = False):
    from src.collector.rss_collector import collect_rss
    from src.collector.arxiv_collector import collect_arxiv
    from src.processor.cleaner import clean_article
    from src.processor.deduplicator import deduplicate
    from src.storage import db

    db.init_db()
    print(f"[collect] Fetching RSS feeds...")
    rss = collect_rss()
    print(f"[collect] RSS: {len(rss)} articles")

    print(f"[collect] Fetching arXiv...")
    arxiv = collect_arxiv()
    print(f"[collect] arXiv: {len(arxiv)} articles")

    from src.collector.semanticscholar_collector import collect_semanticscholar
    print(f"[collect] Fetching Semantic Scholar...")
    ss = collect_semanticscholar()
    print(f"[collect] Semantic Scholar: {len(ss)} articles")

    from src.collector.huggingface_collector import collect_huggingface
    print(f"[collect] Fetching HuggingFace Daily Papers...")
    hf = collect_huggingface()
    print(f"[collect] HuggingFace: {len(hf)} articles")

    all_articles = rss + arxiv + ss + hf
    print(f"[collect] Total before dedup: {len(all_articles)}")

    all_articles = [clean_article(a) for a in all_articles]
    from src.processor.cleaner import is_valid_article
    all_articles = [a for a in all_articles if a["title"] and is_valid_article(a)]

    all_articles = deduplicate(all_articles)
    print(f"[collect] After dedup: {len(all_articles)}")

    today_articles = [a for a in all_articles if a["date"] == target_date]
    print(f"[collect] Articles dated {target_date}: {len(today_articles)}")

    saved = 0
    for a in all_articles:
        if not db.article_exists(a["id"], a["date"]):
            db.upsert_article(a)
            saved += 1

    print(f"[collect] Saved {saved} new articles to DB")


def cmd_analyze(target_date: str):
    import json
    import numpy as np
    from src.storage import db
    from src.nlp.embedder import embed_articles, get_embeddings_matrix
    from src.nlp.clusterer import cluster_articles
    from src.trends.detector import build_clusters

    # Les newsletters (digest quotidiens, éditoriaux) ne doivent pas entrer
    # dans le clustering : leur contenu est un résumé multi-sujets qui crée
    # des clusters incohérents. Elles sont affichées séparément dans l'UI.
    NEWSLETTER_SOURCES = {"latent_space", "import_ai", "tldr_ai"}

    db.init_db()
    all_articles = db.get_articles_by_date(target_date)
    articles = [a for a in all_articles if a["source"] not in NEWSLETTER_SOURCES]
    print(f"[analyze] {len(articles)} articles for {target_date} ({len(all_articles) - len(articles)} newsletters excluded)")

    if not articles:
        print("[analyze] No articles found. Run 'collect' first.")
        return

    # Always re-embed all articles together: TF-IDF SVD is fit on the full
    # corpus each run, so mixing stored embeddings (different SVD fit) with
    # new ones produces inhomogeneous shapes.
    print(f"[analyze] Computing embeddings for {len(articles)} articles...")
    articles = embed_articles(articles)
    for a in articles:
        db.upsert_article(a)
    print("[analyze] Embeddings saved.")

    all_articles = articles
    embeddings = get_embeddings_matrix(all_articles)

    print(f"[analyze] Clustering {len(all_articles)} articles...")
    all_articles = cluster_articles(all_articles, embeddings)

    for a in all_articles:
        db.update_article_cluster(a["id"], a["cluster_id"])

    print("[analyze] Building trend scores...")
    clusters = build_clusters(all_articles, target_date)
    print(f"[analyze] {len(clusters)} clusters found")

    db.save_clusters(clusters, target_date)
    print(f"[analyze] Saved clusters.")

    for c in clusters[:5]:
        print(f"  #{c['id']} {c['name']} — {c['article_count']} articles, score {c['trend_score']:.2f}")


def cmd_digest(target_date: str):
    from src.storage import db
    from src.trends.detector import build_clusters
    from src.digest.generator import generate_digest

    db.init_db()
    clusters = db.get_clusters_by_date(target_date)

    if not clusters:
        print(f"No clusters for {target_date}. Run 'analyze' first.")
        return

    articles_today = db.get_articles_by_date(target_date)
    by_cluster: dict[int, list] = {}
    for a in articles_today:
        by_cluster.setdefault(a["cluster_id"], []).append(a)

    full_clusters = []
    for c in clusters:
        c["articles"] = by_cluster.get(c["id"], [])
        c["yesterday_count"] = 0
        full_clusters.append(c)

    print(generate_digest(full_clusters, target_date))


def cmd_serve(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    print(f"[serve] Starting AI Radar on http://{host}:{port}")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(description="AI Radar")
    parser.add_argument("command", choices=["collect", "analyze", "digest", "run", "serve"])
    parser.add_argument("--date", default=date.today().isoformat(), help="Target date (YYYY-MM-DD)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "collect":
        cmd_collect(args.date)
    elif args.command == "analyze":
        cmd_analyze(args.date)
    elif args.command == "digest":
        cmd_digest(args.date)
    elif args.command == "run":
        cmd_collect(args.date)
        cmd_analyze(args.date)
        cmd_digest(args.date)
    elif args.command == "serve":
        cmd_serve(args.host, args.port)


if __name__ == "__main__":
    main()
