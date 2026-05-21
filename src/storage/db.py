import sqlite3
import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional

# Base de données SQLite stockée dans data/radar.db à la racine du projet.
# Path(__file__) remonte depuis src/storage/ jusqu'à la racine avec .parent.parent.parent
DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "radar.db"


def get_conn() -> sqlite3.Connection:
    # resolve() garantit un chemin absolu même si __file__ est relatif (ex: launchd)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # row_factory = sqlite3.Row : les résultats sont accessibles par nom de colonne
    # (ex: row["title"]) plutôt que par index numérique (row[2])
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        # WAL (Write-Ahead Logging) : les lectures et écritures peuvent se faire
        # en parallèle sans se bloquer. Idéal pour un serveur web + cron simultanés.
        conn.executescript("""
            PRAGMA journal_mode=WAL;
        """)

        # Migration live : si la colonne "name" n'existe pas encore dans clusters
        # (base créée avant qu'on l'ajoute), on l'ajoute sans perdre les données.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(clusters)").fetchall()}
        if cols and "name" not in cols:
            conn.execute("ALTER TABLE clusters ADD COLUMN name TEXT")
        if cols and "yesterday_count" not in cols:
            conn.execute("ALTER TABLE clusters ADD COLUMN yesterday_count INTEGER DEFAULT 0")

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,          -- MD5 de source:url
                source TEXT NOT NULL,         -- "arxiv", "techcrunch", "reddit_ml"...
                title TEXT NOT NULL,
                content TEXT,                 -- résumé ou corps de l'article
                url TEXT,
                date TEXT NOT NULL,           -- format YYYY-MM-DD
                embedding TEXT,               -- vecteur JSON (liste de floats)
                cluster_id INTEGER DEFAULT -1, -- -1 = non assigné (bruit HDBSCAN)
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER,
                date TEXT NOT NULL,
                name TEXT,                    -- titre de l'article centroïde
                keywords TEXT NOT NULL,       -- JSON : liste de mots-clés TF-IDF
                trend_score REAL DEFAULT 0,
                article_count INTEGER DEFAULT 0,
                yesterday_count INTEGER DEFAULT 0,
                summary TEXT,
                top_titles TEXT,              -- JSON : liste de {title, url, source}
                PRIMARY KEY (id, date)        -- un cluster peut exister sur plusieurs jours
            );

            -- Index sur date pour les requêtes "articles du jour" (très fréquentes)
            CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(date);
            -- Index sur cluster_id pour reconstituer les articles d'un cluster
            CREATE INDEX IF NOT EXISTS idx_articles_cluster ON articles(cluster_id);
        """)


def upsert_article(article: dict):
    # INSERT OR REPLACE : si l'article existe déjà (même id), on le remplace.
    # Utilisé pour mettre à jour l'embedding et le cluster_id après analyse.
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO articles
                (id, source, title, content, url, date, embedding, cluster_id)
            VALUES
                (:id, :source, :title, :content, :url, :date,
                 :embedding, :cluster_id)
        """, {
            "id": article["id"],
            "source": article["source"],
            "title": article["title"],
            "content": article.get("content", ""),
            "url": article.get("url", ""),
            "date": article["date"],
            # L'embedding est stocké comme JSON string car SQLite n'a pas de type ARRAY
            "embedding": json.dumps(article["embedding"]) if article.get("embedding") else None,
            "cluster_id": article.get("cluster_id", -1),
        })


def get_articles_by_date(target_date: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM articles WHERE date = ?", (target_date,)
        ).fetchall()
    return [dict(r) for r in rows]


def update_article_cluster(article_id: str, cluster_id: int):
    # Mise à jour ciblée après clustering — plus efficace qu'un upsert complet
    with get_conn() as conn:
        conn.execute(
            "UPDATE articles SET cluster_id = ? WHERE id = ?",
            (cluster_id, article_id),
        )


def save_clusters(clusters: list[dict], target_date: str):
    # On supprime et recrée les clusters du jour à chaque run.
    # Plus simple que de gérer des UPDATE/INSERT différenciés.
    with get_conn() as conn:
        conn.execute("DELETE FROM clusters WHERE date = ?", (target_date,))
        for c in clusters:
            conn.execute("""
                INSERT INTO clusters
                    (id, date, name, keywords, trend_score, article_count, yesterday_count, summary, top_titles)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                c["id"],
                target_date,
                c.get("name", ""),
                json.dumps(c["keywords"]),
                c.get("trend_score", 0),
                c.get("article_count", 0),
                c.get("yesterday_count", 0),
                c.get("summary", ""),
                json.dumps(c.get("top_titles", [])),
            ))


def get_clusters_by_date(target_date: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM clusters WHERE date = ? ORDER BY trend_score DESC",
            (target_date,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        # On désérialise les champs JSON au retour
        d["keywords"] = json.loads(d["keywords"])
        d["top_titles"] = json.loads(d["top_titles"])
        result.append(d)
    return result


def count_articles_by_date(target_date: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as n FROM articles WHERE date = ?", (target_date,)
        ).fetchone()
    return row["n"]


def count_articles_by_source(target_date: str) -> dict:
    # Retourne un dict {source: count} pour afficher la répartition dans la stats bar
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT source, COUNT(*) as n FROM articles WHERE date = ? GROUP BY source",
            (target_date,),
        ).fetchall()
    return {r["source"]: r["n"] for r in rows}


def article_exists(article_id: str) -> bool:
    # Vérification rapide avant upsert pour compter les "vrais" nouveaux articles
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM articles WHERE id = ?", (article_id,)
        ).fetchone()
    return row is not None
