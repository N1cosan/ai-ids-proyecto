"""
db.py — Persistencia simple con SQLite para THE TRUTH ENGINE
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_DB_PATH = Path("data/phishing/truth_engine.db")


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analisis (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT NOT NULL,
                texto            TEXT NOT NULL,
                canal            TEXT NOT NULL,
                remitente        TEXT,
                score            REAL NOT NULL,
                etiqueta         TEXT NOT NULL,
                prob_modelo_ml   REAL,
                motivos          TEXT,
                indicadores      TEXT,
                telegram_enviado INTEGER NOT NULL DEFAULT 0,
                telegram_error   TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analisis_timestamp ON analisis(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analisis_etiqueta ON analisis(etiqueta)")
        conn.commit()
    finally:
        conn.close()


def guardar_analisis(
    *,
    texto: str,
    canal: str,
    remitente: Optional[str],
    score: float,
    etiqueta: str,
    prob_modelo_ml: Optional[float],
    motivos: list[str],
    indicadores: dict[str, Any],
    telegram_enviado: bool,
    telegram_error: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO analisis (
                timestamp, texto, canal, remitente,
                score, etiqueta, prob_modelo_ml,
                motivos, indicadores,
                telegram_enviado, telegram_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                texto,
                canal,
                remitente,
                score,
                etiqueta,
                prob_modelo_ml,
                json.dumps(motivos, ensure_ascii=False),
                json.dumps(indicadores, ensure_ascii=False),
                1 if telegram_enviado else 0,
                telegram_error,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def listar_ultimos(n: int = 20, db_path: Path | str = DEFAULT_DB_PATH) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, timestamp, texto, canal, remitente,
                   score, etiqueta, prob_modelo_ml,
                   motivos, indicadores,
                   telegram_enviado, telegram_error
            FROM analisis
            ORDER BY id DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "texto": r["texto"],
                "canal": r["canal"],
                "remitente": r["remitente"],
                "score": r["score"],
                "etiqueta": r["etiqueta"],
                "prob_modelo_ml": r["prob_modelo_ml"],
                "motivos": json.loads(r["motivos"] or "[]"),
                "indicadores": json.loads(r["indicadores"] or "{}"),
                "telegram_enviado": bool(r["telegram_enviado"]),
                "telegram_error": r["telegram_error"],
            })
        return result
    finally:
        conn.close()