"""
db.py — Persistencia para THE TRUTH ENGINE.

Soporta dos backends, elegidos automaticamente segun el entorno:

- Si existe la variable de entorno DATABASE_URL -> usa PostgreSQL
  (pensado para Neon en produccion/Render). Los datos NO se pierden
  en cada redeploy.
- Si NO existe DATABASE_URL -> usa SQLite local (data/phishing/truth_engine.db),
  como hasta ahora. Util para desarrollo local sin depender de internet.

El resto del modulo (guardar_analisis, listar_ultimos, etc.) tiene la
misma firma sin importar el backend -- app.py no necesita cambiar.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_DB_PATH = Path("data/phishing/truth_engine.db")

DATABASE_URL = os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras


# ---------------------------------------------------------------------------
# Conexion
# ---------------------------------------------------------------------------
def get_connection(db_path: Path | str = DEFAULT_DB_PATH):
    if USE_POSTGRES:
        # cursor_factory=RealDictCursor hace que las filas se puedan leer
        # como row["columna"], igual que sqlite3.Row -- el resto del
        # codigo no necesita saber cual backend esta usando.
        return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Creacion de tabla + migracion automatica de columnas nuevas
# ---------------------------------------------------------------------------
COLUMNAS_REQUERIDAS = {
    "analizado_por_llm": "INTEGER NOT NULL DEFAULT 0",
    "llm_categoria": "TEXT",
}


def _columnas_existentes_pg(cur) -> set[str]:
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'analisis'
    """)
    return {row["column_name"] for row in cur.fetchall()}


def _columnas_existentes_sqlite(cur) -> set[str]:
    cur.execute("PRAGMA table_info(analisis)")
    return {row["name"] for row in cur.fetchall()}


def _migrar_columnas_faltantes(conn) -> None:
    cur = conn.cursor()
    existentes = _columnas_existentes_pg(cur) if USE_POSTGRES else _columnas_existentes_sqlite(cur)
    for columna, tipo_sql in COLUMNAS_REQUERIDAS.items():
        if columna not in existentes:
            print(f"[db] Migrando: agregando columna '{columna}' a la tabla analisis")
            cur.execute(f"ALTER TABLE analisis ADD COLUMN {columna} {tipo_sql}")
    conn.commit()


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        if USE_POSTGRES:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analisis (
                    id               SERIAL PRIMARY KEY,
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
                    telegram_error   TEXT,
                    analizado_por_llm INTEGER NOT NULL DEFAULT 0,
                    llm_categoria    TEXT
                )
            """)
        else:
            cur.execute("""
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
                    telegram_error   TEXT,
                    analizado_por_llm INTEGER NOT NULL DEFAULT 0,
                    llm_categoria    TEXT
                )
            """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_analisis_timestamp ON analisis(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_analisis_etiqueta ON analisis(etiqueta)")
        conn.commit()

        # Migra columnas nuevas en tablas que ya existian de antes
        # (por ejemplo, una base SQLite vieja sin analizado_por_llm).
        _migrar_columnas_faltantes(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Escritura / lectura
# ---------------------------------------------------------------------------
def _placeholder() -> str:
    """Postgres usa %s, SQLite usa ?."""
    return "%s" if USE_POSTGRES else "?"


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
    analizado_por_llm: bool = False,
    llm_categoria: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        p = _placeholder()
        query = f"""
            INSERT INTO analisis (
                timestamp, texto, canal, remitente,
                score, etiqueta, prob_modelo_ml,
                motivos, indicadores,
                telegram_enviado, telegram_error,
                analizado_por_llm, llm_categoria
            ) VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})
            {"RETURNING id" if USE_POSTGRES else ""}
        """
        params = (
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
            1 if analizado_por_llm else 0,
            llm_categoria,
        )
        cur.execute(query, params)

        if USE_POSTGRES:
            new_id = cur.fetchone()["id"]
        else:
            new_id = cur.lastrowid

        conn.commit()
        return new_id
    finally:
        conn.close()


def listar_ultimos(n: int = 20, db_path: Path | str = DEFAULT_DB_PATH) -> list[dict]:
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        p = _placeholder()
        cur.execute(
            f"""
            SELECT id, timestamp, texto, canal, remitente,
                   score, etiqueta, prob_modelo_ml,
                   motivos, indicadores,
                   telegram_enviado, telegram_error,
                   analizado_por_llm, llm_categoria
            FROM analisis
            ORDER BY id DESC
            LIMIT {p}
            """,
            (n,),
        )
        rows = cur.fetchall()
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
                "analizado_por_llm": bool(r["analizado_por_llm"]),
                "llm_categoria": r["llm_categoria"],
            })
        return result
    finally:
        conn.close()