"""
db.py — Persistencia simple con SQLite para THE TRUTH ENGINE

Nota sobre Render (plan gratuito): el disco es efimero, cualquier
archivo SQLite se pierde en cada redeploy/reinicio. Este modulo sigue
usando SQLite porque es lo mas simple para el MVP, pero deja el punto
de entrada (get_connection) centralizado a proposito: el dia que se
conecte una base persistente real (Postgres via Supabase/Neon/Render
Postgres, usando la variable de entorno DATABASE_URL), el cambio se
hace en un solo lugar. Implementar ese soporte es un cambio de fondo
(sintaxis SQL distinta: AUTOINCREMENT vs SERIAL, placeholders %s vs ?,
requiere psycopg2/asyncpg) -- se deja como siguiente paso aparte, no
se improvisa aqui sin poder probarlo.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_DB_PATH = Path("data/phishing/truth_engine.db")

# Columnas que la tabla DEBE tener. Si falta alguna (por ejemplo, en un
# .db creado antes de agregar la capa LLM), init_db() la agrega sola
# con ALTER TABLE, sin tocar las filas existentes.
COLUMNAS_REQUERIDAS = {
    "analizado_por_llm": "INTEGER NOT NULL DEFAULT 0",
    "llm_categoria": "TEXT",
}


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _migrar_columnas_faltantes(conn: sqlite3.Connection) -> None:
    """Agrega columnas nuevas a una tabla 'analisis' ya existente, sin
    perder los datos que ya tiene. CREATE TABLE IF NOT EXISTS no hace
    nada si la tabla ya existe con un esquema viejo -- por eso hace
    falta este paso aparte."""
    columnas_actuales = {
        row["name"] for row in conn.execute("PRAGMA table_info(analisis)").fetchall()
    }
    for columna, tipo_sql in COLUMNAS_REQUERIDAS.items():
        if columna not in columnas_actuales:
            print(f"[db] Migrando: agregando columna '{columna}' a la tabla analisis")
            conn.execute(f"ALTER TABLE analisis ADD COLUMN {columna} {tipo_sql}")
    conn.commit()


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
                telegram_error   TEXT,
                analizado_por_llm INTEGER NOT NULL DEFAULT 0,
                llm_categoria    TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analisis_timestamp ON analisis(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analisis_etiqueta ON analisis(etiqueta)")
        conn.commit()

        # Si la tabla ya existia de antes (esquema viejo), esto agrega
        # las columnas que falten sin borrar las filas actuales.
        _migrar_columnas_faltantes(conn)
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
    analizado_por_llm: bool = False,
    llm_categoria: Optional[str] = None,
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
                telegram_enviado, telegram_error,
                analizado_por_llm, llm_categoria
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                1 if analizado_por_llm else 0,
                llm_categoria,
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
                   telegram_enviado, telegram_error,
                   analizado_por_llm, llm_categoria
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
                "analizado_por_llm": bool(r["analizado_por_llm"]),
                "llm_categoria": r["llm_categoria"],
            })
        return result
    finally:
        conn.close()