"""PostgreSQL access: connection pool and migrations (applied under an advisory lock, so several
replicas starting at once do not race)."""

from __future__ import annotations

import os
from importlib import resources

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

MIGRATION_LOCK = 742_9917


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "postgresql://postgres@localhost:5432/cbo")


def make_pool(url: str | None = None, min_size: int = 1, max_size: int = 10) -> ConnectionPool:
    return ConnectionPool(
        url or database_url(),
        min_size=min_size,
        max_size=max_size,
        kwargs={"row_factory": dict_row, "autocommit": False},
        open=True,
        timeout=10,
    )


def migrate(pool: ConnectionPool) -> list[str]:
    applied = []
    files = sorted(
        (p for p in resources.files("cbo.db").joinpath("migrations").iterdir() if p.name.endswith(".sql")),
        key=lambda p: p.name,
    )
    with pool.connection() as conn:
        conn.execute("select pg_advisory_lock(%s)", (MIGRATION_LOCK,))
        try:
            conn.execute(
                "create table if not exists schema_migrations (version text primary key, applied_at timestamptz not null default now())"
            )
            done = {r["version"] for r in conn.execute("select version from schema_migrations").fetchall()}
            for f in files:
                v = f.name.split("__")[0]
                if v in done:
                    continue
                conn.execute(f.read_text())
                conn.execute("insert into schema_migrations (version) values (%s)", (v,))
                applied.append(v)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.execute("select pg_advisory_unlock(%s)", (MIGRATION_LOCK,))
            conn.commit()
    return applied
