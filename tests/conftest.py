import os
import uuid

import psycopg
import pytest


def _admin_url() -> str | None:
    return os.environ.get("DATABASE_URL")


@pytest.fixture(scope="session")
def database_url():
    """A fresh database per test session on the server in DATABASE_URL (CI: a Postgres service)."""
    base = _admin_url()
    if not base:
        pytest.skip("DATABASE_URL not set")
    name = f"cbo_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(f"create database {name}")
    url = base.replace("/cbo", f"/{name}", 1) if "/cbo" in base else base.rsplit("/", 1)[0] + f"/{name}"
    yield url
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(f"drop database if exists {name} with (force)")


@pytest.fixture(scope="session")
def fatigue_model(tmp_path_factory):
    from cbo.fatigue.train import train

    out = tmp_path_factory.mktemp("fatigue")
    report = train(str(out), n_campaigns=40, days=120, register=False)
    from cbo.fatigue.model import load_champion

    return load_champion(str(out)), report
