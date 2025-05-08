"""neo4j_connection.py

Utility helpers for reading Neo4j connection details from a `.env` file and
returning a ready‑to‑use Neo4j Python driver.

Example .env file (place at your project root):

    # .env
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=changeme

Install dependencies (already handled in requirements.txt):
    pip install neo4j python-dotenv
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth, Driver

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------

# By default, look for `.env` two directories up from this file (project root)
# but allow overriding via the DOTENV_PATH environment variable.
_default_env_path = Path(os.getenv("DOTENV_PATH", Path(__file__).resolve().parent.parent / ".env"))

# `override=False` ensures existing environment variables take precedence.
load_dotenv(dotenv_path=_default_env_path, override=False)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_neo4j_credentials() -> Tuple[str, str, str]:
    """Return (uri, user, password) fetched from the environment.

    Raises
    ------
    EnvironmentError
        If any of the required variables are missing.
    """
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")

    if not all([uri, user, password]):
        raise EnvironmentError(
            "Missing one or more Neo4j environment variables. "
            "Expected NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD in your .env file."
        )
    return uri, user, password


def get_driver(encrypted: bool | None = None, **kwargs) -> Driver:
    """Create and return a `neo4j.Driver` instance.

    Parameters
    ----------
    encrypted : bool | None, optional
        Whether to force encryption. If *None* (default), Neo4j chooses based on the URI scheme.
    **kwargs
        Extra keyword arguments forwarded to `GraphDatabase.driver` (e.g. `max_connection_lifetime`).

    Returns
    -------
    neo4j.Driver
    """
    uri, user, password = get_neo4j_credentials()
    return GraphDatabase.driver(uri, auth=basic_auth(user, password), encrypted=encrypted, **kwargs)


# ---------------------------------------------------------------------------
# CLI quick‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        with get_driver() as driver:
            driver.verify_connectivity()
            print("✅ Successfully connected to Neo4j at", driver.target)
    except Exception as exc:
        print("❌ Connection failed:", exc)