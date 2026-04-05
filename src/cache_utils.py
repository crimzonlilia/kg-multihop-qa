"""Helpers for working with named triples caches."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

CACHE_DIR = Path("data/cache")
PROCESSED_DIR = Path("data/processed")
DEFAULT_TRIPLES_CACHE = CACHE_DIR / "triples.json"
DEFAULT_GRAPH_PATH = PROCESSED_DIR / "kg.pkl"


def normalize_triples_cache_name(cache_name: str | int | None) -> str | None:
    """Normalize cache aliases such as 'full' or '300'."""
    if cache_name is None:
        return None

    raw = str(cache_name).strip()
    if not raw:
        return None

    lowered = raw.lower()
    if lowered in {"default", "latest", "triples"}:
        return None
    if lowered in {"all", "full", "none"}:
        return "full"

    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_")
    return normalized or None


def infer_triples_cache_name(sample_limit: int | str | None) -> str:
    """Infer a readable cache label from a sample limit."""
    if sample_limit in (None, "", "all", "full"):
        return "full"

    try:
        return str(int(sample_limit))
    except (TypeError, ValueError):
        return normalize_triples_cache_name(sample_limit) or "full"


def infer_sample_limit_from_cache_name(cache_name: str | int | None) -> int | None:
    """Infer a sample limit from a cache label like '500'; return None for 'full'."""
    normalized = normalize_triples_cache_name(cache_name)
    if normalized in (None, "full"):
        return None

    if str(normalized).isdigit():
        return int(normalized)

    return None


def build_triples_cache_path(cache_name: str | int | None = None) -> Path:
    """Return the file path for a named triples cache."""
    normalized = normalize_triples_cache_name(cache_name)
    if normalized is None:
        return DEFAULT_TRIPLES_CACHE
    return CACHE_DIR / f"triples_{normalized}.json"


def build_graph_output_path(graph_name: str | int | None = None) -> Path:
    """Return the file path for a named graph output."""
    normalized = normalize_triples_cache_name(graph_name)
    if normalized is None:
        return DEFAULT_GRAPH_PATH
    return PROCESSED_DIR / f"kg_{normalized}.pkl"


def graph_backup_path_for(path: str | os.PathLike) -> Path:
    """Return a backup path next to a graph pickle file."""
    graph_path = Path(path)
    return graph_path.with_suffix(graph_path.suffix + ".backup")


def resolve_graph_output_path(
    graph_name: str | int | None = None,
    graph_path: str | os.PathLike | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve either an explicit path or a named graph file like '500'/'full'."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if graph_path:
        path = Path(graph_path)
    else:
        env_path = os.getenv("KG_GRAPH_PATH")
        if env_path and graph_name is None:
            path = Path(env_path)
        else:
            chosen_name = normalize_triples_cache_name(
                graph_name if graph_name is not None else os.getenv("KG_GRAPH_NAME")
            )

            candidates: list[Path] = []
            if chosen_name is not None:
                candidates.append(build_graph_output_path(chosen_name))
                candidates.append(DEFAULT_GRAPH_PATH)
            else:
                candidates.append(DEFAULT_GRAPH_PATH)
                full_graph = build_graph_output_path("full")
                if full_graph not in candidates:
                    candidates.append(full_graph)

            if must_exist:
                path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
            else:
                path = candidates[0]

    if must_exist and not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    return path


def resolve_triples_cache_path(
    cache_name: str | int | None = None,
    cache_path: str | os.PathLike | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve either an explicit path or a named cache like 'full'/'300'."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path:
        path = Path(cache_path)
    else:
        env_path = os.getenv("TRIPLES_CACHE_PATH")
        if env_path and cache_name is None:
            path = Path(env_path)
        else:
            chosen_name = normalize_triples_cache_name(
                cache_name if cache_name is not None else os.getenv("TRIPLES_CACHE_NAME")
            )

            candidates: list[Path] = []
            if chosen_name is not None:
                candidates.append(build_triples_cache_path(chosen_name))
                candidates.append(DEFAULT_TRIPLES_CACHE)
            else:
                candidates.append(DEFAULT_TRIPLES_CACHE)
                full_cache = build_triples_cache_path("full")
                if full_cache not in candidates:
                    candidates.append(full_cache)

            if must_exist:
                path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
            else:
                path = candidates[0]

    if must_exist and not path.exists():
        raise FileNotFoundError(f"Triples cache not found: {path}")

    return path


def load_triples_cache(
    cache_path: str | os.PathLike | None = None,
    cache_name: str | int | None = None,
) -> tuple[list[dict], dict, Path]:
    """Load triples cache and unwrap either payload or legacy list format."""
    resolved_path = resolve_triples_cache_path(
        cache_name=cache_name,
        cache_path=cache_path,
        must_exist=True,
    )

    with open(resolved_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "results" in payload:
        return payload.get("results", []), payload.get("meta", {}), resolved_path

    if isinstance(payload, list):
        return payload, {}, resolved_path

    raise ValueError(f"Unsupported triples cache format in {resolved_path}")
