"""Disk cache for upstream HTTP responses, keyed by method + URL + params.

Cached responses make eval runs repeatable and keep us under
ClinicalTrials.gov's ~50 requests/min limit.
"""
import hashlib
import json
from pathlib import Path
from typing import Any


def _key(url: str, params: dict[str, Any]) -> str:
    raw = url + "?" + json.dumps(params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class DiskCache:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, url: str, params: dict[str, Any]) -> Any | None:
        path = self.dir / f"{_key(url, params)}.json"
        if path.exists():
            return json.loads(path.read_text())["response"]
        return None

    def put(self, url: str, params: dict[str, Any], response: Any) -> None:
        path = self.dir / f"{_key(url, params)}.json"
        path.write_text(json.dumps({"url": url, "params": params, "response": response}))
