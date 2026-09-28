"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._inputs: dict[str, str] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id/user_id."""
        req_id = request_id or f"{user_id}_{utc_now_iso()}"
        self._open[req_id] = time.time()
        self._inputs[req_id] = text

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        req_id = request_id or f"{user_id}_{utc_now_iso()}"
        start_time = self._open.pop(req_id, time.time())
        latency_ms = (time.time() - start_time) * 1000
        input_text = self._inputs.pop(req_id, "")

        entry = {
            "timestamp": utc_now_iso(),
            "user_id": user_id,
            "request_id": req_id,
            "input": input_text[:500] if input_text else "",
            "output": text[:500] if text else "",
            "blocked": blocked,
            "blocking_layer": layer,
            "latency_ms": round(latency_ms, 2),
        }
        self.logs.append(entry)

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
