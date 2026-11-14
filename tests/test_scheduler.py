"""Scheduler: a REAL hard kill mid-run, then a restart that resumes it (F8)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from argus.storage import db
from argus.storage.lease import pid_alive

from tests.test_graph_crash import PAGES, URL, _descendants

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


def _setup(tmp_path):
    cfg = tmp_path / "config"
    cfg.mkdir()
    shutil.copy(REPO / "config" / "models.yaml", cfg / "models.yaml")
    shutil.copy(REPO / "config" / "routing.yaml", cfg / "routing.yaml")
    target = {"id": "uni", "name": "Uni Master of AI", "url": URL, "frequency": "weekly",
              "extract": ["intake_dates", "duration", "fees", "entry_requirements"], "notify": False}
    (cfg / "targets.yaml").write_text(yaml.safe_dump({"targets": [target]}), encoding="utf-8")
    (tmp_path / "pages.json").write_text(json.dumps(PAGES), encoding="utf-8")
    env = dict(os.environ, ARGUS_CONFIG_DIR=str(cfg))
    args = [sys.executable, "-m", "argus.cli", "--db", str(tmp_path / "argus.db"), "scheduler",
            "--interval-s", "3600", "--fake-backend", "--fake-embed", "--delay", "0.4",
            "--pages", str(tmp_path / "pages.json"), "--models", "model-a:latest", "model-b:latest"]
    return env, args, tmp_path / "argus.db", target


def _q(path, sql, *p):
    with db.connect(path) as conn:
        return [dict(r) for r in conn.execute(sql, p)]


def _wait(pred, timeout, what):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if pred():
                return
        except Exception:  # noqa: BLE001 - db not created yet
            pass
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


def test_hard_kill_mid_run_then_restart_resumes_it(tmp_path):
    env, args, db_path, target = _setup(tmp_path)
    proc = subprocess.Popen(args, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _wait(lambda: _q(db_path, "SELECT 1 FROM graph_nodes WHERE node_type='critique' AND status='running'"),
              120, "the first run to reach the critic")
        first_run = _q(db_path, "SELECT run_id FROM runs WHERE target_id='uni' ORDER BY started_at LIMIT 1")[0]["run_id"]
        tree = _descendants(proc.pid)
    finally:
        proc.kill()
        proc.wait(timeout=30)
    deadline = time.time() + 20
    while time.time() < deadline and any(pid_alive(p) for p in tree):
        time.sleep(0.3)
    assert not [p for p in tree if pid_alive(p)], "the scheduler left a process behind"
    assert _q(db_path, "SELECT status FROM graphs WHERE run_id=?", first_run)[0]["status"] == "running"

    proc2 = subprocess.Popen(args, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _wait(lambda: _q(db_path, "SELECT status FROM graphs WHERE run_id=?", first_run)[0]["status"] == "done",
              180, "the interrupted run to be resumed and finished")
    finally:
        proc2.kill()
        proc2.wait(timeout=30)

    current = {r["entity_key"] for r in _q(db_path, "SELECT entity_key FROM facts WHERE is_current=1")}
    assert current == set(target["extract"])
    dup = _q(db_path, "SELECT target_id, entity_key, version, COUNT(*) AS n FROM facts "
                      "GROUP BY target_id, entity_key, version HAVING n > 1")
    assert dup == []
    events = _q(db_path, "SELECT detail FROM node_events e JOIN graphs g ON g.graph_id = e.graph_id "
                         "WHERE g.run_id = ? AND detail LIKE '%resume%'", first_run)
    assert events, "the critic that was in flight went back to ready on resume"
