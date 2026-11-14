"""Graph engine crash-resume with REAL processes and REAL kills (Phase E 7).

Each scenario starts the engine in a subprocess (fake model, fake embeddings,
static pages: no Ollama), makes the fake model hang inside one call, waits
until the database shows that node `running`, kills the process
(TerminateProcess on Windows), then resumes in a fresh process and drives the
run to completion (approve the action, resume again).

What must hold after every kill (ARGUS_BUILD 6.8):
  - every node that was `done` before the kill has byte-identical output and
    no new transitions afterwards (outputs reused, never regenerated)
  - a routing decision frozen before the kill is the one used after it
  - stored samples are reused: the cross_model's first sample is not re-called
  - facts match a run that never crashed: no duplicates, no phantom versions
  - exactly one approval and exactly one executed action (one outbox file)
  - the killed process is gone and left no child process behind
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from argus.storage import db

REPO = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8765/uni/master-ai"
TARGET = {"id": "uni", "name": "Uni Master of AI", "url": URL,
          "extract": ["intake_dates", "duration", "fees", "entry_requirements"], "notify": True}
PAGES = {URL: {"title": "Master of AI", "text": (
    "Master of Artificial Intelligence. Duration: 2 years full-time. "
    "Intakes: February and July each year. The annual tuition fee for 2027 is AUD 45,000. "
    "Entry requires a bachelor degree with a WAM of 65.")}}

pytestmark = pytest.mark.slow


def _env():
    # Inherit PYTHONPATH: scripts/mutation_check.py points it at a mutated
    # copy of src/, and the subprocess must run THAT code or every mutant
    # "survives" for the wrong reason (found the hard way in Phase E).
    return dict(os.environ)


class Case:
    def __init__(self, root: Path, name: str):
        self.dir = root / name
        self.dir.mkdir()
        self.db = self.dir / "argus.db"
        self.outbox = self.dir / "outbox"
        (self.dir / "target.json").write_text(json.dumps(TARGET), encoding="utf-8")
        (self.dir / "pages.json").write_text(json.dumps(PAGES), encoding="utf-8")
        self.run_id = f"run-{name}"

    def args(self, *extra):
        return [sys.executable, "-m", "argus.graph.runner", "--db", str(self.db),
                "--pages", str(self.dir / "pages.json"), "--outbox", str(self.outbox),
                "--fake-backend", "--fake-embed", "--models", "model-a:latest", "model-b:latest",
                *extra]

    def start(self, hang=None):
        extra = ["--target-json", str(self.dir / "target.json"), "--run-id", self.run_id]
        if hang and hang.startswith("remember:"):
            extra += ["--hang-remember-after", hang.split(":")[1]]
        elif hang:
            extra += ["--hang", hang]
        return subprocess.Popen(self.args(*extra), cwd=REPO, env=_env(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def resume(self) -> int:
        p = subprocess.run(self.args("--resume", self.run_id), cwd=REPO, env=_env(),
                           capture_output=True, text=True, timeout=180)
        assert p.returncode in (0, 4), p.stderr[-3000:]
        return p.returncode

    def q(self, sql, *params):
        with db.connect(self.db) as conn:
            return [dict(r) for r in conn.execute(sql, params)]

    def approve(self):
        with db.connect(self.db) as conn, db.atomic(conn):
            conn.execute("UPDATE approvals SET status='approved', decided_by='test' WHERE status='pending'")

    def finish(self):
        assert self.resume() == 4
        self.approve()
        assert self.resume() == 0


def _wait_running(case: Case, node_type: str, proc, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"process exited early: {proc.stderr.read()[-2000:]}")
        if case.db.exists():
            try:
                rows = case.q("SELECT 1 FROM graph_nodes WHERE node_type = ? AND status = 'running'", node_type)
            except Exception:  # noqa: BLE001 - schema still being created
                rows = []
            if rows:
                time.sleep(0.3)  # let the hang begin inside the call
                return
        time.sleep(0.05)
    raise AssertionError(f"{node_type} never reached running")


def _children_of(pid: int):
    if sys.platform != "win32":
        return []
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}' | Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True, timeout=60)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def _facts(case):
    return sorted((r["entity_key"], r["content"], r["version"], r["change_kind"], r["is_current"])
                  for r in case.q("SELECT * FROM facts"))


@pytest.fixture(scope="module")
def control(tmp_path_factory):
    case = Case(tmp_path_factory.mktemp("crash"), "control")
    p = case.start()
    out, err = p.communicate(timeout=180)
    assert p.returncode == 4, err[-3000:]
    case.approve()
    assert case.resume() == 0
    return case


@pytest.mark.parametrize("hang", ["plan:0", "extract:0", "critique:0", "propose_action:1", "summarize:0",
                                  "write_fact:0", "remember:2"])
def test_kill_mid_call_then_resume_matches_a_run_that_never_crashed(tmp_path, control, hang):
    case = Case(tmp_path, hang.replace(":", "_"))
    role = hang.split(":")[0]
    proc = case.start(hang=hang)
    _wait_running(case, role, proc)
    children = _children_of(proc.pid)

    before_done = {r["node_id"]: r for r in case.q(
        "SELECT node_id, output, (SELECT MAX(seq) FROM node_events e WHERE e.node_id = n.node_id) AS seq "
        "FROM graph_nodes n WHERE status = 'done'")}
    frozen = {r["node_id"]: r["decision_id"] for r in case.q(
        "SELECT node_id, decision_id FROM routing_decisions WHERE frozen = 1 AND superseded = 0")}
    samples_before = case.q("SELECT node_id, sample_index, output FROM node_samples")

    proc.kill()
    proc.wait(timeout=30)
    assert proc.poll() is not None
    for child in children:
        assert not _children_of(child) and child not in _children_of(os.getppid()), "orphan left behind"

    case.finish()

    after = {r["node_id"]: r for r in case.q(
        "SELECT node_id, output, (SELECT MAX(seq) FROM node_events e WHERE e.node_id = n.node_id) AS seq "
        "FROM graph_nodes n")}
    for node_id, row in before_done.items():
        assert after[node_id]["output"] == row["output"], f"{node_id} output changed after resume"
        assert after[node_id]["seq"] == row["seq"], f"{node_id} transitioned again after resume"
    live = {r["node_id"]: r["decision_id"] for r in case.q(
        "SELECT node_id, decision_id FROM routing_decisions WHERE superseded = 0")}
    for node_id, decision_id in frozen.items():
        assert live[node_id] == decision_id, "a frozen decision was recomputed on resume"
    for s in samples_before:
        calls = case.q("SELECT COUNT(*) AS n FROM llm_calls WHERE node_id = ? AND sample_index = ?",
                       s["node_id"], s["sample_index"])[0]["n"]
        assert calls == 1, "a stored sample was regenerated"

    assert _facts(case) == _facts(control)
    # The remember node's results must say what a never-crashed run says:
    # a fact written just before the kill is still reported 'added' (the
    # receipt replays it), not 'unchanged' (audit H3).
    def kinds(c):
        rows = c.q("SELECT output FROM graph_nodes WHERE node_type = 'remember' AND status = 'done'")
        return sorted((r["entity_key"], r["change_kind"]) for row in rows
                      for r in json.loads(row["output"])["stored"]["results"])
    assert kinds(case) == kinds(control)
    approvals = case.q("SELECT status FROM approvals")
    assert [a["status"] for a in approvals] == ["executed"]
    assert len(list(case.outbox.iterdir())) == 1
    for node_id in after:
        seqs = [r["seq"] for r in case.q("SELECT seq FROM node_events WHERE node_id = ? ORDER BY seq", node_id)]
        assert seqs == list(range(1, len(seqs) + 1))
    assert case.q("SELECT status FROM graphs")[0]["status"] == "done"


@pytest.mark.parametrize("when", ["after_claim", "after_effect"])
def test_kill_during_the_approved_action_executes_it_exactly_once(tmp_path, when):
    """The external-effect window: approval claimed (and maybe the outbox
    file already written), process killed before the node's done lands."""
    case = Case(tmp_path, f"exec_{when}")
    p = case.start()
    p.communicate(timeout=180)
    assert p.returncode == 4
    case.approve()
    proc = subprocess.Popen(case.args("--resume", case.run_id, "--hang-execute", when), cwd=REPO,
                            env=_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _wait_running(case, "execute_action", proc)
    time.sleep(0.5)
    proc.kill()
    proc.wait(timeout=30)
    claimed = case.q("SELECT status, result FROM approvals")[0]
    assert claimed["status"] == "executed" and json.loads(claimed["result"])["state"] == "claimed"
    assert case.resume() == 0
    assert len(list(case.outbox.iterdir())) == 1
    final = json.loads(case.q("SELECT result FROM approvals")[0]["result"])
    assert final["state"] == "done"
    assert final["duplicate"] is (when == "after_effect")


def test_a_second_resume_while_the_first_is_alive_is_refused(tmp_path):
    case = Case(tmp_path, "lease")
    proc = case.start(hang="extract:0")
    try:
        _wait_running(case, "extract", proc)
        p = subprocess.run(case.args("--resume", case.run_id), cwd=REPO, env=_env(),
                           capture_output=True, text=True, timeout=120)
        assert p.returncode == 3, (p.returncode, p.stderr[-2000:])
    finally:
        proc.kill()
        proc.wait(timeout=30)
