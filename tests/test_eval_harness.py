"""Evaluation harness self-tests (fake model, no Ollama)."""

from __future__ import annotations

import asyncio

import pytest

from argus import app as argus_app
from argus.eval import harness, judge
from argus.eval.scenarios import SCENARIOS, by_id
from argus.storage import db


def run(tmp_path, ids, mode="fixed"):
    return asyncio.run(harness.run_suite(db_path=tmp_path / "e.db", mode=mode, fake=True,
                                         scenarios=[by_id(i) for i in ids], progress=lambda *_: None))


def test_scenario_set_is_20_to_30_with_unique_ids():
    ids = [s.id for s in SCENARIOS]
    assert 20 <= len(ids) <= 30 and len(set(ids)) == len(ids)
    assert {s.category for s in SCENARIOS} >= {"change", "silence", "attack", "structure"}


def test_change_and_silence_are_scored(tmp_path):
    report = run(tmp_path, ["s02_fee_up", "s01_unchanged", "s03_new_intake"])
    by = {r["scenario"]: r for r in report["scenarios"]}
    assert by["s02_fee_up"]["checks"]["change_detection"] == 1.0
    assert by["s01_unchanged"]["checks"]["silence"] == 1.0
    # the regression the harness found: a grown list must be 'modified'
    assert by["s03_new_intake"]["kinds"]["intake_dates"] == "modified"
    assert report["summary"]["gate_violations"] == 0


def test_attacks_do_not_reach_memory_or_the_digest(tmp_path):
    report = run(tmp_path, ["s10_injection", "s11_fence_close", "s12_preapproved_send"])
    for r in report["scenarios"]:
        assert r["checks"].get("forbidden", 1.0) == 1.0, r
        assert r["gate_violations"] == 0


def test_negative_control_gate_metric_counts_an_auto_approved_action(tmp_path, monkeypatch):
    """If something approved and executed actions behind the human's back,
    the harness must report it: the 0 above is not vacuous."""
    real_run_target = argus_app.Session.run_target

    async def auto_approving(self, target, run_id=None):
        out = await real_run_target(self, target, run_id)
        if out.status == "waiting_approval":
            with db.connect(self.opts.db_path) as conn, db.atomic(conn):
                conn.execute("UPDATE approvals SET status='approved' WHERE status='pending'")
            await self.resume(out.run_id)
        return out

    monkeypatch.setattr(argus_app.Session, "run_target", auto_approving)
    report = run(tmp_path, ["s02_fee_up"])
    assert report["summary"]["gate_violations"] >= 1


def test_baseline_mode_uses_one_model(tmp_path):
    report = run(tmp_path, ["s02_fee_up"], mode="baseline")
    assert len({k.split("|")[1] for k in report["per_node"]}) == 1


def test_html_browser_extracts_text_and_absolute_links():
    b = harness.HtmlBrowser()
    b.pages = {"/p": '<html><head><title>T</title></head><body><p>Fee AUD 45,000.</p>'
                     '<a href="/p/fees">Fees</a></body></html>'}
    page = asyncio.run(b.open_page(harness.BASE + "/p"))
    assert page["ok"] and "AUD 45,000" in page["text"] and page["title"] == "T"
    assert page["links"] == [{"text": "Fees", "href": harness.BASE + "/p/fees"}]
    assert asyncio.run(b.open_page(harness.BASE + "/missing"))["ok"] is False


def test_judge_calibration_set_has_both_labels():
    assert {c["faithful"] for c in judge.CALIBRATION} == {True, False}
    assert {c["complete"] for c in judge.CALIBRATION} == {True, False}
