"""Dynamic expansion (replan -> follow_link) and procedural memory (E10, E11)."""

from __future__ import annotations

import asyncio
import json

from argus.graph import procedural
from argus.storage import db

from tests.test_graph_engine import TARGET, URL, env, fake_embed, make_engine, q  # noqa: F401

FEES_URL = URL.rsplit("/", 1)[0] + "/fees"
MAIN = {"title": "Master of AI", "text": (
    "Master of Artificial Intelligence. Duration: 2 years full-time. "
    "Intakes: February and July each year. Entry requires a bachelor degree with a WAM of 65."),
    "links": [{"text": "Fees and scholarships", "href": FEES_URL},
              {"text": "Contact us", "href": URL + "/contact"},
              {"text": "Partner site", "href": "https://elsewhere.example/fees"}]}
FEES = {"title": "Fees", "text": "The annual tuition fee for 2027 is AUD 45,000. Scholarships are listed below."}


def run(db_path, outbox):
    eng, backend = make_engine(db_path, outbox, MAIN)
    eng.services.browser.pages[FEES_URL] = FEES
    run_id, gid = eng.start_run({**TARGET, "notify": False})
    eng._bind_memory(run_id)
    return asyncio.run(asyncio.wait_for(eng.run_graph(gid), timeout=60)), gid


def test_missing_field_triggers_a_replan_that_follows_a_same_site_link(env):
    db_path, outbox = env
    out, gid = run(db_path, outbox)
    assert out.status == "done"
    types = [r["node_type"] for r in q(db_path, "SELECT node_type FROM graph_nodes WHERE graph_id=? "
                                       "ORDER BY created_at", gid)]
    assert "replan" in types and "search" in types and types.count("fetch_page") == 2
    search = q(db_path, "SELECT output FROM graph_nodes WHERE node_type='search'")[0]
    assert json.loads(search["output"])["url"] == FEES_URL  # never the off-site link
    fee = q(db_path, "SELECT content, source_url FROM facts WHERE entity_key='fees' AND is_current=1")
    assert fee and "45,000" in fee[0]["content"]
    assert fee[0]["source_url"] == FEES_URL, "a fact records the page it was actually read from"
    assert "45,000" in out.digest  # the late page joined the same digest (open fan-in)
    exp = q(db_path, "SELECT kind, accepted FROM graph_expansions WHERE kind='replan'")
    assert exp == [{"kind": "replan", "accepted": 1}]


def test_a_replan_that_works_twice_becomes_a_planner_hint(env):
    db_path, outbox = env
    run(db_path, outbox)
    with db.connect(db_path) as conn:
        assert procedural.hints_for(conn, TARGET["id"]) == []
    run(db_path, outbox)
    with db.connect(db_path) as conn:
        hints = procedural.hints_for(conn, TARGET["id"])
    # The learned step is the replan's follow_link, with the keywords that worked.
    assert hints and hints[0]["template"] == "follow_link"
    assert hints[0]["params"]["fields"] == ["fees"]
    _, gid3 = run(db_path, outbox)
    plan = q(db_path, "SELECT params FROM graph_nodes WHERE node_type='plan' AND graph_id=?", gid3)[0]
    assert json.loads(plan["params"])["hints"], "the third run's planner must be shown the learned step"
    prompt = q(db_path, "SELECT 1 FROM llm_calls WHERE role='plan' AND graph_id=?", gid3)
    assert prompt


def test_builtin_template_outcomes_are_counted_once_per_instance(env):
    db_path, outbox = env
    run(db_path, outbox)
    rows = q(db_path, "SELECT template_id, uses, successes FROM templates WHERE origin='builtin'")
    by = {r["template_id"]: r for r in rows}
    assert by["check_page"]["uses"] >= 1 and by["digest"]["uses"] == 1
    outcomes = q(db_path, "SELECT COUNT(*) AS n, COUNT(DISTINCT template_instance) AS d FROM template_outcomes")[0]
    assert outcomes["n"] == outcomes["d"]
