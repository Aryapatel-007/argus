"""Two-minute offline demo of Argus. No internet needed.

    python scripts/demo.py           # real local models (Ollama) + real Chromium
    python scripts/demo.py --fake    # deterministic stand-in model, seconds

What happens, on a throwaway database and a local fixture website:
  1. first look at a course page: every field is new, a digest is written,
     and a notification is drafted and BLOCKED waiting for approval
  2. you (the script) approve it: the message lands in the outbox, once
  3. the fee changes on the page: Argus reports exactly that change
  4. nothing changes: Argus stays silent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from argus import app, observe  # noqa: E402
from argus.eval.fixture_server import FixtureServer  # noqa: E402
from argus.eval.scenarios import program  # noqa: E402
from argus.graph.fakes import FakeBackend  # noqa: E402


def say(title, body=""):
    print(f"\n=== {title}")
    if body:
        print(body)


async def main(fake: bool) -> None:
    work = Path(tempfile.mkdtemp(prefix="argus-demo-"))
    db_path = work / "argus.db"
    from argus.eval.harness import copy_profiles
    from argus.storage import db as dbm

    if dbm.DB_PATH.exists():
        copy_profiles(dbm.DB_PATH, db_path)  # route on this machine's measured models
    with FixtureServer({"/course": program()}) as site:
        target = {"id": "demo_course", "name": "Demo University Master of AI", "url": site.url("/course"),
                  "extract": ["intake_dates", "duration", "fees", "entry_requirements"], "notify": True}
        opts = app.Options(db_path=db_path, mode="fixed", memory="mcp", browser_args=["--no-robots"],
                           backend=FakeBackend() if fake else None, fake_embed=fake)

        async with app.Session(opts) as s:
            say("1. First look", f"watching {target['url']}")
            out = await s.run_target(target)
            say("   digest", out.digest)
            say("   status", f"{out.status}: the notification is waiting for a human")
            for a in app.list_approvals(db_path):
                print(f"   approval #{a['approval_id']}: {json.loads(a['proposed_action']).get('subject')}")

            say("2. Approve it")
            for a in app.list_approvals(db_path):
                app.decide(db_path, a["approval_id"], "approved", by="demo")
            outs = await s.resume(out.run_id)
            print(f"   run is now {outs[-1].status}; outbox: {[p.name for p in (work / 'outbox').glob('*.json')]}")

            site.pages["/course"] = program(fee="48,500")
            say("3. The fee changes on the page")
            out2 = await s.run_target(target)
            say("   digest", out2.digest)

            say("4. Nothing changes")
            out3 = await s.run_target(target)
            print(f"   digest: {out3.digest!r} (silence is a valid result)")

        tr = observe.trace(db_path, out2.run_id)["graphs"][0]
        say("Trace of run 3", f"{len(tr['spans'])} nodes, {tr['swaps']} model swaps, "
                              f"models in order: {' -> '.join(tr['model_sequence'])}")
    print(f"\nEverything is in {db_path}. Try: argus --db \"{db_path}\" dashboard")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="no Ollama: deterministic stand-in model")
    asyncio.run(main(ap.parse_args().fake))
