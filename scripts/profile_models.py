"""Profile every model in the pool. Results become the router's priors.

    python scripts/profile_models.py                 # whole pool, data/argus.db
    python scripts/profile_models.py --models phi4-mini:latest --repeats 1
    python scripts/profile_models.py --skip-coresidency --skip-think-levels

Measured per model (node_type '*'):
  cold_load_s, warm_load_s   load_duration from Ollama, after unloading everything
  vram_mb, gpu_fraction      `ollama ps` size_vram / size
  tok_s                      mean over the battery's calls
  think_capable              'thinking' in `ollama show` capabilities
  think_false_honored        think=False returns no thinking text
  think_levels_coerced       think="low" output identical to think=True
                             (thinking-capable models only, 90s cap each;
                             NULL if both runs hit the cap)
Per (model, node type) over the labelled fixtures (src/argus/eval/node_fixtures.py):
  json_valid_rate, quality, stability, mean_eval_tokens, mean_latency_s
Per pair (incl. the embedder): co_residency via `ollama ps` after loading both.

Every chat call: think=False top-level (except the think-levels probe, which
is the variable under test), 120s timeout, one llm_calls row (role 'profiler').
Results are written to the DB as they are measured, so a crash keeps
everything finished so far. A summary goes to notes/model-profiles.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import httpx  # noqa: E402
from ollama import Client  # noqa: E402

from argus import config  # noqa: E402
from argus.eval import node_fixtures  # noqa: E402
from argus.graph import llm_nodes, registry  # noqa: E402
from argus.graph.llm import parse  # noqa: E402
from argus.ollama_client import TIMEOUT_SECONDS, log_call, tok_s  # noqa: E402
from argus.storage import db  # noqa: E402
from argus.storage.migrate import apply_migrations  # noqa: E402

NS = 1_000_000_000
THINK_PROBE_CAP_S = 90
PROBE_PROMPT = "In one sentence, what does a task scheduler do?"


def log(msg: str) -> None:
    print(f"[profile {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Profiler:
    def __init__(self, db_path: Path, repeats: int):
        self.db_path = db_path
        self.repeats = repeats
        self.run_id = f"profile-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        host = config.ollama_options().get("host")
        self.client = Client(host=host, timeout=TIMEOUT_SECONDS)
        self.probe_client = Client(host=host, timeout=THINK_PROBE_CAP_S)
        self.conn_cm = db.connect(db_path)
        self.conn = self.conn_cm.__enter__()
        with db.atomic(self.conn):
            self.conn.execute(
                "INSERT INTO runs (run_id, target_id, status) VALUES (?, 'profiler', 'running')",
                (self.run_id,),
            )

    def close(self, status: str) -> None:
        with db.atomic(self.conn):
            self.conn.execute(
                "UPDATE runs SET status = ?, finished_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                "WHERE run_id = ?", (status, self.run_id))
        self.conn_cm.__exit__(None, None, None)

    def put(self, model: str, node_type: str, metric: str, value: Optional[float],
            samples: int = 0, detail: Any = None) -> None:
        with db.atomic(self.conn):
            registry.write_profile(self.conn, model, node_type, metric, value, samples,
                                   json.dumps(detail, default=str) if detail is not None else None,
                                   self.run_id)

    # -- residency ----------------------------------------------------------

    def loaded(self) -> Dict[str, Any]:
        return {config.normalize_tag(m.model): m for m in (self.client.ps().models or [])}

    def unload_all(self) -> None:
        for tag in list(self.loaded()):
            try:
                if "embed" in tag:
                    self.client.embed(model=tag, input="x", keep_alive=0)
                else:
                    self.client.generate(model=tag, prompt="", keep_alive=0, think=False)
            except Exception as exc:  # noqa: BLE001
                log(f"unload {tag}: {exc!r}")
        deadline = time.time() + 30
        while self.loaded() and time.time() < deadline:
            time.sleep(0.5)

    def load(self, tag: str) -> float:
        """Seconds to make `tag` resident. Wall clock: an empty-prompt
        generate does not report load_duration (measured: 0), and wall time is
        what a swap actually costs the scheduler."""
        start = time.perf_counter()
        if "embed" in tag:
            r = self.client.embed(model=tag, input="warm", keep_alive="10m")
        else:
            r = self.client.generate(model=tag, prompt="", keep_alive="10m", think=False)
        wall = time.perf_counter() - start
        return max(wall, (getattr(r, "load_duration", 0) or 0) / NS)

    # -- per model ----------------------------------------------------------

    def chat(self, model: str, messages, schema, role_detail: str):
        start = time.perf_counter()
        response, err, timed_out, raw = None, None, False, None
        try:
            response = self.client.chat(
                model=model, messages=messages, think=False, keep_alive="10m",
                format=schema.model_json_schema(),
                options={"temperature": 0, "num_ctx": config.ollama_options().get("num_ctx", 4096)},
            )
            raw = response.message.content
        except httpx.TimeoutException as exc:
            timed_out, err = True, f"timeout after {TIMEOUT_SECONDS}s: {exc!r}"
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        wall = time.perf_counter() - start
        parsed = None
        if response is not None:
            parsed, err = parse(schema, raw)
        log_call(self.conn, run_id=self.run_id, task_id=None, step_index=None, role="profiler",
                 model_tag=model, think_mode="False", prompt_file=role_detail, response=response,
                 ok=parsed is not None, timed_out=timed_out, error=err, raw_response=raw)
        return parsed, response, wall, err

    def think_probe(self, model: str, capable: bool) -> None:
        r = self.client.chat(model=model, messages=[{"role": "user", "content": PROBE_PROMPT}],
                             think=False, options={"temperature": 0})
        thinking = getattr(r.message, "thinking", None)
        self.put(model, "*", "think_false_honored", 1.0 if not thinking else 0.0, 1,
                 {"eval_count": r.eval_count, "thinking_chars": len(thinking or "")})
        if not capable or self.skip_levels:
            self.put(model, "*", "think_levels_coerced", None, 0,
                     {"skipped": "not thinking-capable" if not capable else "--skip-think-levels"})
            return
        outs = {}
        for level in ("low", True):
            try:
                rr = self.probe_client.chat(  # security-scan: ok think is the variable under test here
                    model=model, messages=[{"role": "user", "content": PROBE_PROMPT}],
                    think=level, options={"temperature": 0})
                outs[str(level)] = (rr.eval_count, rr.message.content)
            except Exception as exc:  # noqa: BLE001 - a runaway is a result
                outs[str(level)] = ("capped", repr(exc)[:200])
        a, b = outs["low"], outs["True"]
        value = None if a[0] == "capped" and b[0] == "capped" else (1.0 if a == b else 0.0)
        self.put(model, "*", "think_levels_coerced", value, 2, {"low": str(a)[:300], "true": str(b)[:300]})

    def battery(self, model: str) -> List[float]:
        rates: List[float] = []
        for node_type in node_fixtures.LLM_NODE_TYPES:
            spec = llm_nodes.node(node_type)
            # entity_match is the model check INSIDE resolve_entity; the router
            # looks profiles up by node type, so store it under that name.
            profile_key = "resolve_entity" if node_type == "entity_match" else node_type
            valid = calls = 0
            scores, tokens, latencies, stable = [], [], [], []
            for fx in node_fixtures.by_type(node_type):
                sigs = []
                for rep in range(self.repeats):
                    msgs = spec.messages(fx.inputs, fx.params, fx.target)
                    parsed, resp, wall, err = self.chat(model, msgs, spec.schema, f"{node_type}:{fx.name}")
                    calls += 1
                    latencies.append(wall)
                    if resp is not None and resp.eval_count:
                        tokens.append(resp.eval_count)
                        r = tok_s(resp.eval_count, resp.eval_duration)
                        if r:
                            rates.append(r)
                    if parsed is not None:
                        valid += 1
                        sigs.append(fx.signature(parsed))
                    ev = node_fixtures.evaluate(fx, parsed)
                    scores.append(ev["score"])
                    log(f"  {model} {node_type}:{fx.name} rep{rep} valid={parsed is not None} "
                        f"score={ev['score']:.2f} {wall:.1f}s {err or ''}"[:220])
                if len(sigs) >= 2:
                    stable.append(sum(s == sigs[0] for s in sigs[1:]) / (len(sigs) - 1))
            self.put(model, profile_key, "json_valid_rate", valid / calls if calls else None, calls)
            self.put(model, profile_key, "quality", statistics.mean(scores) if scores else None, len(scores))
            self.put(model, profile_key, "stability", statistics.mean(stable) if stable else None, len(stable))
            self.put(model, profile_key, "mean_eval_tokens", statistics.mean(tokens) if tokens else None, len(tokens))
            self.put(model, profile_key, "mean_latency_s", statistics.mean(latencies) if latencies else None,
                     len(latencies))
        return rates

    def profile_model(self, model: str) -> None:
        log(f"=== {model}")
        self.unload_all()
        t0 = time.perf_counter()
        cold = self.load(model)
        cold_wall = time.perf_counter() - t0
        ps = self.loaded().get(model)
        size = getattr(ps, "size", None)
        size_vram = getattr(ps, "size_vram", None)
        warm = self.load(model)
        self.put(model, "*", "cold_load_s", cold, 1, {"wall_s": cold_wall})
        self.put(model, "*", "warm_load_s", warm, 1)
        self.put(model, "*", "vram_mb", (size_vram or 0) / 2**20 if size_vram else None, 1, {"size": size})
        self.put(model, "*", "gpu_fraction", (size_vram / size) if size and size_vram is not None else None, 1)
        log(f"  cold {cold:.2f}s warm {warm:.2f}s vram {((size_vram or 0) / 2**20):.0f}MB "
            f"gpu {((size_vram or 0) / size if size else 0):.2f}")
        capable = "thinking" in (getattr(self.client.show(model), "capabilities", None) or [])
        self.put(model, "*", "think_capable", 1.0 if capable else 0.0, 1)
        self.think_probe(model, capable)
        rates = self.battery(model)
        self.put(model, "*", "tok_s", statistics.mean(rates) if rates else None, len(rates))
        log(f"  tok/s {statistics.mean(rates) if rates else float('nan'):.1f}")

    def coresidency(self, models: List[str]) -> None:
        embed = config.normalize_tag(config.load_config()["models"]["embed"])
        # BOTH load orders: measured order-dependent (2026-11-14). Loading
        # nomic then qwen evicts nomic; Sprint 2 loaded qwen then nomic and
        # both stayed. A pair counts as co-resident only if both orders keep
        # both, because the scheduler cannot control which one loads first.
        for a, b in itertools.combinations(sorted(set(models + [embed])), 2):
            orders = {}
            for first, second in ((a, b), (b, a)):
                self.unload_all()
                try:
                    self.load(first)
                    self.load(second)
                    time.sleep(1.0)
                    loaded = self.loaded()
                    orders[f"{first} then {second}"] = {
                        "both": first in loaded and second in loaded,
                        "loaded": {t: {"size": getattr(m, "size", None),
                                       "size_vram": getattr(m, "size_vram", None)} for t, m in loaded.items()}}
                except Exception as exc:  # noqa: BLE001
                    orders[f"{first} then {second}"] = {"both": False, "error": repr(exc)}
            both = all(o["both"] for o in orders.values())
            with db.atomic(self.conn):
                registry.write_coresidency(self.conn, a, b, both, json.dumps(orders))
            per_order = ", ".join(f"{k}: {v['both']}" for k, v in orders.items())
            log(f"  co-resident {a} + {b}: {both} ({per_order})")


def write_notes(db_path: Path, models: List[str]) -> Path:
    with db.connect(db_path) as conn:
        prof = registry.profiles(conn)
        co = registry.coresidency(conn)
    lines = ["# Model profiles", "",
             f"Written by `scripts/profile_models.py` on {time.strftime('%Y-%m-%d %H:%M')}. "
             "Live numbers from this machine; the router reads them from `model_profiles`.", "",
             "## Model-wide", "",
             "| model | cold load s | warm load s | VRAM MB | GPU share | tok/s | think capable | think=False honoured | levels coerced |",
             "|---|---|---|---|---|---|---|---|---|"]

    def f(v, fmt="{:.2f}"):
        return "-" if v is None else fmt.format(v)

    for m in models:
        g = lambda k: prof.get((m, "*", k))  # noqa: E731
        lines.append(f"| {m} | {f(g('cold_load_s'))} | {f(g('warm_load_s'))} | {f(g('vram_mb'), '{:.0f}')} | "
                     f"{f(g('gpu_fraction'))} | {f(g('tok_s'), '{:.1f}')} | {f(g('think_capable'), '{:.0f}')} | "
                     f"{f(g('think_false_honored'), '{:.0f}')} | {f(g('think_levels_coerced'), '{:.0f}')} |")
    lines += ["", "## Per node type: quality / JSON valid rate / stability / mean latency s", "",
              "| node type | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for nt in node_fixtures.LLM_NODE_TYPES:
        cells = []
        for m in models:
            q, v, s, lat = (prof.get((m, nt, k)) for k in ("quality", "json_valid_rate", "stability", "mean_latency_s"))
            cells.append(f"{f(q)} / {f(v)} / {f(s)} / {f(lat, '{:.1f}')}")
        lines.append(f"| {nt} | " + " | ".join(cells) + " |")
    lines += ["", "## Co-residency (both stayed loaded together)", ""]
    seen = set()
    for (a, b), v in sorted(co.items()):
        if (b, a) in seen:
            continue
        seen.add((a, b))
        lines.append(f"- {a} + {b}: {'yes' if v else 'no'}")
    out = REPO / "notes" / "model-profiles.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(db.DB_PATH))
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--skip-coresidency", action="store_true")
    ap.add_argument("--skip-think-levels", action="store_true")
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    apply_migrations(db_path)
    usable, missing = registry.pool()
    if missing:
        log(f"allowed but not installed (skipped): {missing}")
    models = args.models or [m.tag for m in usable]
    prof = Profiler(db_path, args.repeats)
    prof.skip_levels = args.skip_think_levels
    status = "failed"
    try:
        for m in models:
            try:
                prof.profile_model(m)
            except Exception as exc:  # noqa: BLE001 - one bad model must not lose the rest
                log(f"!! {m} failed: {exc!r}")
                prof.put(m, "*", "profile_error", 1.0, 0, {"error": repr(exc)})
        if not args.skip_coresidency:
            prof.coresidency(models)
        status = "done"
    finally:
        prof.unload_all()
        prof.close(status)
    log(f"notes: {write_notes(db_path, models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
