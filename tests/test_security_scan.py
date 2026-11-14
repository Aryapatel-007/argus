"""The security scanner catches what it claims to, and nothing in clean code."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("security_scan", REPO / "scripts" / "security_scan.py")
scan_mod = importlib.util.module_from_spec(spec)
sys.modules["security_scan"] = scan_mod  # dataclasses resolve their module by name
spec.loader.exec_module(scan_mod)


def _codes(path: Path):
    return {f.code for f in scan_mod.scan_file(path) if not f.allowed}


@pytest.mark.parametrize("code,source", [
    ("hosted-llm-sdk", "import openai\n"),
    ("hosted-llm-sdk", "from anthropic import Anthropic\n"),
    ("telemetry", "import posthog\n"),
    ("telemetry", "from opentelemetry.sdk.trace import TracerProvider\n"),
    ("secret-literal", "API_KEY = 'abcd1234efgh5678'\n"),
    ("shell-true", "subprocess.run(cmd, shell=True)\n"),
    ("eval-exec", "eval(user_input)\n"),
    ("os-kill", "os.kill(pid, 0)\n"),
    ("ollama-no-think", "client.chat(model=m, messages=msgs)\n"),
    ("sql-fstring", "conn.execute(f\"SELECT * FROM t WHERE x = {x}\")\n"),
    ("outbound-url", "URL = 'https://tracker.example-analytics.net/collect'\n"),
    ("blocking-requests", "import requests\n"),
])
def test_rule_fires(tmp_path, code, source):
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    assert code in _codes(path)


def test_print_is_flagged_only_inside_servers(tmp_path):
    server = tmp_path / "servers" / "thing.py"
    server.parent.mkdir()
    server.write_text("print('hi')\nprint('ok', file=sys.stderr)\n", encoding="utf-8")
    other = tmp_path / "tool.py"
    other.write_text("print('hi')\n", encoding="utf-8")
    flagged = [f for f in scan_mod.scan_file(server) if f.code == "server-print"]
    assert [f.line for f in flagged] == [1]
    assert "server-print" not in _codes(other)


def test_think_on_a_following_line_is_accepted(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("r = client.chat(\n    model=m,\n    think=False,\n)\n", encoding="utf-8")
    assert "ollama-no-think" not in _codes(path)


def test_unfenced_page_text_in_prompt_is_flagged(tmp_path):
    prompt = tmp_path / "prompts" / "extract.md"
    prompt.parent.mkdir()
    prompt.write_text("Page:\n$page_text\n<untrusted>\n$page_text\n</untrusted>\n", encoding="utf-8")
    lines = [f.line for f in scan_mod.scan_file(prompt) if f.code == "unfenced-page-text"]
    assert lines == [2]


def test_docstrings_and_comments_do_not_trip_code_rules(tmp_path):
    path = tmp_path / "servers" / "doc.py"
    path.parent.mkdir()
    path.write_text('"""Never print() here or call os.kill(pid, 0).\n\nshell=True is bad."""\n'
                    "# eval(x) would be bad\nx = 1\n", encoding="utf-8")
    assert _codes(path) == set()


def test_allowlist_needs_a_reason(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("os.kill(pid, 0)  # security-scan: ok posix only\n"
                    "os.kill(pid, 0)  # security-scan: ok\n", encoding="utf-8")
    findings = scan_mod.scan_file(path)
    assert [f.allowed for f in findings if f.code == "os-kill"] == [True, False]


def test_negative_control_clean_file_has_no_findings(tmp_path):
    path = tmp_path / "clean.py"
    path.write_text("import httpx\n\nasync def f(c):\n    return await c.get('http://127.0.0.1:11434/')\n",
                    encoding="utf-8")
    assert scan_mod.scan_file(path) == []


def test_the_repo_itself_has_no_blocking_findings():
    findings = scan_mod.scan([REPO / "src", REPO / "scripts"])
    blocking = [f for f in findings if not f.allowed and f.severity in ("CRITICAL", "HIGH")]
    assert blocking == [], "\n".join(f"{f.path}:{f.line} {f.code}" for f in blocking)
