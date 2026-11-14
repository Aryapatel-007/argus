"""Argus security scan: fast, local, rule-based. Used by the
`argus-security-audit` project skill and run at the end of Phase D and F.

    python scripts/security_scan.py            # scan src/ and scripts/
    python scripts/security_scan.py PATH ...   # scan specific files or dirs

Exit 1 if any CRITICAL or HIGH finding is not allowlisted. A line is
allowlisted with a trailing comment `# security-scan: ok <reason>`; the reason
is mandatory so every exception is explained where it lives.

What it looks for is specific to Argus's hard constraints (CLAUDE.md):
hosted model SDKs and telemetry (constraints 1 and 3), stdout writes in MCP
servers, Ollama calls without explicit think=False, shell injection,
os.kill (TerminateProcess on Windows), secrets, SQL built from untrusted
strings, outbound URLs that are not localhost, and prompt templates that
splice untrusted page text without a fence.

It is a tripwire, not a proof. The skill's checklist covers what a regex
cannot (approval gate structure, prompt injection into tool arguments).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

REPO = Path(__file__).resolve().parents[1]
ALLOW = re.compile(r"#\s*security-scan:\s*ok\s+\S")


@dataclass(frozen=True)
class Rule:
    code: str
    severity: str
    pattern: re.Pattern
    message: str
    applies: str = "py"  # "py", "server", "prompt"


RULES: List[Rule] = [
    Rule("hosted-llm-sdk", "CRITICAL",
         re.compile(r"^\s*(import|from)\s+(openai|anthropic|google\.generativeai|google\.genai|cohere|mistralai|groq)\b"),
         "hosted model SDK import (hard constraint 1: local models only)"),
    Rule("telemetry", "CRITICAL",
         re.compile(r"^\s*(import|from)\s+(posthog|sentry_sdk|opentelemetry\.sdk|opentelemetry\.exporter|segment|mixpanel|datadog)\b"),
         "telemetry library (hard constraint 3: no telemetry)"),
    Rule("secret-literal", "CRITICAL",
         re.compile(r"(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|(?i:(api_key|apikey|secret|password|token)\s*=\s*['\"][^'\"]{8,}['\"]))"),
         "credential-looking literal in source"),
    Rule("shell-true", "HIGH", re.compile(r"shell\s*=\s*True"),
         "subprocess with shell=True (injection)"),
    Rule("eval-exec", "HIGH", re.compile(r"(?<![\w.])(eval|exec)\s*\("),
         "eval/exec on dynamic input"),
    Rule("os-kill", "HIGH", re.compile(r"\bos\.kill\s*\("),
         "os.kill: on Windows this is TerminateProcess, even with signal 0"),
    Rule("server-print", "HIGH", re.compile(r"(?<![\w.])print\s*\((?![^)]*file\s*=\s*sys\.stderr)"),
         "print() to stdout inside an MCP server corrupts the JSON-RPC stream", applies="server"),
    Rule("ollama-no-think", "HIGH",
         re.compile(r"\.(chat|generate)\s*\((?![^)]*\bthink\s*=)"),
         "Ollama chat/generate call without explicit think= (CLAUDE.md: think=False on every call)"),
    Rule("sql-fstring", "MEDIUM",
         re.compile(r"\.execute\(\s*f[\"']"),
         "SQL built with an f-string; only placeholder lists may be interpolated"),
    Rule("outbound-url", "MEDIUM",
         re.compile(r"https?://(?!127\.0\.0\.1|localhost|example\.(invalid|com|edu|org)|\[::1\])[\w.-]+"),
         "literal non-local URL in code; runtime network use must be explained"),
    Rule("blocking-requests", "MEDIUM", re.compile(r"^\s*(import|from)\s+requests\b"),
         "blocking `requests` (use httpx.AsyncClient)"),
    Rule("unfenced-page-text", "HIGH",
         re.compile(r"^(?!.*<untrusted).*\$(page_text|text|content|body)\b"),
         "untrusted text spliced into a prompt without an <untrusted> fence", applies="prompt"),
]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    code: str
    severity: str
    message: str
    text: str
    allowed: bool


def _kind(path: Path) -> List[str]:
    kinds = []
    if path.suffix == ".py":
        kinds.append("py")
        if "servers" in path.parts and path.name not in ("__init__.py", "scoring.py"):
            kinds.append("server")
    if path.suffix == ".md" and "prompts" in path.parts:
        kinds.append("prompt")
    return kinds


#: Rules that look at string contents on purpose. Every other rule sees code
#: only: strings blanked, comments removed, so a docstring that MENTIONS
#: print() or os.kill does not trip the rule about calling them.
STRING_RULES = {"secret-literal", "outbound-url", "unfenced-page-text", "sql-fstring"}


def _code_views(text: str):
    """Per line: (code_only, inside_multiline_string)."""
    import io
    import tokenize

    lines = text.splitlines()
    code = [list(line) for line in lines]
    in_multiline = [False] * len(lines)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type not in (tokenize.STRING, tokenize.COMMENT) and getattr(tokenize, "FSTRING_MIDDLE", -1) != tok.type:
                continue
            (sr, sc), (er, ec) = tok.start, tok.end
            for r in range(sr, er + 1):
                if r - 1 >= len(code):
                    break
                start = sc if r == sr else 0
                end = ec if r == er else len(code[r - 1])
                for c in range(start, min(end, len(code[r - 1]))):
                    code[r - 1][c] = " "
                if er > sr and tok.type == tokenize.STRING:
                    in_multiline[r - 1] = True
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return ["".join(c) for c in code], in_multiline


def scan_file(path: Path) -> List[Finding]:
    kinds = _kind(path)
    if not kinds:
        return []
    findings = []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if path.suffix == ".py":
        code_lines, in_multiline = _code_views(text)
    else:
        code_lines, in_multiline = lines, [False] * len(lines)
    # Prompt files: a line is fenced if it sits between <untrusted ...> and
    # </untrusted>, or carries the tag itself.
    fenced = []
    depth = 0
    for line in lines:
        opens = len(re.findall(r"<untrusted\b", line))
        closes = line.count("</untrusted>")
        fenced.append(depth > 0 or opens > 0)
        depth = max(0, depth + opens - closes)
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        code = code_lines[number - 1]
        for rule in RULES:
            if rule.applies not in kinds:
                continue
            if rule.code in STRING_RULES:
                if path.suffix == ".py" and in_multiline[number - 1]:
                    continue  # docstrings and long text blocks
                subject = line
            else:
                subject = code
            if rule.code == "ollama-no-think":
                if not re.search(r"\b\w*(client|ollama)\w*\.(chat|generate)\s*\(", subject, re.I):
                    continue
                window = " ".join(lines[number - 1:number + 12])
                if re.search(r"\bthink\s*=", window):
                    continue
            elif rule.code == "unfenced-page-text":
                if fenced[number - 1] or not re.search(r"\$(page_text|text|content|body)\b", subject):
                    continue
            elif not rule.pattern.search(subject):
                continue
            findings.append(Finding(
                str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                number, rule.code, rule.severity, rule.message, stripped[:160],
                bool(ALLOW.search(line)),
            ))
    return findings


def iter_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file():
            yield root
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and ".venv" not in path.parts:
                yield path


def scan(roots: Iterable[Path]) -> List[Finding]:
    out: List[Finding] = []
    for path in iter_files(roots):
        out.extend(scan_file(path))
    return out


def main(argv: List[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    roots = [Path(a).resolve() for a in argv] or [REPO / "src", REPO / "scripts"]
    findings = scan(roots)
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    findings.sort(key=lambda f: (f.allowed, order[f.severity], f.path, f.line))
    blocking = [f for f in findings if not f.allowed and f.severity in ("CRITICAL", "HIGH")]
    for f in findings:
        tag = "allowed" if f.allowed else f.severity
        print(f"[{tag:8}] {f.path}:{f.line} {f.code}: {f.message}\n           {f.text}")
    print(f"\n{len(findings)} findings, {len(blocking)} blocking (unallowlisted CRITICAL/HIGH)")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
