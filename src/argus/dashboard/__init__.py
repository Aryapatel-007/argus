"""Local dashboard: runs, trace trees, routing explanations, approvals, health.

    argus dashboard            http://127.0.0.1:8787

Binds 127.0.0.1 only (never 0.0.0.0). Read pages come from argus.observe.
The only writes are approve/reject, which go through app.decide (the same
path as the CLI) and require:
  - POST
  - a per-process CSRF token embedded in the form
  - a Host header that is localhost / 127.0.0.1 (DNS-rebinding defence)
No JavaScript, no CDN, no external fonts: the page loads nothing from the
internet (hard constraint 3).
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from argus import app as argus_app
from argus import observe

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def create_app(db_path: Path, csrf_token: Optional[str] = None, extra_hosts: tuple = ()) -> FastAPI:
    api = FastAPI(title="Argus", docs_url=None, redoc_url=None, openapi_url=None)
    token = csrf_token or secrets.token_urlsafe(32)
    api.state.csrf = token

    def check_host(request: Request) -> None:
        host = (request.headers.get("host") or "").split(":")[0]
        if host not in LOCAL_HOSTS | set(extra_hosts):
            raise HTTPException(403, "dashboard is local only")

    def page(request: Request, name: str, **ctx):
        check_host(request)
        return TEMPLATES.TemplateResponse(request, name, {"csrf": token, **ctx})

    @api.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return page(request, "index.html", runs=observe.runs(db_path, 30),
                    pending=argus_app.list_approvals(db_path, "pending"), health=observe.health(db_path))

    @api.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str):
        tr = observe.trace(db_path, run_id)
        if not tr["graphs"]:
            raise HTTPException(404, "no such run")
        return page(request, "run.html", trace=tr, digest=observe.digest_of(db_path, run_id))

    @api.get("/nodes/{node_id}", response_class=HTMLResponse)
    def node_page(request: Request, node_id: str):
        return page(request, "node.html", why=observe.why(db_path, node_id))

    @api.get("/approvals", response_class=HTMLResponse)
    def approvals_page(request: Request):
        import json

        rows = argus_app.list_approvals(db_path, None)
        for r in rows:
            r["action"] = json.loads(r["proposed_action"])
        return page(request, "approvals.html", rows=rows)

    @api.post("/approvals/{approval_id}")
    def decide(request: Request, approval_id: int, decision: str = Form(...), csrf: str = Form(...),
               note: str = Form("")):
        check_host(request)
        if not secrets.compare_digest(csrf.encode("utf-8"), token.encode("utf-8")):
            raise HTTPException(403, "bad csrf token")
        if decision not in ("approve", "reject"):
            raise HTTPException(400, "decision must be approve or reject")
        try:
            argus_app.decide(db_path, approval_id, "approved" if decision == "approve" else "rejected",
                             by="dashboard", note=note)
        except argus_app.DecisionError as exc:
            raise HTTPException(409, str(exc))
        return RedirectResponse("/approvals", status_code=303)

    @api.get("/health", response_class=HTMLResponse)
    def health_page(request: Request):
        return page(request, "health.html", health=observe.health(db_path))

    return api


def serve(db_path: Path, port: int = 8787) -> None:
    import uvicorn

    uvicorn.run(create_app(Path(db_path)), host="127.0.0.1", port=port, log_level="warning")
