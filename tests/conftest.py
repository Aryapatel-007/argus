"""Suite-wide test hygiene."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _restore_embedder_and_env():
    """Some tests install deterministic fake embeddings process-wide
    (graph.fakes.install_fake_embeddings, the eval harness, app.Options with
    fake_embed) and set ARGUS_FAKE_EMBED for child servers. Undo both after
    every test, so a later test that measures REAL embeddings never runs on
    fake ones because of test order."""
    from argus import embeddings

    real = embeddings.embed
    had = os.environ.get("ARGUS_FAKE_EMBED")
    yield
    embeddings.embed = real
    if had is None:
        os.environ.pop("ARGUS_FAKE_EMBED", None)
    else:
        os.environ["ARGUS_FAKE_EMBED"] = had
