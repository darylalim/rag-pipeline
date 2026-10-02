"""Optional tracing to LangSmith: the client the pipeline sends its runs with.

Off unless ``LANGSMITH_TRACING`` is true. When it is, every question becomes one
trace in LangSmith: a root run opened by ``RAGPipeline.stream_answer``, with the
vector search, the rerank and the generation nested under it (pipeline.py
builds them). Only questions are traced -- ingest runs nothing a trace would
describe.

The client is LangSmith's own, but not as it comes. Its defaults retry a
failed send three times on top of its own three attempts, and wait 10 s to
connect, and the queue of runs it has not yet sent is drained before the
process may exit: with LangSmith unreachable, a finished ``rag query`` held
its exit for 12 s against a refused connection and 88 s against a host that
never answers. Questions themselves never wait -- runs are sent from a
background thread. Capped here (one try per attempt, a 1 s connect), the
same exits took 0.3 s and 10 s; the 10 s is the SDK's floor, three attempts
at a fixed 3 s connect it offers no setting for.
"""

from __future__ import annotations

import threading

from langsmith import Client
from urllib3.util.retry import Retry

from rag_pipeline.config import Settings, require_env_key

# (connect, read) in milliseconds, for every request the client makes itself.
# LangSmith answers in well under a second; these are reached only when it is
# down or the network is, and they bound how long that costs -- not tunables.
_TIMEOUT_MS = (1_000, 5_000)

# One client per API key per process. Each owns a background thread that sends
# runs in batches, so a client per pipeline would leave one more thread behind
# every time the app rebuilds its pipeline after an ingest.
_clients: dict[str, Client] = {}
_clients_lock = threading.Lock()


def tracing_client(settings: Settings) -> Client | None:
    """The client questions are traced with, or None when tracing is off.

    Built on the pipeline-load path, so a missing ``LANGSMITH_API_KEY`` is a
    RuntimeError there -- before the first question rather than as a trace
    silently lost after it. Shared by every pipeline in the process.
    """
    if not settings.langsmith_tracing:
        return None
    key = require_env_key(
        "LANGSMITH_API_KEY", "Tracing to LangSmith (LANGSMITH_TRACING=true)"
    )
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            # LANGSMITH_ENDPOINT, if set, is read by the client itself, as for
            # `rag eval`'s: the region is LangSmith's to name, not a setting.
            client = Client(
                api_key=key, timeout_ms=_TIMEOUT_MS, retry_config=Retry(total=0)
            )
            _clients[key] = client
        return client
