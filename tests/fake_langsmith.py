"""A LangSmith client that records what it would send, and sends nothing.

The client is the real SDK's, so what the tests see is what LangSmith would
receive -- every run's body as the SDK serializes it, posted when the run
starts and patched when it ends. Only the transport is replaced: an adapter
mounted on the client's own HTTP session answers every request itself. The
client sends each run as it happens rather than in background batches
(``auto_batch_tracing=False``), so a test can read a run the moment it ends.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import requests
import requests.adapters
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client

_API_URL = "http://langsmith.test"


class _Recording(requests.adapters.BaseAdapter):
    """Answers every request the client makes, recording its runs."""

    def __init__(self, recorder: TraceRecorder) -> None:
        super().__init__()
        self.recorder = recorder

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: Any = None,
        verify: Any = True,
        cert: Any = None,
        proxies: dict[str, str] | None = None,
    ) -> requests.Response:
        # Nothing is sent, so how it would have been sent does not matter.
        del stream, timeout, verify, cert, proxies
        self.recorder.record(request)
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.request = request
        response.url = request.url or ""
        return response

    def close(self) -> None:
        pass


class TraceRecorder:
    """The runs a test sent to LangSmith, as LangSmith would hold them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}
        self.client = Client(
            api_url=_API_URL, api_key="lsv2_test", auto_batch_tracing=False, info={}
        )
        # After construction: the client mounts its own adapter on its session.
        for prefix in ("http://", "https://"):
            self.client.session.mount(prefix, _Recording(self))

    def record(self, request: requests.PreparedRequest) -> None:
        path = (request.url or "").removeprefix(_API_URL)
        if not path.startswith("/runs") or not isinstance(request.body, bytes | str):
            return
        body = json.loads(request.body)
        with self._lock:
            # A POST starts a run, a PATCH ends it with its outputs; LangSmith
            # keeps the two as one record.
            self._runs.setdefault(body["id"], {}).update(body)

    @property
    def runs(self) -> list[dict[str, Any]]:
        # LangChain's tracer may hand a run to a worker; wait for every one.
        wait_for_all_tracers()
        with self._lock:
            return [dict(run) for run in self._runs.values()]

    def named(self, name: str) -> dict[str, Any]:
        (run,) = [run for run in self.runs if run["name"] == name]
        return run

    def parents(self) -> dict[str, str | None]:
        """Each run's name, mapped to its parent's (None for a root)."""
        runs = self.runs
        by_id = {run["id"]: run["name"] for run in runs}
        return {run["name"]: by_id.get(run.get("parent_run_id") or "") for run in runs}
