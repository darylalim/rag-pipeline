"""Optional tracing to a self-hosted Phoenix, over OpenTelemetry.

Off unless ``PHOENIX_COLLECTOR_ENDPOINT`` is set. When it is, every question
becomes one trace in Phoenix: a root span opened by ``stream_answer``, with the
vector search, the rerank and the generation nested under it. The split is
OpenTelemetry's own: ``pipeline.py`` opens spans through the API alone, which
does nothing until a provider exists, and each frontend calls
:func:`setup_tracing` to install one. Only questions are traced -- ingest runs
nothing a span would describe.

Assembled from OpenTelemetry's parts rather than through ``phoenix.otel``'s
``register()``, because each of its conveniences is a trap here:

- Given a base URL, ``register()`` posts to it as-is -- Phoenix answers ``POST
  /`` with a 405 -- while ``force_flush()`` still reports success, and left to
  infer a protocol it picks gRPC, whose sockets the test suite's offline guard
  cannot see. So the collector URL is always built here, and always HTTP.
- It cannot set the exporter's timeout, and that timeout is how long a
  finished ``rag query`` hangs at exit when Phoenix is not running: about 7 s
  at the default, about 1 s at the 2 s set here.
- It reads ``PHOENIX_*`` variables, and ``.env.phoenix`` files anywhere above
  the working directory, that ``Settings`` knows nothing about.
"""

from __future__ import annotations

import threading
from urllib.parse import urlsplit, urlunsplit

from rag_pipeline.config import Settings

# Seconds the exporter may spend on one batch, retries included, when Phoenix
# is down: a refused connection costs about half of it, a host that never
# answers all of it (twice it before OpenTelemetry 1.45: the older exporter
# retried a timed-out connect with a fresh budget). It is what keeps the flush a
# finished `rag query` makes at exit short -- that flush first waits out any
# batch already being sent, so it caps each export rather than the whole wait.
# A local Phoenix accepts a batch in milliseconds, so this is reached only when
# something is wrong -- not a tunable.
_EXPORT_TIMEOUT_S = 2.0

# Streamlit runs every session's script on its own thread, and each rerun calls
# setup_tracing again.
_setup_lock = threading.Lock()


def traces_url(endpoint: str) -> str:
    """The OTLP/HTTP collector under a Phoenix base URL: ``<endpoint>/v1/traces``.

    A path prefix is kept, for a Phoenix behind a reverse proxy, and an endpoint
    that already names the collector is used as it is.
    """
    url = urlsplit(endpoint)
    path = url.path.rstrip("/")
    if not path.endswith("/v1/traces"):
        path += "/v1/traces"
    return urlunsplit(url._replace(path=path))


def setup_tracing(settings: Settings) -> None:
    """Send this process's traces to Phoenix, if an endpoint is configured.

    Once per process: the first call with an endpoint installs the provider and
    instruments LangChain, and every later call returns at once, which is what
    lets the app call this on every rerun. Each provider owns an exporter
    thread and an exit-time flush, and OpenTelemetry keeps only the first
    provider a process installs, so a later endpoint could not take effect
    anyway.

    Raises only RuntimeError, like the model adapters, because the app calls
    this on its pipeline-load path, whose handler sits below the sidebar and
    does not catch ValueError.
    """
    if not settings.phoenix_collector_endpoint:
        return

    # Imported only once tracing is on: off -- the default -- none of this
    # loads, and `rag --help` never pays for it either way.
    from openinference.instrumentation import TracerProvider
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    with _setup_lock:
        instrumentor = LangChainInstrumentor()  # a process-wide singleton
        if instrumentor.is_instrumented_by_opentelemetry:
            return
        # OpenInference's provider, not the SDK's: the SDK's keeps only the
        # first 128 attributes of a span, and a reranker span carries three per
        # candidate and four per kept passage -- past that cap from a FETCH_K
        # of about 37. Phoenix files each trace under the project this
        # resource names, creating it on first use.
        #
        # Batched: a simple processor exports inside span.end(), so with
        # Phoenix down every span of a question would stall it for the whole
        # retry budget. The batch goes out on a background thread instead, and
        # whatever is queued at exit is flushed then.
        #
        # Each of these also reads the standard OTEL_* variables. Most
        # malformed ones are logged and replaced by a default, but a malformed
        # span limit, or an out-of-range batch setting (zero, or a batch larger
        # than its queue), is a builtins ValueError that would escape the app's
        # guard -- as were an unknown compression and an out-of-range sampler
        # ratio, before OpenTelemetry 1.45. The exporter refuses a credential
        # provider (OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER) that is
        # not installed with a RuntimeError, which is caught as well: it comes
        # after the provider is built, and its message names no variable.
        # Nothing is installed until all of it is built, so a failure leaves
        # tracing off and the next rerun reports the same error; a provider
        # already built is shut down, which unregisters the exit hook each
        # failed rerun would otherwise add.
        provider = None
        try:
            provider = TracerProvider(
                resource=Resource.create(
                    {ResourceAttributes.PROJECT_NAME: settings.phoenix_project}
                )
            )
            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=traces_url(settings.phoenix_collector_endpoint),
                        timeout=_EXPORT_TIMEOUT_S,
                    )
                )
            )
        except (RuntimeError, ValueError) as exc:
            if provider is not None:
                provider.shutdown()
            raise RuntimeError(
                f"PHOENIX_COLLECTOR_ENDPOINT is set, but tracing could not be set "
                f"up: {exc}. Check the OTEL_* variables in the environment and .env."
            ) from exc
        trace.set_tracer_provider(provider)
        instrumentor.instrument(tracer_provider=provider)
