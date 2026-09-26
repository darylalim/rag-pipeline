# tests/CLAUDE.md

The test machinery, loaded when work touches `tests/`. The rules it serves are in
the root `CLAUDE.md`: production passes no models and tests inject fakes, what
`test_app.py` is there to guarantee, and which test enforces which invariant.

## Fixtures to take rather than rebuild (`conftest.py`)

| For | Take |
| --- | ---- |
| a frontend (`streamlit_app.py` or `cli.py`) over fakes | `wired_env` — every `ENV_VARS` name from `settings`, and fakes behind the three model factories. Derived from `ENV_VARS` because a hand-kept list would let the developer's `.env` answer a missed name |
| the real adapters, down to the generation lock, with no MLX | `fake_mlx` over a `model_dir` — the shared fake stack in `fake_mlx.py`, installed for one test after `_no_real_models`, with the model memo swapped so no fake model outlives it |
| asserting on spans | `spans` — production's provider, a synchronous processor into memory, LangChain instrumented; all undone after |
| running `setup_tracing` for real | `undo_tracing` |
| what only a fresh process shows (a library reading a variable as it is first imported) | `fresh_interpreter` |
| a generation that fails partway, as a real one would | `fail_mid_stream` |

## The autouse guards

- `_no_real_models` puts `None` in `sys.modules` for `mlx` and `mlx_lm`, so
  `import mlx_lm` raises and `load_mlx_model()` reports a RuntimeError before it
  looks in the cache. **This is the one that catches a forgotten injection.** A
  test that forgets `embeddings=` names no banned symbol, and with the models
  cached it opens no socket either — a network block alone would let it load
  gigabytes of weights. It behaves identically on a Mac with MLX installed and on
  the Linux CI legs without it. Tests marked `models` are exempt; `hide_mlx`
  exposes that decision.
- `_offline` blocks every socket, to any host: Chroma is in-process and the
  models are local files, so a connection means something is downloading (a
  model, Chroma's default ONNX embedder) or phoning home.
- `_no_tracing` (session-scoped) deletes `PHOENIX_COLLECTOR_ENDPOINT` and forces
  LangSmith's switches to `false` (the pipeline no longer uses LangSmith, but
  langsmith ships inside langchain-core and still acts on them). `.env` is loaded
  at import time, and either would otherwise send test traces from a background
  flush that can land after `_offline` is undone. **`_offline` cannot catch an
  exporter**: the socket block's `RuntimeError` is caught inside the tracing
  stack and only logged — by the exporter's own HTTP transport from
  OpenTelemetry 1.45, by the batch processor before — so the test passes.
- `_no_tracer_left_on` is what makes that failure loud: after every test it
  fails one that left a global tracer provider installed or LangChain
  instrumented. It switches tracing off first, through the same
  `_switch_tracing_off()` as `undo_tracing`, so only the test that leaked fails
  (`test_a_leak_fails_only_the_test_that_left_tracing_on`, in a pytest
  subprocess). OpenTelemetry allows one global provider per process, and
  `opentelemetry-test-utils`' `reset_trace_globals()` is what undoes it.
- `_reset_store_client` drops chromadb's System cache at every test boundary —
  the stale-view hazard in the root `CLAUDE.md`.

`test_offline_guard.py` trips every route to a real model on purpose — each
factory, an ingest and a pipeline left without a fake — and checks the socket
block, the tracing guards and the `models` exemption, so a guard that loosens
reads as a failure rather than as green.

## The invariant sweep (`invariants.py`, `test_invariants.py`)

Adding a rule means:

1. a `Rule` in `RULES`;
2. a case in each direction in `test_invariants.py`: one in `VIOLATIONS`, and
   in `ALLOWED` at least one *near miss* — text the raw pattern matches that
   masking or the path lets through. A lookbehind is part of the pattern, so a
   case it excludes never matches raw: keep it as a precision case, but it
   cannot be the near miss;
3. the rule count `test_every_rule_has_a_case_in_both_directions` pins. That
   test fails a rule no violation reports or with no near miss; it checks per
   rule, so an `ALLOWED` case that matches no rule still passes;
4. a row in the README rule table. `test_every_rule_is_documented` catches a
   missing one; it exists because the README's prose had already fallen two
   rules behind `RULES` with the whole suite green.

Two properties are load-bearing and easy to break:

- Rules match a **masked** copy of the text: string literals are blanked for
  every rule, comments too for all but the suppression rule. Without that, a
  comment describing a rule is blocked by the rule it describes. Strings and
  comments are found in one pass, so a quote inside a comment opens no string:
  masking strings first let an apostrophe and a later quote in one comment
  hide the text between them, suppressions included.
- The masking alternation must stay **linear**. An earlier form let two branches
  both match a backslash, and an unterminated quote took 6.5s at 8 lines and
  never finished at 12 — the sweep hanging rather than failing.
  `test_masking_is_linear_on_pathological_input` is the guard.
