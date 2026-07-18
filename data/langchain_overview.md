# The LangChain Ecosystem

LangChain Inc. maintains a layered stack of open-source tools for building
applications on top of large language models. Each layer builds on the one below
it, so you can start simple and reach for more power only when a task demands it.

## LangChain — the framework

LangChain is the bottom layer: a provider-agnostic framework that gives you
common abstractions for models, prompts, tools, embeddings, vector stores, and
retrievers. Because the interfaces are uniform, swapping one embedding model or
one vector store for another is usually a one-line change. LangChain is the
easiest place to start and is well suited to single-purpose agents, retrieval
pipelines, and simple prompt chains that have no long agent loop.

## LangGraph — the runtime

LangGraph is the middle layer: a low-level orchestration runtime for durable
execution, custom control flow, and stateful workflows. You describe an
application as a graph of nodes and edges, which makes it a good fit when control
flow is conditional, iterative, or parallel, or when state must survive failures
and span long sessions. LangChain agents run on top of LangGraph, so you get its
durability without writing graph code yourself.

## Deep Agents — the harness

Deep Agents is the top layer: a batteries-included harness built on LangChain and
LangGraph. It ships with planning, file management, subagent delegation, and
persistent memory out of the box, which makes it a good fit for long-running
tasks that need to decompose work and manage large context across a session.

## LangSmith — observability

LangSmith is a cross-cutting observability and evaluation platform. It is
framework-agnostic and is recommended alongside any of the layers above. Tracing
is enabled with the environment variables LANGSMITH_API_KEY, LANGSMITH_TRACING,
and LANGSMITH_PROJECT.

## Choosing a layer

A good rule of thumb: use plain LangChain for a fixed-tool agent or a retrieval
pipeline, reach for LangGraph when you need deterministic loops or branching, and
choose Deep Agents when you want planning, memory, and subagents without building
them yourself. The layers can also be combined — for example, a Deep Agents
orchestrator can delegate a subtask to a deterministic LangGraph subgraph.
