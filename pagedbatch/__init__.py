"""pagedbatch: a minimal continuous-batching LLM inference engine with a paged KV cache.

The package is deliberately small. Read it in this order:

- ``block_manager``: physical KV blocks, the free list and per-sequence block tables.
- ``sequence``: the lifecycle of one request.
- ``scheduler``: which tokens of which sequences run in the next step, and what
  gets preempted when the cache is full.
- ``model``: a Llama-architecture transformer whose attention reads and writes the
  paged cache.
- ``engine``: ties the above together: one ``step()`` is one model forward.
- ``server``: an OpenAI-compatible HTTP front end over the engine.
"""

__version__ = "0.1.0"
