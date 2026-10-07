# Related projects

torch-tbccl is an adapter over TBCCL. The projects below are the runtime it needs and the projects built on top of it or beside it; each has its own documentation, version and release schedule.

| Project | Role | Source | Planned release |
|---|---|---|---|
| [tbccl](https://tbccl.tensorsofthewall.com/en/stable/) | The runtime: C++ library, stable C ABI, TCP transport and collectives. | [source](https://github.com/tensorsofthewall/tbccl) | 0.6.0 (unreleased) |
| [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl) | A vLLM platform plugin that routes communication through torch-tbccl. | [source](https://github.com/tensorsofthewall/vllm-tbccl) | 0.2.0 (unreleased) |
| [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl) | A pipeline-parallel data plane for exo over the TBCCL C ABI. | [source](https://github.com/tensorsofthewall/exo-tbccl) | 0.3.0 (unreleased) |

The core documentation is hosted at the stable site linked above once TBCCL's first release is tagged. The adapters' hosting is decided with their first releases; until then each project's documentation is built from its repository with `make docs` (see {doc}`development/building-docs`). The planned releases are unreleased targets, recorded in each project's `compatibility.json`.
