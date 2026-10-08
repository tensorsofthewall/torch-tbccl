# Validated configurations

Configurations that were exercised, not a benchmark report. Evidence is in the [validation report](../validation/0.2.0.md). "Release-candidate wheels" means the published 0.2.0rc1 wheels were used. Versions and platforms are in the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html).

| Configuration | What was run | Result |
|---|---|---|
| Linux, CPU, loopback (several processes) | operations x dtypes x world sizes 1 to 4 through the public `torch.distributed` API; DDP at world sizes 2 and 3; mixed point-to-point and collectives | passed |
| Linux, CUDA, loopback | CUDA tensors and mixed CUDA/CPU ranks, DDP, mixed traffic | passed |
| macOS, CPU and MPS, loopback | MPS operation matrix at world sizes 1 and 2, MPS DDP with a CPU side channel | passed |
| Linux CUDA to macOS MPS over Thunderbolt 4 | release-candidate wheels; point-to-point, mixed traffic, `all_reduce`, `broadcast`, `all_gather`, `barrier`; both rank orientations | passed with no data mismatch and unchanged link health |
| DDP between a Linux CUDA rank and a macOS MPS rank over Thunderbolt 4 | the development tree (10 training steps, repeated cycles, with a concurrent point-to-point thread); not repeated with the release-candidate wheels | passed on the development tree |
| Failure and lifecycle | peer exit, abort, destroy with pending work, interpreter shutdown with un-waited work (loopback); nine failure cases with a Linux CUDA survivor and a Mac peer, with the release-candidate wheels | bounded and reported with `torch-tbccl:` errors |
| Two Linux hosts over a physical network | not part of the recorded validation | not claimed |
