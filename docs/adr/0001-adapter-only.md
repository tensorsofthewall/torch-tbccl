# ADR 0001: torch-tbccl is an adapter over an installed TBCCL

- Status: accepted
- Date: 2026-10-01

## Context

TBCCL is framework-neutral and must stay so; PyTorch needs a `torch.distributed` backend.

## Decision

torch-tbccl is an out-of-tree c10d backend that only maps PyTorch concepts onto TBCCL's public API (store to endpoint exchange, tensor to buffer view, dtype and reduce op to TBCCL enums, current CUDA stream to execution context, TBCCL work to c10d work). It contains no communication algorithm, TCP code, chunk scheduling, device staging or local reduction, never includes TBCCL sources, links only an installed prefix, and never patches TBCCL or PyTorch. A defect that torch-tbccl exposes in TBCCL is fixed in TBCCL as a generic change. The same rule keeps adapter-level workarounds temporary: the guard that once refused overlapping collective and point-to-point calls was removed when TBCCL guaranteed independent ordering domains.

## Consequences

- The wheel is tied to the torch minor series and to a TBCCL C ABI and wire protocol range, both checked at build and import.
- After a TBCCL wire protocol change the wheel must be rebuilt against the new prefix.
