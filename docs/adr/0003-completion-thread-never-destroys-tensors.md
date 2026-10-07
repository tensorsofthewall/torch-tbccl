# ADR 0003: The completion thread never destroys tensors

- Status: accepted
- Date: 2026-10-06

## Context

A process that exited with a live process group hung in interpreter finalization. A tensor retained by a work state can be the last reference to a Python object; dropping it takes the GIL. The garbage collector destroys the group at interpreter exit while holding the GIL and joins the completion thread, so the two deadlock if the completion thread destroys a tensor.

## Decision

The completion worker only completes futures. Finished work states are retired to a list; the submitting thread (`reap()`) or the group destructor destroys them, and the destructor releases the GIL while it joins the worker.

## Consequences

- Exiting with a live group, or with an un-waited asynchronous operation, no longer hangs (tested repeatedly).
- Any new code that touches tensors must run on the submitting thread or under the GIL, never on the completion worker.
