# Work lifetime, threads and the GIL

`BufferView` is non-owning, so every `WorkTBCCL` retains strong references to all tensors involved until the TBCCL operation completes. The adapter never makes an implicit `.contiguous()` copy or any other hidden copy; a non-contiguous tensor is rejected.

Each process group has one completion worker thread. It waits for submitted operations in order and completes their futures, and it never touches the Python API. It also never **destroys** a finished work state: the tensors it retains may hold the last reference to a live Python object, and dropping that reference takes the GIL, while interpreter shutdown joins the worker thread holding the GIL. Finished states are therefore retired to a list that the submitting thread (`reap()`) or the group destructor destroys, and the destructor releases the GIL while it joins the worker. This prevents a process that exits with a live group from hanging in interpreter finalization.

`Work.wait(timeout)` raises on expiry but does not cancel the operation.
