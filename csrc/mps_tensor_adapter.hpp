#pragma once

// PyTorch MPS tensor -> CPU-visible pointer into its (shared-storage) MTLBuffer, plus the device-wide synchronization used before TBCCL
// reads or writes it. Compiled only on macOS (TORCH_TBCCL_WITH_MPS); see docs/phase72_mps_architecture.md.

#include <ATen/ATen.h>

namespace torch_tbccl
{

// `t` must be a non-empty, contiguous MPS tensor. Returns `[MTLBuffer contents] + storage_offset * element_size` after checking that the buffer is
// MTLStorageModeShared (a Private buffer has no CPU-visible contents and is rejected, never copied) and that the tensor's byte range lies inside it.
void *mps_host_pointer(const at::Tensor &t);

// Waits for all work PyTorch has queued on the MPS device (what torch.mps.synchronize() does). Coarse by design: it is global to the device, not to the tensor.
void mps_synchronize();

} // namespace torch_tbccl
