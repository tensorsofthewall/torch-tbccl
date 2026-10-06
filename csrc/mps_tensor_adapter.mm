// Objective-C++ half of the MPS tensor adapter: the only place that touches Metal objects. PyTorch's MPS allocator hands out one MTLBuffer per
// allocation and stores the id<MTLBuffer> itself in the storage's data pointer (it is NOT a CPU address); a view's data_ptr() is that handle plus
// the byte offset. Built without ARC: the handle is borrowed from the storage, which the caller (WorkState) keeps alive.

#include "mps_tensor_adapter.hpp"

#import <Metal/Metal.h>

#include <ATen/detail/MPSHooksInterface.h>
#include <c10/util/Exception.h>

namespace torch_tbccl
{

void *mps_host_pointer(const at::Tensor &t)
{
    TORCH_INTERNAL_ASSERT(t.device().type() == at::kMPS && t.numel() > 0);
    id<MTLBuffer> buffer = (id<MTLBuffer>)t.storage().mutable_data();
    TORCH_CHECK_NOT_IMPLEMENTED(buffer != nil, "torch-tbccl: unsupported operation: MPS tensor has no Metal buffer");
    TORCH_CHECK_NOT_IMPLEMENTED(
        [buffer storageMode] == MTLStorageModeShared,
        "torch-tbccl: unsupported operation: MPS tensor is not in shared storage (MTLStorageMode ", static_cast<int>([buffer storageMode]),
        "); only MTLStorageModeShared buffers are CPU-visible and no hidden copy is made");
    char *base = static_cast<char *>([buffer contents]);
    TORCH_CHECK_NOT_IMPLEMENTED(base != nullptr, "torch-tbccl: unsupported operation: the MPS tensor's Metal buffer has no CPU-visible contents");
    const std::size_t offset = static_cast<std::size_t>(t.storage_offset()) * t.element_size();
    TORCH_INTERNAL_ASSERT(
        offset + t.nbytes() <= [buffer length], "torch-tbccl: internal error: MPS tensor range (offset ", offset, ", ", t.nbytes(),
        " bytes) exceeds its Metal buffer (", [buffer length], " bytes)");
    return base + offset;
}

void mps_synchronize()
{
    at::detail::getMPSHooks().deviceSynchronize();
}

} // namespace torch_tbccl
