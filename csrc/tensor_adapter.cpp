#include "tensor_adapter.hpp"

#include <c10/util/Exception.h>

#ifdef TORCH_TBCCL_WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace torch_tbccl
{

tbccl::DataType to_tbccl_dtype(at::ScalarType type)
{
    switch (type)
    {
    case at::kFloat: return tbccl::DataType::Float32;
    default:
        TORCH_CHECK_NOT_IMPLEMENTED(
            false, "torch-tbccl: unsupported operation: dtype ", type, " (Phase 42 supports torch.float32 only)");
    }
}

tbccl::ReduceOp to_tbccl_reduce_op(const c10d::ReduceOp &op)
{
    switch (op.op_)
    {
    case c10d::ReduceOp::SUM: return tbccl::ReduceOp::Sum;
    default:
        TORCH_CHECK_NOT_IMPLEMENTED(false, "torch-tbccl: unsupported operation: only ReduceOp.SUM is supported (Phase 42)");
    }
}

TbcclBuffer to_tbccl_buffer(const at::Tensor &t, bool bytes_only)
{
    TORCH_CHECK_VALUE(t.defined(), "torch-tbccl: invalid argument: undefined tensor");
    TORCH_CHECK_NOT_IMPLEMENTED(
        t.layout() == at::kStrided && !t.is_quantized(),
        "torch-tbccl: unsupported operation: only dense strided tensors are supported (got layout ", t.layout(),
        t.is_quantized() ? ", quantized" : "", ")");
    const bool is_cuda = t.device().type() == at::kCUDA;
#ifdef TORCH_TBCCL_WITH_CUDA
    const bool device_ok = t.device().type() == at::kCPU || is_cuda;
#else
    const bool device_ok = t.device().type() == at::kCPU;
#endif
    TORCH_CHECK_NOT_IMPLEMENTED(
        device_ok, "torch-tbccl: unsupported operation: device ", t.device(),
        is_cuda ? " (this build has no CUDA support)" : " (supported: CPU, CUDA)");

    TbcclBuffer out;
    if (!bytes_only) out.datatype = to_tbccl_dtype(t.scalar_type());
    // No silent .contiguous(): that would copy and break in-place/async semantics.
    TORCH_CHECK_VALUE(
        t.is_contiguous(), "torch-tbccl: invalid argument: tensor must be contiguous (no implicit copy is made)");

    out.count = static_cast<std::size_t>(t.numel());
    out.view.data = t.numel() == 0 ? nullptr : t.data_ptr();
    out.view.bytes = static_cast<std::size_t>(t.nbytes());
    if (!is_cuda)
    {
        out.view.memory_kind = tbccl::MemoryKind::Host;
        out.view.device_ordinal = -1;
        out.context = tbccl::ExecutionContext{tbccl::ExecutionContextKind::Host, nullptr};
        return out;
    }
#ifdef TORCH_TBCCL_WITH_CUDA
    // The producer stream is whatever PyTorch's current stream for this device is at submission;
    // TBCCL owns the cross-stream event dependency.
    out.view.memory_kind = tbccl::MemoryKind::Cuda;
    out.view.device_ordinal = t.device().index();
    out.context = tbccl::ExecutionContext{
        tbccl::ExecutionContextKind::CudaStream,
        reinterpret_cast<void *>(c10::cuda::getCurrentCUDAStream(t.device().index()).stream())};
#endif
    return out;
}

} // namespace torch_tbccl
