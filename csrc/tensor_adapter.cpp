#include "tensor_adapter.hpp"

#include <c10/util/Exception.h>

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

TbcclBuffer to_tbccl_buffer(const at::Tensor &t)
{
    TORCH_CHECK_VALUE(t.defined(), "torch-tbccl: invalid argument: undefined tensor");
    TORCH_CHECK_NOT_IMPLEMENTED(
        t.layout() == at::kStrided && !t.is_quantized(),
        "torch-tbccl: unsupported operation: only dense strided tensors are supported (got layout ", t.layout(),
        t.is_quantized() ? ", quantized" : "", ")");
    TORCH_CHECK_NOT_IMPLEMENTED(
        t.device().type() == at::kCPU,
        "torch-tbccl: unsupported operation: device ", t.device(), " (CPU tensors only in this build step)");

    TbcclBuffer out;
    out.datatype = to_tbccl_dtype(t.scalar_type());
    // No silent .contiguous(): that would copy and break in-place/async semantics.
    TORCH_CHECK_VALUE(
        t.is_contiguous(), "torch-tbccl: invalid argument: tensor must be contiguous (no implicit copy is made)");

    out.count = static_cast<std::size_t>(t.numel());
    out.view.memory_kind = tbccl::MemoryKind::Host;
    out.view.data = t.numel() == 0 ? nullptr : t.data_ptr();
    out.view.bytes = static_cast<std::size_t>(t.nbytes());
    out.view.device_ordinal = -1;
    out.context = tbccl::ExecutionContext{tbccl::ExecutionContextKind::Host, nullptr};
    return out;
}

} // namespace torch_tbccl
