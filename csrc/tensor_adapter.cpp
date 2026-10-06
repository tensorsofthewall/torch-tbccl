#include "tensor_adapter.hpp"

#include <c10/util/Exception.h>

#ifdef TORCH_TBCCL_WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#endif
#ifdef TORCH_TBCCL_WITH_MPS
#include "mps_tensor_adapter.hpp"
#endif

#if defined(TORCH_TBCCL_WITH_CUDA)
#define TORCH_TBCCL_DEVICE_LIST "CPU, CUDA"
#elif defined(TORCH_TBCCL_WITH_MPS)
#define TORCH_TBCCL_DEVICE_LIST "CPU, MPS"
#else
#define TORCH_TBCCL_DEVICE_LIST "CPU"
#endif

namespace torch_tbccl
{

tbccl::DataType to_tbccl_dtype(at::ScalarType type)
{
    switch (type)
    {
    case at::kFloat: return tbccl::DataType::Float32;
    case at::kDouble: return tbccl::DataType::Float64;
    case at::kHalf: return tbccl::DataType::Float16;
    case at::kBFloat16: return tbccl::DataType::BFloat16;
    case at::kInt: return tbccl::DataType::Int32;
    case at::kLong: return tbccl::DataType::Int64;
    case at::kChar: return tbccl::DataType::Int8;
    case at::kByte: return tbccl::DataType::UInt8;
    default:
        TORCH_CHECK_NOT_IMPLEMENTED(
            false, "torch-tbccl: unsupported operation: dtype ", type,
            " has no reduction (all_reduce supports float16, bfloat16, float32, float64, int8, uint8, int32, int64; send/recv, broadcast and all_gather "
            "move any dense dtype as raw bytes)");
    }
}

const char *reduce_op_label(const c10d::ReduceOp &op)
{
    switch (op.op_)
    {
    case c10d::ReduceOp::SUM: return "SUM";
    case c10d::ReduceOp::PRODUCT: return "PRODUCT";
    case c10d::ReduceOp::MIN: return "MIN";
    case c10d::ReduceOp::MAX: return "MAX";
    case c10d::ReduceOp::BAND: return "BAND";
    case c10d::ReduceOp::BOR: return "BOR";
    case c10d::ReduceOp::BXOR: return "BXOR";
    case c10d::ReduceOp::PREMUL_SUM: return "PREMUL_SUM";
    case c10d::ReduceOp::AVG: return "AVG";
    default: return "unknown";
    }
}

tbccl::ReduceOp to_tbccl_reduce_op(const c10d::ReduceOp &op, tbccl::DataType datatype)
{
    switch (op.op_)
    {
    case c10d::ReduceOp::SUM: return tbccl::ReduceOp::Sum;
    default:
        TORCH_CHECK_NOT_IMPLEMENTED(
            false, "torch-tbccl: unsupported operation: unsupported reduction dtype=", tbccl::datatype_label(datatype), " op=", reduce_op_label(op),
            " (supported ops: SUM)");
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
    const bool is_mps = t.device().type() == at::kMPS;
    bool device_ok = t.device().type() == at::kCPU;
#ifdef TORCH_TBCCL_WITH_CUDA
    device_ok = device_ok || is_cuda;
#endif
#ifdef TORCH_TBCCL_WITH_MPS
    device_ok = device_ok || is_mps;
#endif
    TORCH_CHECK_NOT_IMPLEMENTED(
        device_ok, "torch-tbccl: unsupported operation: device ", t.device(),
        is_cuda  ? " (this build has no CUDA support)"
        : is_mps ? " (this build has no MPS support)"
                 : " (supported on this build: " TORCH_TBCCL_DEVICE_LIST ")");

    TbcclBuffer out;
    if (!bytes_only) out.datatype = to_tbccl_dtype(t.scalar_type());
    // No silent .contiguous(): that would copy and break in-place/async semantics.
    TORCH_CHECK_VALUE(
        t.is_contiguous(), "torch-tbccl: invalid argument: tensor must be contiguous (no implicit copy is made)");

    out.count = static_cast<std::size_t>(t.numel());
    out.view.bytes = static_cast<std::size_t>(t.nbytes());
#ifdef TORCH_TBCCL_WITH_MPS
    if (is_mps)
    {
        // An MPS data_ptr() is the MTLBuffer handle plus a byte offset, not an address. Shared storage is ordinary CPU-visible memory to TBCCL (MemoryKind::MetalShared,
        // the same provider as Host). Everything PyTorch queued on the device is finished first (device-wide), so TBCCL never reads a half-written producer and never
        // races a queued kernel that still writes a receive destination.
        out.view.data = nullptr;
        if (t.numel() > 0)
        {
            mps_synchronize();
            out.view.data = mps_host_pointer(t);
        }
        out.view.memory_kind = tbccl::MemoryKind::MetalShared;
        out.view.device_ordinal = -1;
        out.context = tbccl::ExecutionContext{tbccl::ExecutionContextKind::Host, nullptr};
        return out;
    }
#endif
    out.view.data = t.numel() == 0 ? nullptr : t.data_ptr();
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
