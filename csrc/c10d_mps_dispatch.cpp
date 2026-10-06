// Out-of-tree c10d dispatcher kernels for the MPS dispatch key.
//
// PyTorch 2.13 registers the c10d operators for CPU, CUDA and PrivateUse1 only (Ops.cpp), so a collective on an MPS tensor fails in the dispatcher before any
// Backend is consulted. Each kernel here mirrors what the CPU/CUDA kernel does for the operators the tbccl backend supports:
// unwrap the ProcessGroup, getBackend(DeviceType::MPS), call the Backend method, wrap the Work. Registration is explicit (not a static initializer) so it can be
// gated: it is done only on the torch series the schemas were verified against, refuses an operator whose schema differs from the expected text, and skips one
// that already has an MPS kernel (a future PyTorch that supports MPS natively).

#ifdef TORCH_TBCCL_WITH_MPS

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/library.h>
#include <torch/version.h>

#include <ATen/core/dispatch/Dispatcher.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace torch_tbccl
{
namespace
{

using PG = c10::intrusive_ptr<c10d::ProcessGroup>;
using WorkPtr = c10::intrusive_ptr<c10d::Work>;

c10::intrusive_ptr<c10d::Backend> mps_backend(const PG &pg)
{
    return pg->getBackend(c10::DeviceType::MPS);
}

WorkPtr send_mps(at::TensorList tensors, const PG &pg, int64_t dst, int64_t tag)
{
    auto v = tensors.vec();
    return mps_backend(pg)->send(v, static_cast<int>(dst), static_cast<int>(tag));
}

WorkPtr recv_mps(at::TensorList tensors, const PG &pg, int64_t src, int64_t tag)
{
    auto v = tensors.vec();
    return mps_backend(pg)->recv(v, static_cast<int>(src), static_cast<int>(tag));
}

std::tuple<std::vector<at::Tensor>, WorkPtr> broadcast_mps(
    at::TensorList tensors, const PG &pg, int64_t root_rank, int64_t root_tensor, bool async_op, int64_t timeout)
{
    auto v = tensors.vec();
    c10d::BroadcastOptions opts{root_rank, root_tensor, std::chrono::milliseconds(timeout), async_op};
    auto work = mps_backend(pg)->broadcast(v, opts);
    return {std::move(v), work};
}

std::tuple<std::vector<at::Tensor>, WorkPtr> allreduce_mps(
    at::TensorList tensors, const PG &pg, const c10::intrusive_ptr<c10d::ReduceOp> &reduce_op, const std::optional<at::Tensor> &sparse_indices,
    bool async_op, int64_t timeout)
{
    auto v = tensors.vec();
    c10d::AllreduceOptions opts;
    opts.reduceOp = *reduce_op;
    opts.sparseIndices = sparse_indices;
    opts.timeout = std::chrono::milliseconds(timeout);
    opts.asyncOp = async_op;
    auto work = mps_backend(pg)->allreduce(v, opts);
    return {std::move(v), work};
}

std::tuple<std::vector<std::vector<at::Tensor>>, WorkPtr> allgather_mps(
    const std::vector<std::vector<at::Tensor>> &output_tensors, at::TensorList input_tensors, const PG &pg, bool async_op, int64_t timeout)
{
    auto in = input_tensors.vec();
    auto out = output_tensors;
    c10d::AllgatherOptions opts;
    opts.timeout = std::chrono::milliseconds(timeout);
    opts.asyncOp = async_op;
    auto work = mps_backend(pg)->allgather(out, in, opts);
    return {std::move(out), work};
}

WorkPtr gather_mps(
    const std::vector<std::vector<at::Tensor>> &output_tensors, const at::TensorList &input_tensors, const PG &pg, int64_t root_rank, bool async_op,
    int64_t timeout)
{
    auto in = input_tensors.vec();
    auto out = output_tensors;
    c10d::GatherOptions opts;
    opts.rootRank = root_rank;
    opts.timeout = std::chrono::milliseconds(timeout);
    opts.asyncOp = async_op;
    return mps_backend(pg)->gather(out, in, opts);
}

WorkPtr barrier_mps(at::Tensor tensor, const PG &pg, const std::vector<int64_t> &device_ids, bool async_op, int64_t timeout)
{
    c10d::BarrierOptions opts;
    opts.device_ids = device_ids;
    opts.timeout = std::chrono::milliseconds(timeout);
    opts.device = tensor.device();
    opts.asyncOp = async_op;
    return mps_backend(pg)->barrier(opts);
}

struct Expected
{
    const char *name;
    const char *schema;
};

// The 2.13.0 schemas, as printed by the dispatcher (torch._C._dispatch_dump).
constexpr Expected kExpected[] = {
    {"c10d::send", "c10d::send(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int dst, int tag) -> __torch__.torch.classes.c10d.Work"},
    {"c10d::recv_", "c10d::recv_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int src, int tag) -> __torch__.torch.classes.c10d.Work"},
    {"c10d::broadcast_",
     "c10d::broadcast_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, int root_tensor, bool async_op=True, int "
     "timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)"},
    {"c10d::allreduce_",
     "c10d::allreduce_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, Tensor? "
     "sparse_indices, bool async_op=True, int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)"},
    {"c10d::allgather_",
     "c10d::allgather_(Tensor[][] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True, int "
     "timeout=-1) -> (Tensor[][], __torch__.torch.classes.c10d.Work)"},
    {"c10d::gather_",
     "c10d::gather_(Tensor[][] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, bool "
     "async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work"},
    {"c10d::barrier",
     "c10d::barrier(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, int[] device_ids, bool async_op=True, int timeout=-1) -> "
     "__torch__.torch.classes.c10d.Work"},
};

std::unique_ptr<torch::Library> g_library; // kept for the life of the process: destroying it unregisters the kernels

} // namespace

// Returns "" on success, "upstream" when PyTorch already provides MPS kernels (nothing registered), else throws with the reason.
std::string register_mps_dispatch()
{
    if (g_library) return "";
    TORCH_CHECK(
        TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 13,
        "torch-tbccl MPS support was built for PyTorch 2.13 c10d schemas; running torch ", TORCH_VERSION_MAJOR, ".", TORCH_VERSION_MINOR, " is unsupported");
    auto &dispatcher = c10::Dispatcher::singleton();
    std::size_t already = 0;
    for (const auto &e : kExpected)
    {
        const auto op = dispatcher.findSchema(c10::OperatorName(std::string(e.name), ""));
        TORCH_CHECK(
            op.has_value(), "torch-tbccl MPS support was built for PyTorch 2.13 c10d schemas; ", e.name, " is not registered in torch ", TORCH_VERSION_MAJOR, ".",
            TORCH_VERSION_MINOR);
        std::ostringstream actual;
        actual << op->schema();
        TORCH_CHECK(
            actual.str() == e.schema, "torch-tbccl MPS support was built for PyTorch 2.13 c10d schemas; the schema of ", e.name, " differs:\n  expected ", e.schema,
            "\n  found    ", actual.str());
        if (op->hasKernelForDispatchKey(c10::DispatchKey::MPS)) ++already;
    }
    if (already == std::size(kExpected)) return "upstream";
    TORCH_CHECK(already == 0, "torch-tbccl: PyTorch already registers MPS kernels for ", already, " of the ", std::size(kExpected), " c10d operators; refusing to mix");

    auto lib = std::make_unique<torch::Library>(torch::Library::IMPL, "c10d", c10::DispatchKey::MPS, __FILE__, __LINE__);
    lib->impl("send", TORCH_FN(send_mps));
    lib->impl("recv_", TORCH_FN(recv_mps));
    lib->impl("broadcast_", TORCH_FN(broadcast_mps));
    lib->impl("allreduce_", TORCH_FN(allreduce_mps));
    lib->impl("allgather_", TORCH_FN(allgather_mps));
    lib->impl("gather_", TORCH_FN(gather_mps));
    lib->impl("barrier", TORCH_FN(barrier_mps));
    g_library = std::move(lib);
    return "";
}

} // namespace torch_tbccl

#endif // TORCH_TBCCL_WITH_MPS
