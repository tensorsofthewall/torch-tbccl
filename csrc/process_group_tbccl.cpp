#include "process_group_tbccl.hpp"

#include "bootstrap.hpp"
#include "errors.hpp"
#include "tensor_adapter.hpp"
#include "trace.hpp"
#include "work_tbccl.hpp"

#include <Python.h>

#include <c10/util/Exception.h>

#include <tbccl/cuda_support.hpp>

#include <cstdlib>
#include <exception>
#include <mutex>

namespace torch_tbccl
{

namespace
{
void ensure_cuda_support()
{
#ifdef TORCH_TBCCL_WITH_CUDA
    static std::once_flag once;
    std::call_once(once, [] { tbccl::register_cuda_support(); });
#endif
}

// Joining the completion thread must not hold the GIL (the thread may be waiting for it inside a Future callback). The destructor of a process group
// that Python is garbage-collecting at interpreter exit runs with the GIL held, hence the explicit release.
class ReleaseGil
{
public:
    ReleaseGil()
    {
        if (Py_IsInitialized() && PyGILState_Check()) state_ = PyEval_SaveThread();
    }
    ~ReleaseGil()
    {
        if (state_) PyEval_RestoreThread(state_);
    }
    ReleaseGil(const ReleaseGil &) = delete;
    ReleaseGil &operator=(const ReleaseGil &) = delete;

private:
    PyThreadState *state_ = nullptr;
};

std::shared_ptr<TraceRecord> begin_trace(const char *op, const at::Tensor &t, std::uint64_t entry_ns)
{
    auto r = trace_begin(op, static_cast<std::uint64_t>(t.nbytes()), t.device().str());
    if (r) r->entry_ns = entry_ns;
    return r;
}
} // namespace

ProcessGroupTBCCL::ProcessGroupTBCCL(
    const c10::intrusive_ptr<c10d::Store> &store,
    int rank,
    int world_size,
    std::chrono::milliseconds timeout)
    : c10d::Backend(rank, world_size), store_(store), timeout_(timeout)
{
    // Fail at creation, not at the first collective.
    TORCH_CHECK_NOT_IMPLEMENTED(
        world_size >= 1 && static_cast<std::size_t>(world_size) <= tbccl::kMaxFullMeshWorldSize,
        "torch-tbccl supports world_size 1 to ", tbccl::kMaxFullMeshWorldSize, " (got ", world_size, "); validated at 2, 3 and 4");
    TORCH_CHECK_VALUE(rank >= 0 && rank < world_size, "torch-tbccl: invalid argument: rank ", rank, " outside [0, ", world_size, ")");
    TORCH_CHECK_VALUE(store_ != nullptr, "torch-tbccl: invalid argument: store is null");

    ensure_cuda_support();
    if (const char *v = std::getenv("TORCH_TBCCL_FORCE_SYNC_ALLREDUCE")) force_sync_allreduce_ = v[0] != '\0' && v[0] != '0';

    // A one-rank group (e.g. vLLM's TP=1 groups) has no peer and needs no communicator; collectives on it are rejected.
    if (world_size == 1) return;

    // Validate locally before touching the Store or the network.
    const auto local = local_endpoint_from_env();
    auto options = bootstrap_options(store_, rank, world_size, timeout, local);
    try
    {
        comm_ = tbccl::Communicator::create(options);
    }
    catch (const std::exception &e)
    {
        TORCH_CHECK(false, "torch-tbccl: communicator failure: could not create TBCCL communicator: ", e.what());
    }
    completion_ = std::make_unique<CompletionWorker>();
}

ProcessGroupTBCCL::~ProcessGroupTBCCL()
{
    shutdown();
}

void ProcessGroupTBCCL::shutdown()
{
    // Order: reject new submissions, destroy the communicator (it aborts any
    // still-outstanding operation, so a silent peer cannot block teardown and
    // every pending Work settles), then join the completion worker, which
    // finishes the remaining Futures.
    std::unique_ptr<tbccl::Communicator> doomed;
    std::unique_ptr<CompletionWorker> worker;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        doomed = std::move(comm_);
        worker = std::move(completion_);
    }
    doomed.reset();
    if (worker)
    {
        {
            ReleaseGil nogil;
            worker->stop();
        }
        worker.reset(); // the finished states (and their tensors) are destroyed here, on a thread that may take the GIL
    }
}

void ProcessGroupTBCCL::reap()
{
    std::vector<std::shared_ptr<WorkState>> finished;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (completion_) finished = completion_->take_retired();
    }
} // `finished` dies here: no lock held, on the submitting thread

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::allreduce(
    std::vector<at::Tensor> &tensors, const c10d::AllreduceOptions &opts)
{
    reap();
    const auto entry_ns = trace_enabled() ? trace_now_ns() : 0;
    TORCH_CHECK_VALUE(
        tensors.size() == 1,
        "torch-tbccl: invalid argument: allreduce takes exactly one tensor (got ", tensors.size(), ")");
    TORCH_CHECK_NOT_IMPLEMENTED(
        !opts.sparseIndices.has_value(), "torch-tbccl: unsupported operation: sparse allreduce");
    const auto buf = to_tbccl_buffer(tensors[0]);
    const auto op = to_tbccl_reduce_op(opts.reduceOp, buf.datatype);

    auto state = std::make_shared<WorkState>();
    state->tensors = tensors;
    state->op_name = "allreduce";
    state->trace = begin_trace("allreduce", tensors[0], entry_ns);
    state->future = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));

    if (buf.count == 0)
    {
        // Zero elements: nothing to reduce. Both ranks see the same count,
        // so skipping the exchange on both sides stays consistent.
        state->future->markCompleted(c10::IValue(state->tensors));
        return c10::make_intrusive<WorkTBCCL>(getRank(), c10d::OpType::ALLREDUCE, state);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    require_peers(state->op_name.c_str());
    TORCH_CHECK(comm_ != nullptr, "torch-tbccl: communicator failure: process group has been shut down");
    try
    {
        // In place: same BufferView as send and receive; no intermediate tensor.
        tbccl::BufferView view = buf.view;
        // MPS shared storage is ordinary CPU-visible memory. libtbccl's MetalShared reduction table is narrower than Host's (four element types), so a dtype it
        // lacks is presented as Host: same pointer, same provider, no copy.
        if (view.memory_kind == tbccl::MemoryKind::MetalShared &&
            !comm_->capabilities().supports_collective_all_reduce(tbccl::MemoryKind::MetalShared, buf.datatype, op))
            view.memory_kind = tbccl::MemoryKind::Host;
        if (state->trace) state->trace->before_submit_ns = trace_now_ns();
        state->work = comm_->all_reduce(view, view, buf.count, buf.datatype, op, buf.context);
        if (state->trace) state->trace->return_ns = trace_now_ns();
        if (force_sync_allreduce_) state->work->wait();
    }
    catch (const std::runtime_error &e)
    {
        throw_tbccl_error("allreduce", e.what());
    }
    completion_->enqueue(state);
    return c10::make_intrusive<WorkTBCCL>(getRank(), c10d::OpType::ALLREDUCE, state);
}

void ProcessGroupTBCCL::abort()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (comm_) comm_->abort("ProcessGroup abort");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::finish(
    std::shared_ptr<WorkState> state, c10d::OpType op)
{
    // Trivially complete (zero bytes): both ranks skip the exchange consistently.
    state->future->markCompleted(c10::IValue(state->tensors));
    return c10::make_intrusive<WorkTBCCL>(getRank(), op, state);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::broadcast(
    std::vector<at::Tensor> &tensors, const c10d::BroadcastOptions &opts)
{
    reap();
    const auto entry_ns = trace_enabled() ? trace_now_ns() : 0;
    TORCH_CHECK_VALUE(
        tensors.size() == 1,
        "torch-tbccl: invalid argument: broadcast takes exactly one tensor (got ", tensors.size(), ")");
    TORCH_CHECK_VALUE(
        opts.rootRank >= 0 && opts.rootRank < getSize(),
        "torch-tbccl: invalid argument: broadcast rootRank ", opts.rootRank, " outside [0, ", getSize(), ")");
    TORCH_CHECK_NOT_IMPLEMENTED(
        opts.rootTensor == 0, "torch-tbccl: unsupported operation: broadcast rootTensor must be 0 (got ", opts.rootTensor, ")");
    const auto buf = to_tbccl_buffer(tensors[0], true);

    auto state = std::make_shared<WorkState>();
    state->tensors = tensors;
    state->op_name = "broadcast";
    state->trace = begin_trace("broadcast", tensors[0], entry_ns);
    state->future = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));
    if (buf.view.bytes == 0) return finish(state, c10d::OpType::BROADCAST);

    std::lock_guard<std::mutex> lock(mutex_);
    require_peers(state->op_name.c_str());
    TORCH_CHECK(comm_ != nullptr, "torch-tbccl: communicator failure: process group has been shut down");
    try
    {
        if (state->trace) state->trace->before_submit_ns = trace_now_ns();
        state->work = comm_->broadcast(buf.view, static_cast<std::size_t>(opts.rootRank), buf.context);
        if (state->trace) state->trace->return_ns = trace_now_ns();
    }
    catch (const std::runtime_error &e)
    {
        throw_tbccl_error("broadcast", e.what());
    }
    completion_->enqueue(state);
    return c10::make_intrusive<WorkTBCCL>(getRank(), c10d::OpType::BROADCAST, state);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::allgather(
    std::vector<std::vector<at::Tensor>> &outputTensors,
    std::vector<at::Tensor> &inputTensors,
    const c10d::AllgatherOptions &)
{
    reap();
    const auto entry_ns = trace_enabled() ? trace_now_ns() : 0;
    TORCH_CHECK_VALUE(
        inputTensors.size() == 1 && outputTensors.size() == 1,
        "torch-tbccl: invalid argument: allgather takes one input tensor and one output list (got ",
        inputTensors.size(), " inputs, ", outputTensors.size(), " output lists)");
    const auto &input = inputTensors[0];
    auto &outs = outputTensors[0];
    TORCH_CHECK_VALUE(
        static_cast<int>(outs.size()) == getSize(),
        "torch-tbccl: invalid argument: allgather output list must have world_size (", getSize(), ") tensors (got ",
        outs.size(), ")");

    const auto in_buf = to_tbccl_buffer(input, true);
    std::vector<tbccl::BufferView> views;
    for (const auto &o : outs)
    {
        const auto b = to_tbccl_buffer(o, true);
        TORCH_CHECK_VALUE(
            o.scalar_type() == input.scalar_type() && o.numel() == input.numel() && o.device() == input.device(),
            "torch-tbccl: invalid argument: allgather outputs must match the input's dtype, numel and device");
        views.push_back(b.view);
    }

    auto state = std::make_shared<WorkState>();
    state->tensors = outs;
    state->retained = {input};
    state->op_name = "allgather";
    state->trace = begin_trace("allgather", input, entry_ns);
    state->future = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));
    if (in_buf.view.bytes == 0) return finish(state, c10d::OpType::ALLGATHER);

    std::lock_guard<std::mutex> lock(mutex_);
    require_peers(state->op_name.c_str());
    TORCH_CHECK(comm_ != nullptr, "torch-tbccl: communicator failure: process group has been shut down");
    try
    {
        if (state->trace) state->trace->before_submit_ns = trace_now_ns();
        state->work = comm_->all_gather(in_buf.view, views, in_buf.context);
        if (state->trace) state->trace->return_ns = trace_now_ns();
    }
    catch (const std::runtime_error &e)
    {
        throw_tbccl_error("allgather", e.what());
    }
    completion_->enqueue(state);
    return c10::make_intrusive<WorkTBCCL>(getRank(), c10d::OpType::ALLGATHER, state);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::gather(
    std::vector<std::vector<at::Tensor>> &outputTensors,
    std::vector<at::Tensor> &inputTensors,
    const c10d::GatherOptions &opts)
{
    TORCH_CHECK_VALUE(
        inputTensors.size() == 1, "torch-tbccl: invalid argument: gather takes exactly one input tensor per rank");
    TORCH_CHECK_VALUE(
        opts.rootRank >= 0 && opts.rootRank < getSize(), "torch-tbccl: invalid argument: gather rootRank out of range");
    TORCH_CHECK_NOT_IMPLEMENTED(
        getSize() == 2, "torch-tbccl: unsupported operation: gather needs a 2-rank group (this group has ", getSize(), " ranks; rank ", getRank(), ")");
    if (getRank() != opts.rootRank) return p2p(inputTensors, opts.rootRank, true);
    TORCH_CHECK_VALUE(
        outputTensors.size() == 1 && static_cast<int>(outputTensors[0].size()) == getSize(),
        "torch-tbccl: invalid argument: gather on the root needs one output list of world_size tensors");
    auto &outs = outputTensors[0];
    const int peer = 1 - getRank();
    TORCH_CHECK_VALUE(
        outs[peer].scalar_type() == inputTensors[0].scalar_type() && outs[peer].numel() == inputTensors[0].numel() &&
            outs[getRank()].numel() == inputTensors[0].numel(),
        "torch-tbccl: invalid argument: gather outputs must match the input's dtype and numel");
    outs[getRank()].copy_(inputTensors[0]);
    std::vector<at::Tensor> slot{outs[peer]};
    return p2p(slot, peer, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::barrier(const c10d::BarrierOptions &)
{
    reap();
    std::vector<at::Tensor> token{at::zeros({1}, at::kFloat)};
    if (getSize() == 1)
    {
        auto state = std::make_shared<WorkState>();
        state->tensors = token;
        state->op_name = "barrier";
        state->future = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));
        return finish(state, c10d::OpType::BARRIER);
    }
    if (getSize() == 2) return allreduce(token);

    // More than two ranks: the communicator's own barrier (a descriptor exchange, no payload).
    auto state = std::make_shared<WorkState>();
    state->tensors = token;
    state->op_name = "barrier";
    state->future = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(comm_ != nullptr, "torch-tbccl: communicator failure: process group has been shut down");
    try
    {
        state->work = comm_->barrier();
    }
    catch (const std::runtime_error &e)
    {
        throw_tbccl_error("barrier", e.what());
    }
    completion_->enqueue(state);
    return c10::make_intrusive<WorkTBCCL>(getRank(), c10d::OpType::BARRIER, state);
}

#define TBCCL_UNSUPPORTED(name)                                                                                                     \
    TORCH_CHECK_NOT_IMPLEMENTED(                                                                                                    \
        false, "torch-tbccl: unsupported operation: ", name, " is not implemented by the tbccl backend (rank ", getRank(),          \
        "); supported: all_reduce (SUM), broadcast, all_gather, gather (2 ranks), barrier, send/recv")

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::allreduce_sparse(std::vector<at::Tensor> &, const c10d::AllreduceOptions &)
{
    TBCCL_UNSUPPORTED("allreduce_sparse");
}
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::allreduce_coalesced(std::vector<at::Tensor> &, const c10d::AllreduceCoalescedOptions &)
{
    TBCCL_UNSUPPORTED("allreduce_coalesced");
}
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::reduce(std::vector<at::Tensor> &, const c10d::ReduceOptions &)
{
    TBCCL_UNSUPPORTED("reduce");
}
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::_allgather_base(at::Tensor &, at::Tensor &, const c10d::AllgatherOptions &)
{
    TBCCL_UNSUPPORTED("all_gather_into_tensor");
}
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>> &, std::vector<at::Tensor> &, const c10d::AllgatherOptions &)
{
    TBCCL_UNSUPPORTED("all_gather_coalesced");
}
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor> &, std::vector<at::Tensor> &, const c10d::AllgatherOptions &)
{
    TBCCL_UNSUPPORTED("all_gather_into_tensor_coalesced");
}
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::scatter(
    std::vector<at::Tensor> &, std::vector<std::vector<at::Tensor>> &, const c10d::ScatterOptions &)
{
    TBCCL_UNSUPPORTED("scatter");
}
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::reduce_scatter(
    std::vector<at::Tensor> &, std::vector<std::vector<at::Tensor>> &, const c10d::ReduceScatterOptions &)
{
    TBCCL_UNSUPPORTED("reduce_scatter");
}
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::_reduce_scatter_base(at::Tensor &, at::Tensor &, const c10d::ReduceScatterOptions &)
{
    TBCCL_UNSUPPORTED("reduce_scatter_tensor");
}
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor> &, std::vector<at::Tensor> &, const c10d::ReduceScatterOptions &)
{
    TBCCL_UNSUPPORTED("reduce_scatter_tensor_coalesced");
}
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::alltoall_base(
    at::Tensor &, at::Tensor &, std::vector<int64_t> &, std::vector<int64_t> &, const c10d::AllToAllOptions &)
{
    TBCCL_UNSUPPORTED("all_to_all_single");
}
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::alltoall(std::vector<at::Tensor> &, std::vector<at::Tensor> &, const c10d::AllToAllOptions &)
{
    TBCCL_UNSUPPORTED("all_to_all");
}
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::recvAnysource(std::vector<at::Tensor> &, int)
{
    TBCCL_UNSUPPORTED("recv from any source");
}
#undef TBCCL_UNSUPPORTED

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::send(std::vector<at::Tensor> &tensors, int dstRank, int)
{
    return p2p(tensors, dstRank, true);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::recv(std::vector<at::Tensor> &tensors, int srcRank, int)
{
    return p2p(tensors, srcRank, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::p2p(std::vector<at::Tensor> &tensors, int peer, bool is_send)
{
    reap();
    const char *name = is_send ? "send" : "recv";
    const auto entry_ns = trace_enabled() ? trace_now_ns() : 0;
    TORCH_CHECK_VALUE(
        tensors.size() == 1, "torch-tbccl: invalid argument: ", name, " takes exactly one tensor (got ", tensors.size(), ")");
    TORCH_CHECK_VALUE(
        peer >= 0 && peer < getSize() && peer != getRank(),
        "torch-tbccl: invalid argument: ", name, " peer rank ", peer, " must be another rank of this group");
    const auto buf = to_tbccl_buffer(tensors[0], true);

    auto state = std::make_shared<WorkState>();
    state->tensors = tensors;
    state->op_name = name;
    state->trace = begin_trace(name, tensors[0], entry_ns);
    state->future = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));
    const auto op = is_send ? c10d::OpType::SEND : c10d::OpType::RECV;
    if (buf.view.bytes == 0) return finish(state, op);

    // The payload is opaque bytes: describe it to TBCCL as UInt8 elements, one per byte (the datatype and count are only a
    // size check; the transfer itself moves buffer.bytes), so no dtype - FP8, packed 4-bit, anything dense - is ever interpreted.
    const std::size_t count = buf.view.bytes;
    std::lock_guard<std::mutex> lock(mutex_);
    require_peers(state->op_name.c_str());
    TORCH_CHECK(comm_ != nullptr, "torch-tbccl: communicator failure: process group has been shut down");
    try
    {
        if (state->trace) state->trace->before_submit_ns = trace_now_ns();
        state->work = is_send ? comm_->send(buf.view, count, tbccl::DataType::UInt8, static_cast<std::size_t>(peer), buf.context)
                              : comm_->recv(buf.view, count, tbccl::DataType::UInt8, static_cast<std::size_t>(peer), buf.context);
        if (state->trace) state->trace->return_ns = trace_now_ns();
    }
    catch (const std::runtime_error &e)
    {
        throw_tbccl_error(name, e.what());
    }
    completion_->enqueue(state);
    return c10::make_intrusive<WorkTBCCL>(getRank(), op, state);
}

// TBCCL's collectives and point-to-point transfers are independent ordering domains over ONE connection per peer: submitted back to back from one thread in the
// same order on every rank, a collective and a P2P transfer that are in flight together can still interleave on that connection. tools/p71_mixed_inflight.py
// shows it: the operations complete "successfully" with millions of wrong elements, or fail with a framing error. libtbccl has no guard, so the adapter refuses the
// overlap instead of letting it corrupt data. Called under mutex_; finished operations are forgotten here, nothing is waited for.
void ProcessGroupTBCCL::require_peers(const char *op) const
{
    TORCH_CHECK_NOT_IMPLEMENTED(
        getSize() >= 2, "torch-tbccl: unsupported operation: ", op, " needs a group of at least 2 ranks (rank ", getRank(),
        " is alone in a one-rank group)");
}

bool ProcessGroupTBCCL::is_shutdown() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return comm_ == nullptr;
}

} // namespace torch_tbccl
