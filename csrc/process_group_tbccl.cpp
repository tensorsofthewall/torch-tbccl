#include "process_group_tbccl.hpp"

#include "bootstrap.hpp"
#include "errors.hpp"
#include "tensor_adapter.hpp"
#include "work_tbccl.hpp"

#include <c10/util/Exception.h>

#include <tbccl/cuda_support.hpp>

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
} // namespace

ProcessGroupTBCCL::ProcessGroupTBCCL(
    const c10::intrusive_ptr<c10d::Store> &store,
    int rank,
    int world_size,
    std::chrono::milliseconds timeout)
    : c10d::Backend(rank, world_size), store_(store), timeout_(timeout)
{
    // Fail at creation, not at the first collective.
    TORCH_CHECK_NOT_IMPLEMENTED(world_size == 2, "torch-tbccl Phase 42 supports world_size=2 only (got ", world_size, ")");
    TORCH_CHECK_VALUE(rank >= 0 && rank < world_size, "torch-tbccl: invalid argument: rank ", rank, " outside [0, ", world_size, ")");
    TORCH_CHECK_VALUE(store_ != nullptr, "torch-tbccl: invalid argument: store is null");

    ensure_cuda_support();

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
    // Order: reject new submissions, destroy the communicator (TBCCL drains
    // its queue, so every pending Work settles), then join the completion
    // worker, which finishes the remaining Futures.
    std::unique_ptr<tbccl::Communicator> doomed;
    std::unique_ptr<CompletionWorker> worker;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        doomed = std::move(comm_);
        worker = std::move(completion_);
    }
    doomed.reset();
    worker.reset();
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::allreduce(
    std::vector<at::Tensor> &tensors, const c10d::AllreduceOptions &opts)
{
    TORCH_CHECK_VALUE(
        tensors.size() == 1,
        "torch-tbccl: invalid argument: allreduce takes exactly one tensor (got ", tensors.size(), ")");
    TORCH_CHECK_NOT_IMPLEMENTED(
        !opts.sparseIndices.has_value(), "torch-tbccl: unsupported operation: sparse allreduce");
    const auto op = to_tbccl_reduce_op(opts.reduceOp);
    const auto buf = to_tbccl_buffer(tensors[0]);

    auto state = std::make_shared<WorkState>();
    state->tensors = tensors;
    state->op_name = "allreduce";
    state->future = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));

    if (buf.count == 0)
    {
        // Zero elements: nothing to reduce. Both ranks see the same count,
        // so skipping the exchange on both sides stays consistent.
        state->future->markCompleted(c10::IValue(state->tensors));
        return c10::make_intrusive<WorkTBCCL>(getRank(), c10d::OpType::ALLREDUCE, state);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(comm_ != nullptr, "torch-tbccl: communicator failure: process group has been shut down");
    try
    {
        // In place: same BufferView as send and receive; no intermediate tensor.
        state->work = comm_->all_reduce(buf.view, buf.view, buf.count, buf.datatype, op, buf.context);
    }
    catch (const std::runtime_error &e)
    {
        throw_tbccl_error("allreduce", e.what());
    }
    completion_->enqueue(state);
    return c10::make_intrusive<WorkTBCCL>(getRank(), c10d::OpType::ALLREDUCE, state);
}

bool ProcessGroupTBCCL::is_shutdown() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return comm_ == nullptr;
}

} // namespace torch_tbccl
