#include "process_group_tbccl.hpp"

#include "bootstrap.hpp"
#include "errors.hpp"
#include "tensor_adapter.hpp"
#include "work_tbccl.hpp"

#include <c10/util/Exception.h>

#include <exception>

namespace torch_tbccl
{

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
}

ProcessGroupTBCCL::~ProcessGroupTBCCL()
{
    shutdown();
}

void ProcessGroupTBCCL::shutdown()
{
    std::unique_ptr<tbccl::Communicator> doomed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        doomed = std::move(comm_);
    }
    doomed.reset(); // joins TBCCL's workers outside the lock
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

    // Zero elements: nothing to reduce. Both ranks see the same count, so
    // skipping the exchange on both sides stays consistent.
    if (buf.count == 0) return c10::make_intrusive<WorkTBCCL>(getRank(), c10d::OpType::ALLREDUCE, tensors);

    std::lock_guard<std::mutex> serial(collective_mutex_);
    tbccl::Communicator *comm = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        comm = comm_.get();
    }
    TORCH_CHECK(comm != nullptr, "torch-tbccl: communicator failure: process group has been shut down");

    try
    {
        // In place: same BufferView as send and receive; no intermediate tensor.
        auto work = comm->all_reduce(buf.view, buf.view, buf.count, buf.datatype, op, buf.context);
        work.wait();
        if (work.has_error()) throw_tbccl_error("allreduce", work.error());
    }
    catch (const std::runtime_error &e)
    {
        throw_tbccl_error("allreduce", e.what());
    }
    return c10::make_intrusive<WorkTBCCL>(getRank(), c10d::OpType::ALLREDUCE, tensors);
}

bool ProcessGroupTBCCL::is_shutdown() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return comm_ == nullptr;
}

} // namespace torch_tbccl
