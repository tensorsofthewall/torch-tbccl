#include "process_group_tbccl.hpp"

#include "bootstrap.hpp"

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
    TORCH_CHECK_NOT_IMPLEMENTED(world_size == 2, "torch-tbccl supports world_size=2 only (got ", world_size, ")");
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

bool ProcessGroupTBCCL::is_shutdown() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return comm_ == nullptr;
}

} // namespace torch_tbccl
