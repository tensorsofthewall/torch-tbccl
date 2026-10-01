#include "work_tbccl.hpp"

#include "errors.hpp"

#include <thread>

namespace torch_tbccl
{

WorkTBCCL::WorkTBCCL(int rank, c10d::OpType op, std::shared_ptr<WorkState> state)
    : c10d::Work(rank, op, nullptr, state->tensors), state_(std::move(state))
{
}

bool WorkTBCCL::isCompleted()
{
    return state_->is_completed();
}

bool WorkTBCCL::isSuccess() const
{
    return state_->is_completed() && !state_->has_error();
}

std::exception_ptr WorkTBCCL::exception() const
{
    if (!state_->is_completed() || !state_->has_error()) return nullptr;
    return std::make_exception_ptr(std::runtime_error(state_->error()));
}

bool WorkTBCCL::wait(std::chrono::milliseconds timeout)
{
    if (state_->trace)
    {
        std::uint64_t zero = 0;
        state_->trace->wait_entry_ns.compare_exchange_strong(zero, trace_now_ns());
    }
    if (timeout.count() == 0)
    {
        state_->wait();
    }
    else
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        auto nap = std::chrono::microseconds(20);
        while (!state_->is_completed())
        {
            if (std::chrono::steady_clock::now() >= deadline)
                TORCH_CHECK(
                    false, "torch-tbccl: timeout: ", state_->op_name, " did not complete within ", timeout.count(),
                    " ms (the underlying TBCCL operation is not cancelled and may still complete)");
            std::this_thread::sleep_for(nap);
            if (nap < std::chrono::microseconds(1000)) nap *= 2;
        }
    }
    if (state_->trace) state_->trace->wait_exit_ns.store(trace_now_ns());
    if (state_->has_error()) throw_tbccl_error(state_->op_name, state_->error());
    return true;
}

} // namespace torch_tbccl
