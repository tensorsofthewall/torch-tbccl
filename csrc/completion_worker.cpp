#include "completion_worker.hpp"

#include "errors.hpp"

namespace torch_tbccl
{

CompletionWorker::CompletionWorker() : thread_([this] { run(); }) {}

CompletionWorker::~CompletionWorker()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void CompletionWorker::enqueue(std::shared_ptr<WorkState> state)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(state));
    }
    cv_.notify_one();
}

void CompletionWorker::run()
{
    for (;;)
    {
        std::shared_ptr<WorkState> state;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) return; // stop_ && drained
            state = std::move(queue_.front());
            queue_.pop_front();
        }
        state->wait();
        if (state->has_error())
            state->future->setError(std::make_exception_ptr(std::runtime_error(
                "torch-tbccl: " + state->op_name + " failed: " + state->error())));
        else
            state->future->markCompleted(c10::IValue(state->tensors));
    }
}

} // namespace torch_tbccl
