#pragma once

#include "work_tbccl.hpp"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace torch_tbccl
{

// Exactly one persistent thread per process group. Waits, in FIFO order,
// for each submitted TBCCL operation and completes (or fails) its c10
// Future from C++; it never touches the Python API. It also owns a
// reference to each WorkState until completion, which is what keeps tensors
// alive when the user drops the Work handle early.
class CompletionWorker
{
public:
    CompletionWorker();
    ~CompletionWorker(); // drains remaining items, then joins

    void enqueue(std::shared_ptr<WorkState> state);

private:
    void run();

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<WorkState>> queue_;
    bool stop_ = false;
    std::thread thread_;
};

} // namespace torch_tbccl
