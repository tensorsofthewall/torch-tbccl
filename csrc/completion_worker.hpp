#pragma once

#include "work_tbccl.hpp"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace torch_tbccl
{

// Exactly one persistent thread per process group. Waits, in FIFO order,
// for each submitted TBCCL operation and completes (or fails) its c10
// Future from C++; it never touches the Python API. It also owns a
// reference to each WorkState until completion, which is what keeps tensors
// alive when the user drops the Work handle early.
//
// The worker never destroys a WorkState: a tensor's last reference may be the one that keeps its Python object alive, and dropping it takes the GIL
// (torch's decref hook). On the worker thread that deadlocks against any caller that joins the worker while holding the GIL, and during interpreter
// finalization a non-finalizing thread asking for the GIL is parked forever, so a process that simply exited hung in the join. Finished states are
// therefore retired to a list that callers drain (take_retired) and the owner clears on its own thread.
class CompletionWorker
{
public:
    CompletionWorker();
    ~CompletionWorker(); // stop()s if still running; the retired states are then destroyed on the destroying thread

    void enqueue(std::shared_ptr<WorkState> state);
    // Finish the queued items, then join the thread. Idempotent. The caller should not hold the GIL.
    void stop();
    // The states the worker has finished with; destroy the returned vector on a thread that may take the GIL, outside any lock.
    std::vector<std::shared_ptr<WorkState>> take_retired();

private:
    void run();

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<WorkState>> queue_;
    std::vector<std::shared_ptr<WorkState>> retired_;
    bool stop_ = false;
    std::thread thread_;
};

} // namespace torch_tbccl
