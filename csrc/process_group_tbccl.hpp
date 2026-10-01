#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include "completion_worker.hpp"

#include <tbccl/communicator.hpp>

#include <chrono>
#include <memory>
#include <mutex>

namespace torch_tbccl
{

// c10d backend adapting torch.distributed onto an installed TBCCL
// Communicator. Phase 42: world_size == 2 only. No collectives are
// implemented except allreduce; every other inherited collective throws
// "does not support X".
class ProcessGroupTBCCL : public c10d::Backend
{
public:
    // Validates arguments, performs the Store-based endpoint exchange and
    // creates the Communicator. Throws on any failure (never returns a
    // half-initialized backend).
    ProcessGroupTBCCL(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        std::chrono::milliseconds timeout);
    ~ProcessGroupTBCCL() override;

    const std::string getBackendName() const override { return "tbccl"; }
    void setTimeout(std::chrono::milliseconds timeout) override { timeout_ = timeout; }
    void shutdown() override;
    void abort() override { shutdown(); }

    std::chrono::milliseconds timeout() const { return timeout_; }
    bool is_shutdown() const;

    // In-place Float32 SUM over exactly one dense, contiguous tensor.
    // Always returns after submission; completion is observed through the
    // returned Work (PyTorch itself calls wait() when async_op=False).
    c10::intrusive_ptr<c10d::Work> allreduce(
        std::vector<at::Tensor> &tensors,
        const c10d::AllreduceOptions &opts = c10d::AllreduceOptions()) override;

private:
    c10::intrusive_ptr<c10d::Store> store_;
    std::chrono::milliseconds timeout_;
    // Guards comm_/completion_ and serializes submission (TBCCL runs one
    // collective at a time); never held while waiting on a collective.
    mutable std::mutex mutex_;
    std::unique_ptr<tbccl::Communicator> comm_;
    std::unique_ptr<CompletionWorker> completion_;
};

} // namespace torch_tbccl
