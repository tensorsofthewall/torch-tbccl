#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include "completion_worker.hpp"
#include "work_tbccl.hpp"

#include <tbccl/communicator.hpp>

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>

namespace torch_tbccl
{

// c10d backend adapting torch.distributed onto an installed TBCCL
// Communicator. world_size == 2 only. No collectives are
// implemented except allreduce, broadcast and allgather; every other inherited collective throws
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

    // Byte-generic (any dense contiguous dtype). Exactly one tensor, rootTensor == 0.
    c10::intrusive_ptr<c10d::Work> broadcast(
        std::vector<at::Tensor> &tensors,
        const c10d::BroadcastOptions &opts = c10d::BroadcastOptions()) override;

    // One input, one output list of world_size tensors with the input's dtype/numel, all on the
    // input's device. Variable sizes across ranks are not supported (DDP exchanges equal sizes).
    c10::intrusive_ptr<c10d::Work> allgather(
        std::vector<std::vector<at::Tensor>> &outputTensors,
        std::vector<at::Tensor> &inputTensors,
        const c10d::AllgatherOptions &opts = c10d::AllgatherOptions()) override;

private:
    c10::intrusive_ptr<c10d::Work> finish(std::shared_ptr<WorkState> state, c10d::OpType op);

    c10::intrusive_ptr<c10d::Store> store_;
    std::chrono::milliseconds timeout_;
    // Diagnostic only (TORCH_TBCCL_FORCE_SYNC_ALLREDUCE=1): allreduce waits for TBCCL before returning,
    // giving a no-overlap baseline for DDP. Never a production mode.
    bool force_sync_allreduce_ = false;
    // Guards comm_/completion_ and serializes submission (TBCCL runs one
    // collective at a time); never held while waiting on a collective.
    mutable std::mutex mutex_;
    std::unique_ptr<tbccl::Communicator> comm_;
    std::unique_ptr<CompletionWorker> completion_;
};

} // namespace torch_tbccl
