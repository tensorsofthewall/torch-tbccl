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
#include <vector>

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
    // Maps c10d's Backend::abort() onto the communicator-wide, destructive TBCCL abort: outstanding Works fail (their
    // Futures complete exceptionally) and later collectives throw. The communicator object stays alive until shutdown().
    // Work::wait(timeout) is unrelated and never aborts.
    void abort() override;

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

    // One input per rank to `rootRank`: the root's own input is copied into outputTensors[0][rootRank]; the peer's arrives via
    // TBCCL P2P (equal sizes, as all_gather_object-style callers use).
    c10::intrusive_ptr<c10d::Work> gather(
        std::vector<std::vector<at::Tensor>> &outputTensors,
        std::vector<at::Tensor> &inputTensors,
        const c10d::GatherOptions &opts = c10d::GatherOptions()) override;

    // Barrier = a one-element SUM all-reduce (what a synchronizing collective needs; one-rank groups return at once).
    c10::intrusive_ptr<c10d::Work> barrier(const c10d::BarrierOptions &opts = c10d::BarrierOptions()) override;

    // Point-to-point (byte-generic, any dense contiguous dtype, exactly one tensor, the single peer rank). `tag` is accepted
    // and ignored: transfers are matched by FIFO order per direction, as vLLM/PyTorch pipeline use requires. Do not
    // interleave P2P with collectives on the same group from different threads (one ordered byte stream underneath).
    c10::intrusive_ptr<c10d::Work> send(std::vector<at::Tensor> &tensors, int dstRank, int tag) override;
    c10::intrusive_ptr<c10d::Work> recv(std::vector<at::Tensor> &tensors, int srcRank, int tag) override;

    // Operations the backend does not implement: each throws NotImplementedError ("torch-tbccl: unsupported operation: <name> ...") instead of the base
    // class's generic RuntimeError, so callers (and the capability matrix) can tell a deliberate rejection from a failure.
    c10::intrusive_ptr<c10d::Work> allreduce_sparse(std::vector<at::Tensor> &, const c10d::AllreduceOptions &) override;
    c10::intrusive_ptr<c10d::Work> allreduce_coalesced(std::vector<at::Tensor> &, const c10d::AllreduceCoalescedOptions &) override;
    c10::intrusive_ptr<c10d::Work> reduce(std::vector<at::Tensor> &, const c10d::ReduceOptions &) override;
    c10::intrusive_ptr<c10d::Work> _allgather_base(at::Tensor &, at::Tensor &, const c10d::AllgatherOptions &) override;
    c10::intrusive_ptr<c10d::Work> allgather_coalesced(
        std::vector<std::vector<at::Tensor>> &, std::vector<at::Tensor> &, const c10d::AllgatherOptions &) override;
    c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(
        std::vector<at::Tensor> &, std::vector<at::Tensor> &, const c10d::AllgatherOptions &) override;
    c10::intrusive_ptr<c10d::Work> scatter(
        std::vector<at::Tensor> &, std::vector<std::vector<at::Tensor>> &, const c10d::ScatterOptions &) override;
    c10::intrusive_ptr<c10d::Work> reduce_scatter(
        std::vector<at::Tensor> &, std::vector<std::vector<at::Tensor>> &, const c10d::ReduceScatterOptions &) override;
    c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(at::Tensor &, at::Tensor &, const c10d::ReduceScatterOptions &) override;
    c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced(
        std::vector<at::Tensor> &, std::vector<at::Tensor> &, const c10d::ReduceScatterOptions &) override;
    c10::intrusive_ptr<c10d::Work> alltoall_base(
        at::Tensor &, at::Tensor &, std::vector<int64_t> &, std::vector<int64_t> &, const c10d::AllToAllOptions &) override;
    c10::intrusive_ptr<c10d::Work> alltoall(std::vector<at::Tensor> &, std::vector<at::Tensor> &, const c10d::AllToAllOptions &) override;
    c10::intrusive_ptr<c10d::Work> recvAnysource(std::vector<at::Tensor> &, int) override;

private:
    enum class Domain { Collective, P2P };
    // Refuse a submission that would overlap, on this group, with in-flight operations of the other domain; remember the new one (both under mutex_).
    void check_no_overlap(Domain mine, const char *op);
    void track(Domain mine, const std::shared_ptr<WorkState> &state);
    // Destroys the WorkStates the completion thread has finished with (see CompletionWorker) on the calling thread, outside any lock.
    void reap();
    void require_peers(const char *op) const;
    c10::intrusive_ptr<c10d::Work> p2p(std::vector<at::Tensor> &tensors, int peer, bool is_send);
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
    std::vector<std::weak_ptr<WorkState>> collective_in_flight_;
    std::vector<std::weak_ptr<WorkState>> p2p_in_flight_;
};

} // namespace torch_tbccl
