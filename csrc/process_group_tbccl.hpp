#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include <tbccl/communicator.hpp>

#include <chrono>
#include <memory>
#include <mutex>

namespace torch_tbccl
{

// c10d backend adapting torch.distributed onto an installed TBCCL
// Communicator. world_size == 2 only. No collectives are
// implemented yet; every inherited collective throws "does not support X".
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

private:
    c10::intrusive_ptr<c10d::Store> store_;
    std::chrono::milliseconds timeout_;
    mutable std::mutex mutex_;
    std::unique_ptr<tbccl::Communicator> comm_;
};

} // namespace torch_tbccl
