#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include "trace.hpp"

#include <tbccl/work.hpp>

#include <chrono>
#include <memory>
#include <optional>
#include <vector>

namespace torch_tbccl
{

// Shared between the user-visible WorkTBCCL and the process group's
// completion worker. TBCCL's BufferView is non-owning, so this state keeps
// every participating tensor alive until the TBCCL operation has finished,
// even if the user drops both the tensor and the Work handle.
struct WorkState
{
    std::vector<at::Tensor> tensors;   // the result list (what Work::result()/Future yield)
    std::vector<at::Tensor> retained;  // extra keep-alive (e.g. the allgather input)
    // Empty = trivially complete no-op (e.g. zero-element tensor).
    std::optional<tbccl::Work> work;
    c10::intrusive_ptr<c10::ivalue::Future> future;
    std::string op_name;
    std::shared_ptr<TraceRecord> trace; // null unless TORCH_TBCCL_TRACE

    bool is_completed() const { return !work || work->is_completed(); }
    bool has_error() const { return work && work->has_error(); }
    std::string error() const { return work ? work->error() : std::string(); }
    void wait() { if (work) work->wait(); }
};

class WorkTBCCL : public c10d::Work
{
public:
    WorkTBCCL(int rank, c10d::OpType op, std::shared_ptr<WorkState> state);

    bool isCompleted() override;
    bool isSuccess() const override;
    std::exception_ptr exception() const override;
    // timeout == 0 (c10d kNoTimeout): block until done. Otherwise a bounded
    // adapter-side wait; on expiry throws a timeout error. The underlying
    // TBCCL operation is NOT cancelled and may still complete afterwards.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
    void synchronize() override { wait(); }
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override { return state_->future; }
    std::vector<at::Tensor> result() override { return state_->tensors; }

private:
    std::shared_ptr<WorkState> state_;
};

} // namespace torch_tbccl
