#pragma once

#include <torch/csrc/distributed/c10d/Work.hpp>

#include <vector>

namespace torch_tbccl
{

// c10d::Work returned by ProcessGroupTBCCL. Retains strong references to
// every participating tensor (TBCCL's BufferView is non-owning). In this
// step the underlying TBCCL operation has already completed when the Work
// is constructed (blocking path), so it is born completed; the
// asynchronous variant extends this class.
class WorkTBCCL : public c10d::Work
{
public:
    WorkTBCCL(int rank, c10d::OpType op, std::vector<at::Tensor> tensors);

    std::vector<at::Tensor> result() override { return tensors_; }

private:
    std::vector<at::Tensor> tensors_;
};

} // namespace torch_tbccl
