#include "work_tbccl.hpp"

namespace torch_tbccl
{

WorkTBCCL::WorkTBCCL(int rank, c10d::OpType op, std::vector<at::Tensor> tensors)
    : c10d::Work(rank, op, nullptr, tensors), tensors_(std::move(tensors))
{
    finish();
}

} // namespace torch_tbccl
