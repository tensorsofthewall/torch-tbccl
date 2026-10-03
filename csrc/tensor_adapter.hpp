#pragma once

// The single narrow conversion point between PyTorch and TBCCL's
// framework-neutral buffer types. Never copies and never calls
// .contiguous(): the BufferView aliases the tensor's own storage.

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <tbccl/communicator.hpp>

namespace torch_tbccl
{

struct TbcclBuffer
{
    tbccl::BufferView view;
    tbccl::ExecutionContext context;
    std::size_t count = 0;
    tbccl::DataType datatype = tbccl::DataType::Float32;
};

// Validates `t` and maps it. Reductions need a TBCCL numeric dtype
// (`bytes_only == false`: float16/bfloat16/float32/float64/int8/uint8/int32/int64); send/recv, Broadcast and AllGather move bytes and
// accept any dense contiguous dtype (`bytes_only == true`, `datatype` unset).
// Throws c10::ValueError / c10::NotImplementedError with a clear message.
TbcclBuffer to_tbccl_buffer(const at::Tensor &t, bool bytes_only = false);

tbccl::DataType to_tbccl_dtype(at::ScalarType type);
tbccl::ReduceOp to_tbccl_reduce_op(const c10d::ReduceOp &op, tbccl::DataType datatype);

} // namespace torch_tbccl
