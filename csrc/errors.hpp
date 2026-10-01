#pragma once

// Maps TBCCL's tagged error messages ("<error_code_name>: detail") onto
// PyTorch exception types, keeping the original message as context.
// TBCCL currently reports errors only as tagged strings (see audit doc).

#include <c10/util/Exception.h>

#include <string>

namespace torch_tbccl
{

[[noreturn]] inline void throw_tbccl_error(const std::string &what, const std::string &message)
{
    const auto colon = message.find(':');
    const std::string tag = colon == std::string::npos ? "" : message.substr(0, colon);
    const std::string full = "torch-tbccl: " + what + ": " + message;
    if (tag == "invalid_argument") TORCH_CHECK_VALUE(false, "torch-tbccl: invalid argument: ", what, ": ", message);
    if (tag == "unsupported") TORCH_CHECK_NOT_IMPLEMENTED(false, "torch-tbccl: unsupported operation: ", what, ": ", message);
    const char *category = "communicator failure";
    if (tag == "transport_error") category = "transport error";
    else if (tag == "timeout") category = "timeout";
    else if (tag == "device_error") category = "device error";
    else if (tag == "internal_error") category = "internal error";
    TORCH_CHECK(false, "torch-tbccl: ", category, ": ", what, ": ", message);
}

} // namespace torch_tbccl
