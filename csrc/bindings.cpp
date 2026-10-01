// Native module surface for torch_tbccl. Exposes only adapter-level
// diagnostics (and, in later commits, the backend factory); TBCCL's own
// classes are never exposed to Python.

#include <torch/extension.h>

#include <tbccl/communicator.hpp>

#ifndef TORCH_TBCCL_LINKED_TBCCL_VERSION
#define TORCH_TBCCL_LINKED_TBCCL_VERSION "unknown"
#endif

namespace
{

std::string runtime_version()
{
    return TORCH_TBCCL_LINKED_TBCCL_VERSION;
}

pybind11::dict compiled_features()
{
    pybind11::dict d;
    d["cpu"] = tbccl::memory_kind_registered(tbccl::MemoryKind::Host);
    d["cuda"] = false; // enabled once the installed tbccl_cuda component is consumed
    d["mps"] = false;
    return d;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "torch-tbccl native adapter (PyTorch <-> installed TBCCL)";
    m.def("runtime_version", &runtime_version, "Version of the TBCCL runtime this extension was linked against");
    m.def("compiled_features", &compiled_features, "Devices this build can serve");
}
