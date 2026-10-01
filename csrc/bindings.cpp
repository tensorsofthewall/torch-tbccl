// Native module surface for torch_tbccl. Exposes only adapter-level
// entry points; TBCCL's own classes are never exposed to Python.

#include <torch/extension.h>

#include <pybind11/chrono.h>

#include <tbccl/communicator.hpp>

#include "process_group_tbccl.hpp"
#include "trace.hpp"

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
#ifdef TORCH_TBCCL_WITH_CUDA
    d["cuda"] = true;
#else
    d["cuda"] = false;
#endif
    d["mps"] = false;
    return d;
}

// Signature expected by torch.distributed.Backend.register_backend
// (extended_api=False): (store, rank, world_size, timeout).
c10::intrusive_ptr<c10d::Backend> create_backend(
    const c10::intrusive_ptr<c10d::Store> &store,
    int rank,
    int world_size,
    const std::chrono::duration<float> &timeout)
{
    return c10::make_intrusive<torch_tbccl::ProcessGroupTBCCL>(
        store, rank, world_size, std::chrono::duration_cast<std::chrono::milliseconds>(timeout));
}

pybind11::list trace_events()
{
    pybind11::list out;
    for (const auto &r : torch_tbccl::trace_snapshot())
    {
        pybind11::dict d;
        d["seq"] = r->seq;
        d["op"] = r->op;
        d["bytes"] = r->bytes;
        d["device"] = r->device;
        d["thread"] = r->thread;
        d["entry_ns"] = r->entry_ns;
        d["before_submit_ns"] = r->before_submit_ns;
        d["return_ns"] = r->return_ns;
        d["complete_ns"] = r->complete_ns.load();
        d["wait_entry_ns"] = r->wait_entry_ns.load();
        d["wait_exit_ns"] = r->wait_exit_ns.load();
        d["error"] = r->error.load() != 0;
        out.append(d);
    }
    return out;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "torch-tbccl native adapter (PyTorch <-> installed TBCCL)";
    m.def("runtime_version", &runtime_version, "Version of the TBCCL runtime this extension was linked against");
    m.def("compiled_features", &compiled_features, "Devices this build can serve");
    m.def("trace_enabled", &torch_tbccl::trace_enabled, "Whether TORCH_TBCCL_TRACE recording is on");
    m.def("trace_set_enabled", &torch_tbccl::trace_set_enabled, py::arg("on"));
    m.def("trace_now_ns", &torch_tbccl::trace_now_ns, "Clock used for trace stamps (monotonic ns, this process)");
    m.def("trace_reset", &torch_tbccl::trace_reset, "Drop recorded events (call with no collectives in flight)");
    m.def("trace_events", &trace_events, "Recorded per-collective timeline (this process's monotonic ns)");
    // GIL released: bootstrap blocks on the Store / network.
    m.def(
        "create_backend", &create_backend, py::arg("store"), py::arg("rank"), py::arg("world_size"),
        py::arg("timeout"), py::call_guard<py::gil_scoped_release>());
}
