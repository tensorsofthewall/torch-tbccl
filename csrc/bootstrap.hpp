#pragma once

// Translates PyTorch's rendezvous (c10d::Store + rank/world_size/timeout)
// into TBCCL's framework-neutral bootstrap configuration. TBCCL itself
// never sees a c10d::Store.

#include <torch/csrc/distributed/c10d/Store.hpp>

#include <tbccl/communicator.hpp>

#include <chrono>
#include <optional>
#include <string>

namespace torch_tbccl
{

inline constexpr const char *kLocalEndpointEnv = "TBCCL_LOCAL_ENDPOINT";
inline constexpr const char *kEndpointKeyPrefix = "torch_tbccl/v1/endpoint/";

// Parses "host:port". Throws c10::ValueError (invalid argument) on an
// empty host, missing/non-numeric/out-of-range port, or IPv6-style
// colons in the host (not supported in PyTorch backend).
tbccl::CommunicatorPeerEndpoint parse_endpoint(const std::string &text, bool allow_auto_port = false);

// A local endpoint with port 0 ("host:0") means "pick a free port pair for this communicator"; others pass through.
tbccl::CommunicatorPeerEndpoint resolve_auto_port(const tbccl::CommunicatorPeerEndpoint &local);

// Reads TBCCL_LOCAL_ENDPOINT and validates it. Throws c10::ValueError if
// unset or malformed. Touches neither the Store nor the network.
tbccl::CommunicatorPeerEndpoint local_endpoint_from_env();

// Publishes `local` under this rank's namespaced Store key, waits (bounded
// by `timeout`) for every rank's record, and returns CommunicatorOptions
// with peers ordered by rank. Throws on bootstrap timeout or if two ranks
// advertise the same endpoint.
tbccl::CommunicatorOptions bootstrap_options(
    const c10::intrusive_ptr<c10d::Store> &store,
    int rank,
    int world_size,
    std::chrono::milliseconds timeout,
    const tbccl::CommunicatorPeerEndpoint &local);

} // namespace torch_tbccl
