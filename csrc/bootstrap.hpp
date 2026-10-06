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
// v3 (Phase 71): one record per rank holding its ACTUAL control and data endpoint, plus one shared communicator id, under a per-generation
// namespace "torch_tbccl/v3/g<generation>/{endpoint/<rank>,communicator_id}". Every rank joins a group exactly once by incrementing kJoinCounterKey, so
// generation = (joins - 1) / world_size is the same on every rank and differs between successive groups created over the SAME store (a persistent store
// such as torchrun's, where init_process_group -> destroy_process_group -> init_process_group reuses the key prefix): a later group never reads an earlier
// group's records. v2 used fixed keys and mixed up generations after a re-init.
inline constexpr const char *kJoinCounterKey = "torch_tbccl/v3/joined";
inline constexpr const char *kGenerationPrefix = "torch_tbccl/v3/g";

// Parses "host:port". Throws c10::ValueError (invalid argument) on an
// empty host, missing/non-numeric/out-of-range port, or IPv6-style
// colons in the host (not supported in Phase 42).
tbccl::CommunicatorPeerEndpoint parse_endpoint(const std::string &text, bool allow_auto_port = false);

// Reads TBCCL_LOCAL_ENDPOINT and validates it. Throws c10::ValueError if
// unset or malformed. Touches neither the Store nor the network.
tbccl::CommunicatorPeerEndpoint local_endpoint_from_env();

// Everything one rank needs from the rendezvous: the options for tbccl::Communicator::create() (explicit rank directory, shared communicator id,
// pre-bound listeners). TBCCL itself never sees the Store.
//
// `local` is TBCCL_LOCAL_ENDPOINT: its host is where this rank listens. Its port is the control port (0 = any free port) and the data port is
// `port + 1000` for a nonzero port (the long-standing user-facing convention) or any free port for 0. Only ranks that accept connections
// (every rank but the last) bind listeners. The ACTUAL endpoints, not the requested ones, are published, so nothing downstream derives one
// port from another. Rank 0 generates the communicator id and publishes it; every other rank reads it. All keys live under the Store the
// process group was given (a per-group PrefixStore), so several process groups coexist.
//
// Throws on bootstrap timeout (not every rank published within `timeout`). Duplicate or conflicting endpoints are rejected by libtbccl.
tbccl::CommunicatorOptions bootstrap_options(
    const c10::intrusive_ptr<c10d::Store> &store,
    int rank,
    int world_size,
    std::chrono::milliseconds timeout,
    const tbccl::CommunicatorPeerEndpoint &local);

} // namespace torch_tbccl
