#pragma once

// Opt-in (TORCH_TBCCL_TRACE=1) per-collective timeline recorded in memory and read back after the
// measured region; there is no I/O on the hot path and no behavior change when disabled. Timestamps
// are CLOCK_MONOTONIC-style nanoseconds of THIS process: never subtract them across machines.
//
// The recorder is process-wide (a diagnostic, not communicator state) so the Python helpers can
// reach it without a handle to the backend object.

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace torch_tbccl
{

std::uint64_t trace_now_ns();

struct TraceRecord
{
    std::uint64_t seq = 0;
    std::string op;
    std::uint64_t bytes = 0;
    std::string device;
    std::uint64_t thread = 0;
    std::uint64_t entry_ns = 0;
    std::uint64_t before_submit_ns = 0;
    std::uint64_t return_ns = 0;
    // First observer of completion wins (the completion worker or a wait() return), so a caller that reads
    // the trace right after wait() never sees 0.
    std::atomic<std::uint64_t> complete_ns{0};
    std::atomic<std::uint64_t> wait_entry_ns{0}; // first Work::wait() call, if any
    std::atomic<std::uint64_t> wait_exit_ns{0};
    std::atomic<std::uint64_t> error{0};

    void stamp_complete()
    {
        std::uint64_t zero = 0;
        complete_ns.compare_exchange_strong(zero, trace_now_ns());
    }
};

bool trace_enabled();
void trace_set_enabled(bool on);
void trace_reset();
// Allocates a record (entry stamped now) or returns null when tracing is off.
std::shared_ptr<TraceRecord> trace_begin(const char *op, std::uint64_t bytes, const std::string &device);
std::vector<std::shared_ptr<TraceRecord>> trace_snapshot();

} // namespace torch_tbccl
