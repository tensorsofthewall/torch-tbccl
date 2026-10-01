#include "trace.hpp"

#include <pthread.h>

#include <chrono>
#include <cstdlib>
#include <mutex>

namespace torch_tbccl
{

namespace
{
std::mutex g_mutex;
std::vector<std::shared_ptr<TraceRecord>> g_records;
std::atomic<std::uint64_t> g_seq{0};
std::atomic<bool> g_enabled{[] {
    const char *v = std::getenv("TORCH_TBCCL_TRACE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}()};
} // namespace

std::uint64_t trace_now_ns()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool trace_enabled() { return g_enabled.load(std::memory_order_relaxed); }
void trace_set_enabled(bool on) { g_enabled.store(on); }

void trace_reset()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_records.clear();
    g_seq.store(0);
}

std::shared_ptr<TraceRecord> trace_begin(const char *op, std::uint64_t bytes, const std::string &device)
{
    if (!trace_enabled()) return nullptr;
    auto r = std::make_shared<TraceRecord>();
    r->entry_ns = trace_now_ns();
    r->op = op;
    r->bytes = bytes;
    r->device = device;
    r->thread = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pthread_self()));
    std::lock_guard<std::mutex> lock(g_mutex);
    r->seq = g_seq.fetch_add(1);
    g_records.push_back(r);
    return r;
}

std::vector<std::shared_ptr<TraceRecord>> trace_snapshot()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_records;
}

} // namespace torch_tbccl
