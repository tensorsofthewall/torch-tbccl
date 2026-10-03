#include "bootstrap.hpp"

#include <c10/util/Exception.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <random>
#include <exception>
#include <vector>

namespace torch_tbccl
{

tbccl::CommunicatorPeerEndpoint parse_endpoint(const std::string &text, bool allow_auto_port)
{
    const auto colon = text.rfind(':');
    TORCH_CHECK_VALUE(
        colon != std::string::npos,
        "torch-tbccl: invalid argument: endpoint '", text, "' is not of the form host:port");
    const std::string host = text.substr(0, colon);
    const std::string port_text = text.substr(colon + 1);
    TORCH_CHECK_VALUE(!host.empty(), "torch-tbccl: invalid argument: endpoint '", text, "' has an empty host");
    TORCH_CHECK_VALUE(
        host.find(':') == std::string::npos,
        "torch-tbccl: invalid argument: endpoint '", text, "': IPv6 literals are not supported in Phase 42");
    TORCH_CHECK_VALUE(
        !port_text.empty() && port_text.size() <= 5 &&
            port_text.find_first_not_of("0123456789") == std::string::npos,
        "torch-tbccl: invalid argument: endpoint '", text, "' has a non-numeric port");
    const long port = std::strtol(port_text.c_str(), nullptr, 10);
    TORCH_CHECK_VALUE(
        (port >= 1 || (allow_auto_port && port == 0)) && port <= 65535, "torch-tbccl: invalid argument: endpoint '", text, "' port is out of range");
    return {host, static_cast<std::uint16_t>(port)};
}

tbccl::CommunicatorPeerEndpoint local_endpoint_from_env()
{
    const char *value = std::getenv(kLocalEndpointEnv);
    TORCH_CHECK_VALUE(
        value != nullptr && *value != '\0',
        "torch-tbccl: invalid argument: ", kLocalEndpointEnv,
        " is not set; set it to this rank's TBCCL data endpoint, e.g. 127.0.0.1:29600");
    return parse_endpoint(value, true);
}

namespace
{
std::string endpoint_text(const tbccl::Endpoint &e) { return e.host + ":" + std::to_string(e.port); }

std::string record_for(const tbccl::Endpoint &control, const tbccl::Endpoint &data)
{
    return endpoint_text(control) + "," + endpoint_text(data);
}

tbccl::Endpoint parse_part(const std::string &text, int rank)
{
    try
    {
        return parse_endpoint(text, true);
    }
    catch (const c10::Error &e)
    {
        TORCH_CHECK_VALUE(false, "torch-tbccl: invalid argument: rank ", rank, " published a malformed endpoint record '", text, "'");
    }
}
} // namespace

tbccl::CommunicatorOptions bootstrap_options(
    const c10::intrusive_ptr<c10d::Store> &store,
    int rank,
    int world_size,
    std::chrono::milliseconds timeout,
    const tbccl::CommunicatorPeerEndpoint &local)
{
    auto key_for = [](int r) { return std::string(kEndpointKeyPrefix) + std::to_string(r); };
    const auto bytes = [](const std::string &s) { return std::vector<uint8_t>(s.begin(), s.end()); };

    tbccl::CommunicatorOptions opts;
    opts.rank = static_cast<std::size_t>(rank);
    opts.world_size = static_cast<std::size_t>(world_size);
    opts.bootstrap_timeout = timeout;

    std::shared_ptr<tbccl::CommunicatorListeners> listeners;
    std::string mine = "-"; // the last rank never accepts a connection and publishes nothing
    if (tbccl::rank_accepts_connections(static_cast<std::size_t>(rank), static_cast<std::size_t>(world_size)))
    {
        const std::uint16_t data_port = local.port == 0 ? 0 : static_cast<std::uint16_t>(local.port + 1000);
        TORCH_CHECK_VALUE(
            local.port == 0 || local.port + 1000 <= 65535,
            "torch-tbccl: invalid argument: ", kLocalEndpointEnv, " port ", local.port, " leaves no room for the data port (port + 1000)");
        try
        {
            listeners = tbccl::CommunicatorListeners::bind(local.host, local.port, data_port);
        }
        catch (const std::exception &e)
        {
            TORCH_CHECK(false, "torch-tbccl: bootstrap: could not bind TBCCL listeners on ", local.host, ":", local.port, ": ", e.what());
        }
        mine = record_for(listeners->control(), listeners->data());
    }
    opts.listeners = listeners;

    // The shared communicator id: rank 0 generates it, everyone reads it.
    if (rank == 0) store->set(kCommunicatorIdKey, bytes(tbccl::CommunicatorId::generate().to_hex()));
    store->set(key_for(rank), bytes(mine));

    std::vector<std::string> keys{kCommunicatorIdKey};
    for (int r = 0; r < world_size; ++r) keys.push_back(key_for(r));
    try
    {
        store->wait(keys, timeout);
    }
    catch (const std::exception &e)
    {
        TORCH_CHECK(
            false, "torch-tbccl: bootstrap timeout: not all ranks published a TBCCL endpoint within ",
            timeout.count(), " ms (", e.what(), ")");
    }

    {
        const auto raw = store->get(kCommunicatorIdKey);
        opts.communicator_id = tbccl::CommunicatorId::from_hex(std::string(raw.begin(), raw.end()));
    }
    for (int r = 0; r < world_size; ++r)
    {
        const auto raw = store->get(key_for(r));
        const std::string text(raw.begin(), raw.end());
        tbccl::RankEndpoint e;
        e.rank = static_cast<std::size_t>(r);
        if (text != "-")
        {
            const auto comma = text.find(',');
            TORCH_CHECK_VALUE(comma != std::string::npos, "torch-tbccl: invalid argument: rank ", r, " published a malformed endpoint record '", text, "'");
            e.control = parse_part(text.substr(0, comma), r);
            e.data = parse_part(text.substr(comma + 1), r);
        }
        opts.rank_directory.entries.push_back(e);
    }
    return opts;
}

} // namespace torch_tbccl
