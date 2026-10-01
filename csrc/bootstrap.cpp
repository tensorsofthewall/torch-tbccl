#include "bootstrap.hpp"

#include <c10/util/Exception.h>

#include <cstdlib>
#include <exception>
#include <vector>

namespace torch_tbccl
{

tbccl::CommunicatorPeerEndpoint parse_endpoint(const std::string &text)
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
        port >= 1 && port <= 65535, "torch-tbccl: invalid argument: endpoint '", text, "' port is out of range");
    return {host, static_cast<std::uint16_t>(port)};
}

tbccl::CommunicatorPeerEndpoint local_endpoint_from_env()
{
    const char *value = std::getenv(kLocalEndpointEnv);
    TORCH_CHECK_VALUE(
        value != nullptr && *value != '\0',
        "torch-tbccl: invalid argument: ", kLocalEndpointEnv,
        " is not set; set it to this rank's TBCCL data endpoint, e.g. 127.0.0.1:29600");
    return parse_endpoint(value);
}

tbccl::CommunicatorOptions bootstrap_options(
    const c10::intrusive_ptr<c10d::Store> &store,
    int rank,
    int world_size,
    std::chrono::milliseconds timeout,
    const tbccl::CommunicatorPeerEndpoint &local)
{
    auto key_for = [](int r) { return std::string(kEndpointKeyPrefix) + std::to_string(r); };

    const std::string mine = local.host + ":" + std::to_string(local.port);
    store->set(key_for(rank), std::vector<uint8_t>(mine.begin(), mine.end()));

    std::vector<std::string> keys;
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

    tbccl::CommunicatorOptions opts;
    opts.rank = static_cast<std::size_t>(rank);
    opts.bootstrap_timeout = timeout;
    for (int r = 0; r < world_size; ++r)
    {
        const auto raw = store->get(key_for(r));
        opts.peers.push_back(parse_endpoint(std::string(raw.begin(), raw.end())));
    }
    for (int a = 0; a < world_size; ++a)
        for (int b = a + 1; b < world_size; ++b)
            TORCH_CHECK_VALUE(
                opts.peers[a].host != opts.peers[b].host || opts.peers[a].port != opts.peers[b].port,
                "torch-tbccl: invalid argument: ranks ", a, " and ", b, " advertise the same TBCCL endpoint ",
                opts.peers[a].host, ":", opts.peers[a].port,
                "; give each rank a distinct ", kLocalEndpointEnv);
    return opts;
}

} // namespace torch_tbccl
