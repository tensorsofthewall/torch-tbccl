# Security model

torch-tbccl is an adapter over TBCCL and has the same trust model: **trusted peers on a trusted network**. There is no peer authentication, no transport encryption, no message authentication and no authorization. Read the TBCCL security model first; this page lists what torch-tbccl adds.

torch-tbccl adds one thing to TBCCL's trust model: the endpoint exchange. Each rank publishes its control and data endpoints and rank 0 publishes the communicator id through the `torch.distributed` store (see {doc}`rendezvous`). That store, typically a `TCPStore`, is not authenticated or encrypted by PyTorch either; any process that can reach it can read and change those records. Run it on the same trusted network as the ranks.

The compiled extension validates tensors, dtypes and reduction operators at the boundary and reports errors with a `torch-tbccl:` prefix. That is input validation for the API, not a defense against a hostile peer.

## Deployment assumptions

- All ranks are run by the same trusted party, on hosts and links that no untrusted party can reach: loopback, a private network, or a direct link such as Thunderbolt.
- If traffic must cross a network that is not trusted, put it inside a tunnel that provides authentication and encryption (a VPN or an encrypted overlay). torch-tbccl does not provide one.
- Do not expose rank listeners to the public internet. Bind to the specific address of the trusted link rather than a wildcard where the interface allows it.

## Reporting a vulnerability

See the security policy in the repository (`SECURITY.md`). A clean error on malformed input is a robustness property and does not mean the component is safe against a hostile peer.
