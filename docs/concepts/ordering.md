# Collectives and point-to-point on one group

Collectives and point-to-point operations may be in flight on the same process group at the same time, in any relative order on each rank. libtbccl wire protocol 4 carries them on separate connections with separate workers ([TBCCL ordering domains](https://tbccl.tensorsofthewall.com/en/stable/concepts/ordering-domains.html)).

The contracts still hold inside each domain:

- every rank issues collectives in the same order;
- P2P messages match in FIFO order per peer and direction (tags are ignored).

This needs a libtbccl with wire protocol 4 or newer. torch-tbccl builds only against such a prefix; the adapter used to refuse overlapping the two kinds of traffic, and that restriction was removed once libtbccl guaranteed independence ({doc}`../adr/0001-adapter-only`).

Tests cover mixed traffic at world sizes 2 to 4 on CPU, CUDA and MPS, DDP with a concurrent application P2P thread, and the physical Thunderbolt link.
