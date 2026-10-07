# Rendezvous

PyTorch's rank rendezvous (`MASTER_ADDR`/`MASTER_PORT`, a `TCPStore`, `torchrun`) and TBCCL's endpoint exchange are separate. Through the group's store, each rank publishes its **actual** control and data endpoints and rank 0 publishes one communicator id, under keys of the form `torch_tbccl/v3/g<generation>/{endpoint/<rank>,communicator_id}`. Each rank joins once through an atomic counter (`torch_tbccl/v3/joined`), and `generation = (joins - 1) / world_size`, so successive groups created over the same persistent store (such as torchrun's) never read an earlier group's records. libtbccl itself never sees the store: the adapter hands it a rank directory.

With `TBCCL_LOCAL_ENDPOINT=<host>:<port>` the data port is `port + 1000`; `<host>:0` lets the kernel choose free ports, so many groups can coexist in one process. On two machines use each one's reachable address. Only the last rank needs no listener.

The key layout is internal: ranks of one group must all run the same torch-tbccl version.
