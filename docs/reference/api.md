# Python API

The package's public surface is deliberately small: importing `torch_tbccl` registers the `"tbccl"` backend with `torch.distributed`; everything else goes through `torch.distributed`. The module also exposes version and diagnostic helpers (`info`, `runtime_version`, `c_abi_version`, `wire_protocol_version`, `built_with_torch`, `compiled_features`, `supported_devices`, `register_backend`, `trace_events` and related functions). The generated reference below is built by parsing the sources; it does not import the package.

```{toctree}
:maxdepth: 1

/autoapi/torch_tbccl/index
```
