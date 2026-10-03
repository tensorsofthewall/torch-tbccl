__version__ = "0.2.0.dev0"

# torch minor series (major.minor) this package has been built and tested against. The compiled extension is tied to the torch it was built
# with (C++ ABI): torch_tbccl refuses to import when the running torch's major.minor differs from the one recorded at build time, and
# pyproject.toml bounds the dependency to these series. Distinct from the package version and from the linked TBCCL runtime version.
TESTED_TORCH_SERIES = ("2.13",)

# The exact release every environment (torch-tbccl and vllm-tbccl venvs, Linux and Mac) runs, so one wheel set is validated everywhere. vLLM pins it.
TESTED_TORCH_VERSION = "2.13.0"
