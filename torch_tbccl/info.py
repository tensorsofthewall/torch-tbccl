"""`python -m torch_tbccl.info`: print what is installed (package, torch, libtbccl runtime, C ABI, wire protocol, devices)."""
import json
import sys

import torch_tbccl


def main() -> int:
    data = torch_tbccl.info()
    if "--json" in sys.argv[1:]:
        print(json.dumps(data, indent=1))
    else:
        for k, v in data.items():
            print(f"{k:18} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
