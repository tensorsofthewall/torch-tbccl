"""The README's quick-start example is run verbatim under torchrun so the documentation cannot drift from reality."""
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCHRUN = os.path.join(os.path.dirname(sys.executable), "torchrun")


@pytest.mark.multiprocess
@pytest.mark.skipif(not os.path.exists(TORCHRUN), reason="torchrun not installed next to the interpreter")
def test_readme_quick_start_runs_under_torchrun(tmp_path):
    code = re.search(r"```python\n(.*?)```", open(os.path.join(ROOT, "README.md")).read(), re.S).group(1)
    script = tmp_path / "quick_start.py"
    script.write_text(code)
    env = dict(os.environ, TBCCL_LOCAL_ENDPOINT="127.0.0.1:0", OMP_NUM_THREADS="2")
    p = subprocess.run([TORCHRUN, "--standalone", "--nproc-per-node", "2", str(script)], env=env, capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr[-2000:]
    assert p.stdout.count("tensor([3.])") == 2, p.stdout
