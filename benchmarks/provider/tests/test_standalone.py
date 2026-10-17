"""The standalone loopback launcher: the summary follows the suite's definitions (offline), and, where PyTorch is installed, two real CPU ranks run, verify and are cleaned up."""
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

PROVIDER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROVIDER_DIR))
import standalone  # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None


def line(rank, series, values, **extra):
    return {"rank": rank, "backend": "gloo", "device": "cpu", "bytes": 4096, "verify_ok": True, "samples_us": {series: values}, **extra}


class Summary(unittest.TestCase):
    def test_collectives_report_the_per_iteration_maximum_over_both_ranks(self):
        s = standalone.summarize([line(0, "completion_us", [10.0, 50.0, 30.0]), line(1, "completion_us", [20.0, 40.0, 25.0])], "all_reduce")
        self.assertEqual((s["median_us"], s["min_us"], s["max_us"], s["samples"]), (30.0, 20.0, 50.0, 3))

    def test_point_to_point_reports_the_source_rank_only(self):
        s = standalone.summarize([line(0, "round_trip_us", [10.0, 20.0, 30.0]), line(1, "other_us", [999.0])], "p2p_rtt")
        self.assertEqual(s["median_us"], 20.0)

    def test_every_operation_has_a_series(self):
        self.assertEqual(set(standalone.SERIES), {"all_reduce", "broadcast", "all_gather", "p2p_ack", "p2p_rtt", "all_reduce_async", "isend_irecv_ack"})


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed")
class TwoRanks(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(PROVIDER_DIR / "standalone.py"), *args], capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL)

    def test_a_verified_loopback_measurement_is_reported(self):
        out = self.run_cli("--op", "all_reduce", "--bytes", "4096", "--backend", "gloo", "--warmup", "2", "--iterations", "10", "--json")
        self.assertEqual(out.returncode, 0, out.stderr)
        s = json.loads(out.stdout)
        self.assertTrue(s["verified"])
        self.assertEqual((s["samples"], s["backend"], s["devices"]), (10, "gloo", ["cpu", "cpu"]))

    def test_a_failing_rank_ends_the_run_without_a_measurement_and_leaves_no_process(self):
        out = self.run_cli("--op", "all_reduce", "--backend", "gloo", "--devices", "cuda,cuda", "--iterations", "3", "--timeout", "20")
        self.assertEqual(out.returncode, 1)
        self.assertNotIn("median", out.stdout)
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        self.assertNotIn(str(PROVIDER_DIR / "comm_bench.py"), ps)


if __name__ == "__main__":
    unittest.main()
