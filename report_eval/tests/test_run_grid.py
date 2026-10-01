"""Runner tests on the real 245-film table (skipped where the analysis outputs are absent)."""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from report_eval.films import TABLE, load_films  # noqa: E402
from report_eval.llm_backends import BackendError, LLMResult  # noqa: E402
from report_eval.run_grid import load_done, parse_args, run, sample_films  # noqa: E402

HAVE_DATA = os.path.exists(TABLE)


class SilentBackend:
    """Always declines to state a location, in valid S2 JSON."""
    name, model = "fake", "fake-1"

    def __init__(self, fail_every=0):
        self.calls = 0
        self.fail_every = fail_every

    def generate(self, system, user, images=None, schema=None, max_tokens=1024):
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            raise BackendError("simulated outage")
        reply = {"state_location": False, "side": "none", "zone": "none", "extent": "none",
                 "sentence": "No location is reported."}
        return LLMResult(text=json.dumps(reply), model=self.model, latency_s=0.0)


@unittest.skipUnless(HAVE_DATA, "fig7_work analysis outputs not present")
class RunGridTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "grid.jsonl")
        self.all_films = load_films()

    def tearDown(self):
        self.tmp.cleanup()

    def _args(self, gens):
        return parse_args(["--generators", gens, "--backend", "claude", "--out", self.out])

    def test_templates_reproduce_published_claim_count(self):
        run(self._args("s1"), self.all_films, self.all_films, None)
        actual = [r for r in load_done(self.out) if r["condition"] == "actual"]
        self.assertEqual(sum(r["claimed"] for r in actual), 92)
        none = [r for r in load_done(self.out) if r["condition"] == "none"]
        self.assertEqual(sum(r["claimed"] for r in none), 0)

    def test_resume_skips_finished_work(self):
        films = sample_films(self.all_films, 10)
        backend = SilentBackend()
        run(self._args("s2"), films, self.all_films, backend)
        first = backend.calls
        run(self._args("s2"), films, self.all_films, backend)
        self.assertEqual(backend.calls, first)
        self.assertEqual(len(load_done(self.out)), 30)  # 10 films x 3 conditions

    def test_backend_errors_are_recorded_per_film(self):
        films = sample_films(self.all_films, 6)
        run(self._args("s2"), films, self.all_films, SilentBackend(fail_every=3))
        errors = [r for r in load_done(self.out) if r["error"]]
        self.assertEqual(len(errors), 6)  # every third of 18 calls
        self.assertTrue(all("simulated outage" in r["error"] for r in errors))

    def test_permuted_condition_uses_another_films_heatmap(self):
        run(self._args("s1"), sample_films(self.all_films, 5), self.all_films, None)
        permuted = [r for r in load_done(self.out) if r["condition"] == "permuted"]
        self.assertTrue(all(r["heatmap_from"] != r["key"] for r in permuted))


if __name__ == "__main__":
    unittest.main()
