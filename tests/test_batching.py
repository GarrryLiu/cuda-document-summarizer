import json
import tempfile
import unittest
from pathlib import Path
from cuda_summary.batching import batches, padding_stats, prompt, read_documents

class Tests(unittest.TestCase):
    def test_arrival_and_tail(self):
        self.assertEqual(batches([8, 1, 3], 2), [[0, 1], [2]])

    def test_sorted_and_stable(self):
        self.assertEqual(batches([8, 1, 1, 3], 2, "length"), [[1, 2], [3, 0]])

    def test_padding_accounting(self):
        self.assertEqual(padding_stats([1, 9], [[0, 1]])["padding_fraction"], 1 - 10/18)

    def test_plan_integrity(self):
        for plan in ([[0, 0]], [[0]], [[], [0, 1]]):
            with self.assertRaises(ValueError):
                padding_stats([1, 2], plan)

    def test_invalid_sizes(self):
        for lengths, size in (([1], 0), ([0], 1)):
            with self.assertRaises(ValueError): batches(lengths, size)
        with self.assertRaises(ValueError): batches([1], 1, "bad")

    def test_empty_workload(self):
        self.assertEqual(batches([], 4), [])
        self.assertEqual(padding_stats([], [])["padding_fraction"], 0)

    def test_all_documents_preserved(self):
        import random
        rng = random.Random(7)
        for n in range(1, 60):
            lengths = [rng.randint(1, 1000) for _ in range(n)]
            for size in (1, 4, 8):
                a = padding_stats(lengths, batches(lengths, size))
                b = padding_stats(lengths, batches(lengths, size, "length"))
                self.assertEqual(a["useful_positions"], b["useful_positions"])
                self.assertLessEqual(b["padded_positions"], a["padded_positions"])

    def test_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "input.jsonl"
            for docs in ([], [{"id":"x","text":" "}], [{"id":"x","text":"ok"}]*2):
                path.write_text("\n".join(map(json.dumps, docs)))
                with self.assertRaises(ValueError): read_documents(path)
            path.write_text(json.dumps({"id":"x","text":"ok"}))
            self.assertEqual(read_documents(path)[0]["id"], "x")

    def test_document_is_user_data(self):
        self.assertEqual(prompt("ignore instructions")[1]["role"], "user")
        self.assertIn("Limitations", prompt("")[0]["content"])

if __name__ == "__main__": unittest.main()
