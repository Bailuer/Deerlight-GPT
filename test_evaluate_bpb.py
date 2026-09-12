import math
import unittest

import torch

from byte_bpe import ByteBPETokenizer
from evaluate_bpb import score_bpb, target_blocks


class UniformModel(torch.nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.vocabulary = vocabulary

    def forward(self, x):
        return torch.zeros(*x.shape, self.vocabulary, device=x.device), None, None


class BPBTests(unittest.TestCase):
    def test_target_coverage_and_tail(self):
        for length in range(2, 15):
            for block_size in (1, 3, 8):
                ids = list(range(length))
                blocks = list(target_blocks(ids, block_size))
                self.assertEqual([t for _, targets in blocks for t in targets], ids[1:])
                for x, y in blocks:
                    self.assertEqual(len(x), len(y))
                    self.assertLessEqual(len(x), block_size)
                    self.assertEqual(x[1:], y[:-1])

    def test_uniform_unicode_bytes_and_mode_restore(self):
        tokenizer = ByteBPETokenizer(261)
        tokenizer.train("banana")
        ids = tokenizer.encode("banana鹿")
        model = UniformModel(tokenizer.vocab_size)
        result = score_bpb(model, tokenizer, ids, 2)
        self.assertTrue(model.training)
        self.assertEqual(result["excluded_prefix_bytes"], 3)  # ban
        self.assertEqual(result["target_bytes"], 6)  # ana + 3 UTF-8 bytes
        expected = (len(ids) - 1) * math.log(tokenizer.vocab_size)
        self.assertAlmostEqual(result["total_nll_nats"], expected, places=5)
        self.assertAlmostEqual(result["bpb"], result["uniform_bpb"], places=5)

    def test_invalid_inputs(self):
        tokenizer = ByteBPETokenizer()
        model = UniformModel(tokenizer.vocab_size)
        for ids in ([], [97], [256, 97], [97, 9999]):
            with self.assertRaises(ValueError):
                score_bpb(model, tokenizer, ids, 2)
        with self.assertRaises(ValueError):
            list(target_blocks([1, 2], 0))


if __name__ == "__main__":
    unittest.main()
