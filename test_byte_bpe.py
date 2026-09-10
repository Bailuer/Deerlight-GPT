"""Run with: python -m unittest test_byte_bpe -v"""

import json
from pathlib import Path
import random
import tempfile
import unittest

from byte_bpe import ByteBPETokenizer, count_pairs, merge_pair


class ByteBPETests(unittest.TestCase):
    def test_pairs_and_overlap(self):
        self.assertEqual(count_pairs([98, 97, 110, 97, 110, 97]),
                         {(98, 97): 1, (97, 110): 2, (110, 97): 2})
        self.assertEqual(count_pairs([97, 97, 97]), {(97, 97): 2})
        self.assertEqual(merge_pair([97, 97, 97], (97, 97), 259), [259, 97])
        self.assertEqual(count_pairs([]), {})
        self.assertEqual(merge_pair([], (1, 2), 259), [])

    def test_training_and_rank(self):
        tokenizer = ByteBPETokenizer(261)
        tokenizer.train("banana")
        self.assertEqual(list(tokenizer.merges.items()),
                         [((97, 110), 259), ((98, 259), 260)])
        self.assertEqual(tokenizer.encode("banana"), [260, 259, 97])
        self.assertEqual(tokenizer.vocab[260], b"ban")
        old_rules = tokenizer.merges.copy()
        tokenizer.encode("new words")
        self.assertEqual(tokenizer.merges, old_rules)

    def test_unicode_roundtrip(self):
        tokenizer = ByteBPETokenizer(280)
        tokenizer.train("banana 鹿光 hello hello")
        rng = random.Random(1337)
        texts = ["", "中文 🦌", "e\u0301\x00\n", "<BOS>"]
        texts += ["".join(rng.choices("abc 鹿🦌é\n", k=30)) for _ in range(100)]
        for text in texts:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_documents_and_small_corpus(self):
        tokenizer = ByteBPETokenizer(512)
        tokenizer.train(["a", "b", ""])
        self.assertEqual(tokenizer.vocab_size, 259)
        self.assertEqual(tokenizer.merges, {})
        tokenizer.train("ab")
        self.assertEqual(tokenizer.vocab_size, 260)
        tokenizer.train("")
        self.assertEqual(tokenizer.vocab_size, 259)

    def test_special_and_invalid_ids(self):
        tokenizer = ByteBPETokenizer()
        ids = tokenizer.encode("鹿", add_bos=True, add_eos=True)
        self.assertEqual(tokenizer.decode(ids), "<BOS>鹿<EOS>")
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), "鹿")
        self.assertNotIn(256, tokenizer.encode("<BOS>"))
        with self.assertRaises(ValueError):
            tokenizer.decode([9999])
        with self.assertRaises(UnicodeDecodeError):
            tokenizer.decode([255])
        self.assertEqual(tokenizer.decode([255], errors="replace"), "\ufffd")

    def test_save_load_and_validation(self):
        tokenizer = ByteBPETokenizer(275)
        tokenizer.train("banana banana 鹿光")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = ByteBPETokenizer.load(path)
            self.assertEqual(restored.vocab, tokenizer.vocab)
            self.assertEqual(restored.merges, tokenizer.merges)
            self.assertEqual(restored.encode("banana 鹿光"), tokenizer.encode("banana 鹿光"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["merges"][0][2] = 9999
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                ByteBPETokenizer.load(path)
        with self.assertRaises(ValueError):
            ByteBPETokenizer(258)


if __name__ == "__main__":
    unittest.main()
