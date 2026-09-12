import unittest
import torch
import torch.nn.functional as F

from byte_bpe import ByteBPETokenizer
from deerlight_gpt import DeerlightGPTLanguageModel
from sft_data import SFTDataCollator, encode_prompt, IGNORE_INDEX


class SFTDataTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteBPETokenizer(280)
        self.tokenizer.train("Hello friend Hello Hi")
        self.examples = [{"user": "Hi", "assistant": "Hello"},
                         {"user": "Hi", "assistant": "Hello friend"}]
        self.collator = SFTDataCollator(self.tokenizer)

    def test_shift_mask_padding_and_vocab_unchanged(self):
        vocabulary = self.tokenizer.vocab.copy()
        batch = self.collator(self.examples)
        for i, example in enumerate(self.examples):
            prompt = encode_prompt(self.tokenizer, example["user"])
            answer = self.tokenizer.encode(example["assistant"]) + [self.tokenizer.EOS_ID]
            sequence = prompt + answer
            length = len(sequence) - 1
            self.assertEqual(batch["input_ids"][i, :length].tolist(), sequence[:-1])
            self.assertTrue((batch["labels"][i, :len(prompt)-1] == IGNORE_INDEX).all())
            self.assertEqual(batch["labels"][i, len(prompt)-1:length].tolist(), answer)
            self.assertTrue((batch["labels"][i, length:] == IGNORE_INDEX).all())
            self.assertTrue((batch["input_ids"][i, length:] == self.tokenizer.PAD_ID).all())
        self.assertEqual(self.tokenizer.vocab, vocabulary)

    def test_empty_answer_unicode_and_rejections(self):
        batch = self.collator([{"user": "鹿", "assistant": ""}])
        self.assertEqual(batch["labels"][batch["labels"] != IGNORE_INDEX].tolist(), [257])
        with self.assertRaises(ValueError):
            self.collator([])
        with self.assertRaises(ValueError):
            SFTDataCollator(self.tokenizer, max_length=2)(self.examples)

    def test_model_loss_mean_backward_and_padding_invariance(self):
        torch.manual_seed(1337)
        model = DeerlightGPTLanguageModel(vocab_size=self.tokenizer.vocab_size,
            embedding_dim=16, num_heads=2, num_kv_heads=1, num_layers=1,
            block_size=64, feed_forward_dim=32, attention_backend="manual")
        batch = self.collator(self.examples)
        logits, loss, _ = model(batch["input_ids"], batch["labels"])
        valid = batch["labels"] != IGNORE_INDEX
        expected = F.cross_entropy(logits[valid], batch["labels"][valid], reduction="sum") / valid.sum()
        torch.testing.assert_close(loss, expected)
        logits.retain_grad()
        loss.backward()
        self.assertTrue((logits.grad[~valid] == 0).all())
        self.assertTrue(torch.isfinite(model.token_embedding_table.weight.grad).all())
        # Check the shorter sample's real logits alone vs right-padded in batch.
        single = self.collator(self.examples[:1])
        model.eval()
        with torch.no_grad():
            alone = model(single["input_ids"])[0]
            together = model(batch["input_ids"])[0]
        torch.testing.assert_close(alone[0], together[0, :alone.shape[1]], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
