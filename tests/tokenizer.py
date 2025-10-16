import unittest
import src.tokenizer
from src.llm import LLM
from src.devices import Devices
from src.training_data import TrainingData
from src.samplers.truncate import BasicTruncate

class TestTokenizer(unittest.TestCase):
    def test_basics(self):
        tokenizer_inst = src.tokenizer.tokenizeCharacters("ABCD")
        self.assertEqual([0], tokenizer_inst.encode("A"))
        self.assertEqual("CD", tokenizer_inst.decode([2,3]))
        self.assertEqual("DCBA", tokenizer_inst.decode(tokenizer_inst.encode("DCBA")))


    def test_words(self):
        tokenizer_inst = src.tokenizer.tokenizeWords("DCBA")
        self.assertEqual("DCBA", tokenizer_inst.decode(tokenizer_inst.encode("DCBA")))


    def test_tiktoken(self):
        tokenizer_inst = src.tokenizer.tokenizeTiktoken("DCBA")
        self.assertEqual("DCBA", tokenizer_inst.decode(tokenizer_inst.encode("DCBA")))


    def test_get_batch(self):
        device = Devices.getBestDevice()

        tokenizer_inst = src.tokenizer.importTokens('models/shakespeare/basic')

        sampler = BasicTruncate(0.9)
        data = TrainingData(device, tokenizer_inst, sampler)
        xb,yb = data.get_batch(LLM.TRAINING, 1, 16)

        self.assertEqual(xb.size(), yb.size())

        # The two batches, xb & yb, are connected such that the first element
        # of yb (target) matches the second of xb (the input context).
        for i, x in enumerate(xb[0]):
            if i:
                self.assertEqual(x, yb[0][i-1])
