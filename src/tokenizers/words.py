from src.tokenizer import Tokenizer


class TokenizerWords(Tokenizer):
  def __init__(self, source: str):
    words = sorted(list(set(source.split())))

    super().__init__('words', words, source)


  def encode(self, s:str) -> list:
    stoi = { ch:i for i,ch in enumerate(self.tokens) }
    return [stoi[c] for c in s.split()]


  def decode(self, l: list) -> str:
    itos = { i:ch for i,ch in enumerate(self.tokens) }
    return ' '.join([itos[i] for i in l])


  def __str__(self):
    return f"tokenzier (words): {super().__str__()}"

