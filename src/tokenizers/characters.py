from src.tokenizer import Tokenizer


class TokenizerCharacters(Tokenizer):
  def __init__(self, source: str):
    chars = sorted(list(set(source)))

    super().__init__('characters', chars, source)


  def encode(self, s:str) -> list:
    stoi = { ch:i for i,ch in enumerate(self.tokens) }
    return [stoi[c] for c in s]


  def decode(self, l: list) -> str:
    itos = { i:ch for i,ch in enumerate(self.tokens) }
    return ''.join([itos[i] for i in l])


  def __str__(self):
    # TODO: Consider NL/ws to be shown as descriptions
    return f"tokenzier (characters): {super().__str__()} ({', '.join(f'{i} = {ch}' for i,ch in enumerate(self.tokens))}"

