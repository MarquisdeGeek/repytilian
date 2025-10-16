from src.tokenizer import Tokenizer
import tiktoken

# TODO: This tokenizer is a WIP

# https://github.com/openai/tiktoken/blob/main/tiktoken/core.py

class TokenizerTiktoken(Tokenizer):
  def __init__(self, source: str):

    self.enc = tiktoken.get_encoding("o200k_base")

    # Some special character tokens throw, so wrap it
    tokens = []
    for i in range(self.enc.n_vocab):
      try:
        tokens.append(self.enc.decode([i]))
      except:
        tokens.append('')

    super().__init__('tiktoken', tokens, source)


  def encode(self, s:str) -> list:
    return self.enc.encode(s)


  def decode(self, l: list) -> str:
    return self.enc.decode(l)


  def __str__(self):
    return f"tokenzier (tiktoken): {super().__str__()}"

