import pickle
import json
from pathlib import Path

# NOTE: Imports for the specific tokenizers are at the end of the file

class Tokenizer:
  # Types of tokenizer supported
  # TODO: Move to better pattern
  OPTIONS = [ "characters", "words", "tiktoken", "jsontoken" ]
  DEFAULT = OPTIONS[0]

  def __init__(self, tokenizer_type: str, tokens: list, source: str):
    self.type = tokenizer_type
    # We store a copy of these tokens so they can be exported in JSON for dev purposes
    self.tokens = tokens
    self.encoded_data = self.encode(source) if source != "" else []


  # From the character symbols, to a token list
  def encode(self, s: str):
    pass

  # From a list of tokens, to a string
  def decode(self, l: list):
    pass


  def export_tokens(self, filepath: str):
    Path(filepath).mkdir(parents=True, exist_ok=True)

    with open(f"{filepath}/tokens.type", "w") as f:
      f.write(self.type)

    with open(f"{filepath}/tokens.pickle", "wb") as f:
      f.write(pickle.dumps(self))

    with open(f"{filepath}/tokens.json", "w") as f:
      f.write(json.dumps(self.tokens))


  @property
  def vocab_size(self):
      return len(self.tokens)
  

  def __str__(self):
    return f"{self.vocab_size}"



def tokenize(tokenizer_type: str, text: str = "") -> Tokenizer:
  fn = f"tokenize_{tokenizer_type}"
  return globals()[fn](text)


def tokenize_words(text: str) -> Tokenizer:
    return TokenizerWords(text)


def tokenize_characters(text: str) -> Tokenizer:
    return TokenizerCharacters(text)


def tokenize_tiktoken(text: str) -> Tokenizer:
    return TokenizerTiktoken(text)


def tokenize_jsontoken(text: str) -> Tokenizer:
    return TokenizerJsontoken(text)



def import_tokens(filepath: str):
  with open(f"{filepath}/tokens.type", "r") as f:
    tokenizer_type = f.read().strip()

  tokenizer = tokenize(tokenizer_type)

  with open(f"{filepath}/tokens.pickle", "rb") as f:
    data = pickle.loads(f.read())

  for k in ["tokens","encoded_data"]:
      setattr(tokenizer, k, getattr(data, k))

  return tokenizer



from src.tokenizers.characters import TokenizerCharacters
from src.tokenizers.words import TokenizerWords
from src.tokenizers.tiktoken import TokenizerTiktoken
from src.tokenizers.jsontoken import TokenizerJsontoken
