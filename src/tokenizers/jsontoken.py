from src.tokenizer import Tokenizer
import json


class TokenizerJsontoken(Tokenizer):
  def __init__(self, source: str):

    if source == "":
        tokens = []
        dataset = []
    else:
        data = json.loads(source) if source else []

        dataset = data['value_list']
        tokens = list(set(dataset))

        # Our version for dev purposes
        self._token_list = tokens

    # Since we only process strings, convert the list into one
    dataset_string = json.dumps(dataset)

    super().__init__('jsontoken', tokens, dataset_string)



  # s is a JSON literal holding a list of (un-tokenized) values
  def encode(self, s:str) -> list:
    value_list = json.loads(s)

    vtoi = { ch:i for i,ch in enumerate(self.tokens) }
    return [vtoi[c] for c in value_list ]


  # input (l) is the list of tokens, converted back to a string holding values
  def decode(self, l: list) -> str:
    return json.dumps([ self.tokens[idx] for idx in l ])


  def __str__(self):
    return f"tokenzier (jsontoken): {super().__str__()}"

