import torch
from src.tokenizer import Tokenizer
from src.tokenizer import Tokenizer
from src.sampler import Sampler
from src.llm import LLM


class TrainingData:
  def __init__(self, device: str, tokenizer_inst: Tokenizer, sampler: Sampler):
    self.tokenizer_inst = tokenizer_inst

    self._device = device
    all_data = torch.tensor(tokenizer_inst.encoded_data, dtype=torch.long)

    self._training, self._validation = sampler.split(all_data)


  def get_batch(self, split:str, batch_size: int, block_size: int):
    data = self._training if split == LLM.TRAINING else self._validation

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    x,y = x.to(self._device), y.to(self._device)

    return x,y
