import torch
from abc import ABC, abstractmethod

class Sampler(ABC):

  @abstractmethod
  def split(self, _: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass
