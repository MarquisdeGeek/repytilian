import torch
import math
from src.sampler import Sampler


class BasicTruncate(Sampler):
    def __init__(self, percentage_to_train: float = 0.9):
        self.percentage = percentage_to_train


    def split(self, data: torch.Tensor):
        n = int(self.percentage * len(data))

        training = data[:n]
        validation = data[n:]

        return training, validation


    def __str__(self):
        return f"sampler (basic truncate): {math.floor(self.percentage * 100)}% training"

