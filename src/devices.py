import torch

class Devices:

    @staticmethod
    def get_best_device() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
