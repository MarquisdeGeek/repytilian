import torch

class Devices:

    @staticmethod
    def getBestDevice() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
