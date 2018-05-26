import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTLoaders:
    """
    Data loaders for MNIST.
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def train_loader(self, data_dir, kwargs):
        loader = DataLoader(
          datasets.MNIST(
            data_dir, train=True, download=True,
            transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
          ),
          batch_size=self.batch_size, shuffle=True, **kwargs
        )

        return loader

    def test_loader(self, data_dir, kwargs):
        loader = torch.utils.data.DataLoader(
          datasets.MNIST(
            data_dir, train=False, 
            transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
            ),
            batch_size=self.batch_size, shuffle=True, **kwargs
          )
