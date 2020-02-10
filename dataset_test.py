from torch.utils.data import DataLoader
import torch

from snn_dataset import SNNDataset

if __name__ == '__main__':
    dataset = SNNDataset("./dataset", size=5, preload=True)
    loader = DataLoader(dataset, batch_size=1)
    for i, (frames, labels) in enumerate(loader):
        print(torch.nonzero(frames), labels)
