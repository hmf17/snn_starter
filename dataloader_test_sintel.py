from torch.utils.data import DataLoader
import torch

from dataloader_sintel import DVSFlowDataset

if __name__ == '__main__':
    dataset = DVSFlowDataset("dataset", 5, train=False, preload=False)
    loader = DataLoader(dataset, batch_size=1)
    for i, (dvs, flow) in enumerate(loader):
        print(dvs.shape, flow.shape)