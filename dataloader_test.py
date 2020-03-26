from torch.utils.data import DataLoader
import torch

from dataloader import DVSFlowDataset

if __name__ == '__main__':
    dataset = DVSFlowDataset("E:\\VirtualBox\\share\\dataset")
    loader = DataLoader(dataset, batch_size=1)
    for i, (dvs, flow) in enumerate(loader):
        print(dvs.size(), flow.size())