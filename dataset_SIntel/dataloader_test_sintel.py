from torch.utils.data import DataLoader
import torch

from dataset_SIntel.dataloader_sintel import DVSFlowDataset

if __name__ == '__main__':
    dataset = DVSFlowDataset("/home/CBICR/hmf/dataset/use", 5, preload=True)
    loader = DataLoader(dataset, batch_size=1)
    for i, (dvs, flow) in enumerate(loader):
        print(dvs.shape, flow.shape)