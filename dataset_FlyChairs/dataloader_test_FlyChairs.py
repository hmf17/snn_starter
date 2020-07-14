from torch.utils.data import DataLoader

from dataloader import DVSDataset

if __name__ == '__main__':
    dataset = DVSDataset("D:\\Dataset\\train")
    loader = DataLoader(dataset, batch_size=1)
    for i, (events, flow) in enumerate(loader):
        print(events.size(), flow.size())
