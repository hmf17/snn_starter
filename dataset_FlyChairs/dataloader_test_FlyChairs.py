from torch.utils.data import DataLoader

from dataset_FlyChairs.dataloader_FlyChairs import DVSDataset

if __name__ == '__main__':
    dataset = DVSDataset("D:\\Dataset\\train", 9)
    loader = DataLoader(dataset, batch_size=1)
    for i, (events, flow) in enumerate(loader):
        print(events.size(), flow.size())
