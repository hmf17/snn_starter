from torch.utils.data import DataLoader

from snn_dataset import SNNDataset

if __name__ == '__main__':
    dataset = SNNDataset(".//dataset", size=200)
    loader = DataLoader(dataset, batch_size=20, shuffle=True)
    for i, (frames, labels) in enumerate(loader):
        print(frames.size(), labels.size())
