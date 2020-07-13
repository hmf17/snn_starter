import os

import torch
from torch.utils.data import Dataset


class DVSDataset(Dataset):
    def __init__(self, root, train=True):
        if train:
            self.flow_list = torch.load(os.path.join(root, "flow.train.data"))
            self.events_list = torch.load(os.path.join(root, "events.train.data"))
        else:
            self.flow_list = torch.load(os.path.join(root, "flow.test.data"))
            self.events_list = torch.load(os.path.join(root, "events.test.data"))

    def __len__(self):
        return len(self.flow_list)

    def __getitem__(self, index):
        return self.events_list[index], self.flow_list[index]
