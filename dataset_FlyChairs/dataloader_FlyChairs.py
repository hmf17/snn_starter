import os

import torch
from torch.utils.data import Dataset


class DVSDataset(Dataset):
    def __init__(self, root, size, time_window=5, train=True):
        if train:
            path = os.path.join(root, "train")
        else:
            path = os.path.join(root, "test")
        self.flow_path = os.path.join(path, "flow")
        self.events_path = os.path.join(path, "events")
        self.size = size
        self.time_window = time_window

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        name = '{:0=5}'.format(index)
        flow = torch.load(os.path.join(self.flow_path, name))
        events_raw = torch.load(os.path.join(self.events_path, name))
        events = torch.zeros([self.time_window, 2, 1024, 436], dtype=torch.int8)
        for i in range(self.time_window):
            events[i, 0, :, :] = events_raw[:, :, i]
            events[i, 1, :, :] = events_raw[:, :, i+1]
        return events, flow
