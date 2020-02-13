import csv
import os
import random
import struct

import torch
from torch.utils.data import Dataset


# Read file list
def read_list(path):
    adept_list = []
    label_list = []
    with open(path, "r") as f:
        for line in f:
            adept_list.append(line[0:-1])
            label_list.append(line.split(".")[0] + "_labels.csv")

    return adept_list, label_list


# Read .aedat as timestream
def read_aedat(path):
    # Init
    polarities = []
    ys = []
    xs = []
    timestamps = []

    print(path)
    # Open file
    f = open(path, "rb")
    # Skip header
    f.seek(105)

    while True:
        event = f.read(28)
        if not event:
            break
        event_type = struct.unpack('H', event[0:2])[0]
        event_number = struct.unpack('I', event[20:24])[0]

        # Read Data
        if event_type != 1:
            print("Other type: %d".format(event_type))
        else:
            for i in range(event_number):
                data = f.read(8)
                buf = struct.unpack('I', data[0:4])[0]
                x = (buf >> 17) & 0x00001FFF
                y = (buf >> 2) & 0x00001FFF
                polarity = (buf >> 1) & 0x00000001
                timestamp = struct.unpack('I', data[4:8])[0]

                polarities.append(polarity)
                ys.append(y)
                xs.append(x)
                timestamps.append(timestamp)

    # Close file
    f.close()

    return len(polarities), polarities, ys, xs, timestamps


# Read label list
def read_labels(path):
    f = open(path, "r")
    r = csv.reader(f)

    label_per = []
    label_start = []
    label_end = []
    for line in r:
        if r.line_num == 1:
            continue
        label_per.append(int(line[0]))
        label_start.append(int(line[1]))
        label_end.append(int(line[2]))
    return label_per, label_start, label_end


class SNNDataset(Dataset):
    def __init__(self, root, batch_size=20, size=10000, train=True, timewindow=16000, window=20, preload=False):
        self.batch_size = batch_size
        self.size = size
        self.root = root
        self.timewindow = timewindow
        self.window = window

        if preload:
            if train:
                self.frames_list = torch.load(os.path.join(root, "train_frames.data"))
                self.label_list = torch.load(os.path.join(root, "train_label.data"))
                print("Using preloaded data to train")
            else:
                self.frames_list = torch.load(os.path.join(root, "test_frames.data"))
                self.label_list = torch.load(os.path.join(root, "test_label.data"))
                print("Using preloaded data to test")
        else:
            if train:
                self.adept_file_list, self.label_file_list = read_list(os.path.join(root, "trials_to_train.txt"))
            else:
                self.adept_file_list, self.label_file_list = read_list(os.path.join(root, "trials_to_test.txt"))

            self.frames_list = []
            self.label_list = []

            for i in range(len(self.adept_file_list)):
                length, polarities, ys, xs, timestamps = read_aedat(os.path.join(root, self.adept_file_list[i]))
                label_per, label_start, label_end = read_labels(os.path.join(root, self.label_file_list[i]))
                event_p = 0
                label_p = 0
                window_num = int((label_end[label_p] - label_start[label_p]) / self.timewindow)
                frames = torch.ShortTensor(window_num, 64, 64).fill_(0)
                while timestamps[event_p] < label_end[-1]:
                    if timestamps[event_p] < label_start[label_p]:
                        event_p += 1
                        continue
                    elif timestamps[event_p] < label_start[label_p] + timewindow * window_num:
                        if polarities[event_p] == 0:
                            frames[int((timestamps[event_p] - label_start[label_p]) / self.timewindow), int(
                                xs[event_p] / 2), int(ys[event_p] / 2)] = -1
                        else:
                            frames[int((timestamps[event_p] - label_start[label_p]) / self.timewindow), int(
                                xs[event_p] / 2), int(ys[event_p] / 2)] = 1
                        event_p += 1
                        continue
                    else:
                        self.frames_list.append(frames)
                        self.label_list.append(label_per[label_p])
                        label_p += 1
                        if label_p >= len(label_per):
                            break
                        window_num = int((label_end[label_p] - label_start[label_p]) / self.timewindow)
                        frames = torch.ShortTensor(window_num, 64, 64).fill_(0)
                        continue

            if train:
                torch.save(self.frames_list, os.path.join(root, "train_frames.data"))
                torch.save(self.label_list, os.path.join(root, "train_label.data"))
            else:
                torch.save(self.frames_list, os.path.join(root, "test_frames.data"))
                torch.save(self.label_list, os.path.join(root, "test_label.data"))

    def __getitem__(self, item):
        label_p = random.randint(0, len(self.label_list) - 1)
        frame_p = random.randint(0, self.frames_list[label_p].size()[0] - 7 - self.window)

        frames = []
        for i in range(self.window):
            frames.append(self.frames_list[label_p][frame_p + i:frame_p + i + 6, :, :].clone())
        frames = torch.stack(frames)
        label = self.label_list[label_p]

        return frames, label

    def __len__(self):
        return self.size


