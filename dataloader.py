import csv
import os
import struct

import torch
from torch.utils.data import Dataset


#   Read ".flo" file
#
#   bytes  contents
#   0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
#           (just a sanity check that floats are represented correctly)
#   4-7     width as an integer
#   8-11    height as an integer
#   12-end  data (width*height*2*4 bytes total)
#           the float values for u and v, interleaved, in row order, i.e.,
#           u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
#
#   return
#   FloatTensor(width, height, 2)
def read_flow(path):
    if path == "":
        raise Exception(str("Invalid path: %s" % path))

    f = open(path, 'rb')
    buf = f.read(12)
    tag, width, height = struct.unpack('fii', buf)

    if tag != 202021.25:
        raise Exception(str("Invalid header: %s" % path))

    if width < 1 or width > 99999:
        raise Exception(str("Invalid width: %s" % path))

    if height < 1 or height > 99999:
        raise Exception(str("Invalid height: %s" % path))

    flow = torch.FloatTensor(width, height, 2).fill_(0)
    for y in range(height):
        for x in range(width):
            buf = f.read(8)
            flow[x, y, 0], flow[x, y, 1] = struct.unpack('ff', buf)

    return flow


#    Read .dvs file
#
#    contents
#    line1      width, height
#    line2-end  t, x, y, polarity
#
#    return
#    ShortTensor(width, height)
def read_dvs(path1, path2, timewindow):
    if path1 == "" or path2 == "":
        raise Exception(str("Invalid path: %s, %s" % (path1, path2)))

    dvs_list = []
    dvs = None
    end = 0

    f = open(path1, 'r')
    r = csv.reader(f)

    for line in r:
        if r.line_num == 1:
            width = int(line[0])
            height = int(line[1])
            dvs = torch.ShortTensor(width, height).fill_(0)
        else:
            if r.line_num == 2:
                end = (int(line[0])/50000000)*50000000 + timewindow
            if int(line[0]) > end:
                dvs_list.append(dvs)
                dvs = torch.ShortTensor(width, height).fill_(0)
                end += timewindow
            if line[3] == 'True':
                dvs[int(line[1]), int(line[2])] = 1
            else:
                dvs[int(line[1]), int(line[2])] = -1

    f.close()

    f = open(path2, 'r')
    r = csv.reader(f)

    for line in r:
        if r.line_num != 1:
            if int(line[0]) > end:
                dvs_list.append(dvs)
                dvs = torch.ShortTensor(width, height).fill_(0)
                end += timewindow
            if line[3] == 'True':
                dvs[int(line[1]), int(line[2])] = 1
            else:
                dvs[int(line[1]), int(line[2])] = -1

    f.close()

    dvs_list.append(dvs)

    return dvs_list


#    Read .csv list file
#
#    contents
#    name, frame_num
#
#    return
#    list[[name, frame_num]]
def read_list(path):
    if path == "":
        raise Exception(str("Invalid path: %s" % path))

    f = open(path, 'r')
    r = csv.reader(f)

    l = []

    for line in r:
        l.append([line[0], int(line[1])])

    return l


#   Dataset generator
#
#   Feature
#   dvs_data ShortTensor(2, width, height)
#
#   Label
#   flow_data FloatTensor(width, height, 2)
class DVSFlowDataset(Dataset):
    def __init__(self, root, window, train=True, preload=False):
        if train:
            self.root = os.path.join(root, "training")
        else:
            self.root = os.path.join(root, "test")

        self.window = window

        if preload:
            self.dvs = torch.load(os.path.join(self.root, "dvs.data"))
            self.flow = torch.load(os.path.join(self.root, "flow.data"))

        else:
            self.dvs = []
            self.flow = []
            set_list = read_list(os.path.join(self.root, "list.csv"))
            for s in set_list:
                for i in range(s[1]):
                    dvs_1_path = os.path.join(self.root, "final", s[0], str("dvs_%04d" % (i+1)))
                    dvs_2_path = os.path.join(self.root, "final", s[0], str("dvs_%04d" % (i+2)))
                    flow_path = os.path.join(self.root, "flow", s[0], str("frame_%04d.flo" % (i+1)))
                    if os.path.exists(dvs_1_path) and os.path.exists(dvs_2_path) and os.path.exists(flow_path):
                        print(dvs_1_path, dvs_2_path, flow_path)
                        dvs = torch.stack(read_dvs(dvs_1_path, dvs_2_path, 100000000/(self.window+1)))
                        flow = read_flow(flow_path)
                        self.dvs.append(dvs)
                        self.flow.append(flow)

            torch.save(self.dvs, os.path.join(self.root, "dvs.data"))
            torch.save(self.flow, os.path.join(self.root, "flow.data"))

    def __len__(self):
        return len(self.dvs)

    def __getitem__(self, index):
        dvs = []
        for i in range(self.window):
            dvs.append(self.dvs[index][i:i+2])
        dvs = torch.stack(dvs)

        return dvs, self.flow[index]



