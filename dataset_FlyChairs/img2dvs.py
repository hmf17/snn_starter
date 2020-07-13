import re
import random
import progressbar
import numpy as np
import imageio
import os
import torch


root = "D:\\Dataset\\train"
total = 25

t0_root = os.path.join(root, "t_0")
t1_root = os.path.join(root, "t_1")
flow_root = os.path.join(root, "train_flow")


def read_image(name):
    raw_image = imageio.imread(uri=name)
    return np.log(0.299 * raw_image[:, :, 0] + 0.587 * raw_image[:, :, 1] + 0.114 * raw_image[:, :, 2] + .0001)


def read_flow(name):
    file = open(name, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data[:, :, 0:2]


def resize(origin, dim):
    h_old, w_old = 384, 512
    h_new, w_new = 436, 1024

    scale_w = float(w_old) / w_new
    scale_h = float(h_old) / h_new

    new = np.zeros((w_new, h_new, dim), dtype=np.float)
    for y in range(h_new):
        for x in range(w_new):
            x0 = (x + 0.5) * scale_w - 0.5
            y0 = (y + 0.5) * scale_h - 0.5

            x1 = max(int(np.floor(x0)), 0)
            y1 = max(int(np.floor(y0)), 0)
            x2 = min(int(np.floor(x0)) + 1, w_old - 1)
            y2 = min(int(np.floor(y0)) + 1, h_old - 1)

            v1 = (x2 - x0) * origin[y1, x1] + (x0 - x1) * origin[y1, x2]
            v2 = (x2 - x0) * origin[y2, x1] + (x0 - x1) * origin[y2, x2]
            new[x, y] = (y2 - y0) * v1 + (y0 - y1) * v2

    return new


def gen_event(td, time_window):
    e_map = td[:, :, 0] / .15
    event_map = np.zeros((1024, 436, time_window))

    num_list = range(time_window)
    for y in range(436):
        for x in range(1024):
            br = e_map[x, y]
            event_num = int(abs(br))
            if event_num == 0:
                continue
            elif event_num >= time_window:
                if br > 0:
                    event_map[x, y] = 1
                else:
                    event_map[x, y] = -1
            else:
                event_list = random.sample(num_list, event_num)
                if br > 0:
                    for e in event_list:
                        event_map[x, y, e] = 1
                else:
                    for e in event_list:
                        event_map[x, y, e] = -1

    return event_map


def gen_data(index):
    name = '{:0=5}'.format(index)
    t0_path = os.path.join(t0_root, name + ".png")
    t1_path = os.path.join(t1_root, name + ".png")
    flow_path = os.path.join(flow_root, name + ".pfm")

    t0 = read_image(t0_path)
    t1 = read_image(t1_path)
    flow = read_flow(flow_path)

    br = resize(t1 - t0, 1)
    flow = resize(flow, 2)
    events = gen_event(br, 5)
    return flow, events


if __name__ == '__main__':
    events_train_list = []
    flow_train_list = []
    events_test_list = []
    flow_test_list = []
    for i in progressbar.progressbar(range(total)):
        flow, events = gen_data(i)
        flow = torch.tensor(flow, dtype=torch.float)
        events = torch.tensor(events, dtype=torch.int8)
        if i % 10 == 0:
            events_test_list.append(events)
            flow_test_list.append(flow)
        else:
            events_train_list.append(events)
            flow_train_list.append(flow)

    torch.save(events_train_list, os.path.join(root, "events.train.data"))
    torch.save(flow_train_list, os.path.join(root, "flow.train.data"))
    torch.save(events_test_list, os.path.join(root, "events.test.data"))
    torch.save(flow_test_list, os.path.join(root, "flow.test.data"))