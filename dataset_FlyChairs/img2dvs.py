import re
import random
import progressbar
import numpy as np
import imageio
import os
import torch


root = "/data/dataset/dataset/hmf/ChairsSDHom/data/train"
total = 3000
time_window = 5

t0_root = os.path.join(root, "t0")
t1_root = os.path.join(root, "t1")
flow_root = os.path.join(root, "flow")
train_flow_root = os.path.join(root, "train", "flow")
train_events_root = os.path.join(root, "train", "events")
test_flow_root = os.path.join(root, "test", "flow")
test_events_root = os.path.join(root, "test", "events")



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


def gen_event(td):
    e_map = td[:, :, 0] / .15
    event_map = np.zeros((1024, 436, time_window + 1))

    num_list = range(time_window + 1)
    for y in range(436):
        for x in range(1024):
            br = e_map[x, y]
            event_num = int(abs(br))
            if event_num == 0:
                continue
            elif event_num >= time_window + 1:
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
    events = gen_event(br)
    return flow, events


if __name__ == '__main__':
    if not os.path.exists(train_flow_root):
        os.makedirs(train_flow_root)
    if not os.path.exists(train_events_root):
        os.makedirs(train_events_root)
    if not os.path.exists(test_flow_root):
        os.makedirs(test_flow_root)
    if not os.path.exists(test_events_root):
        os.makedirs(test_events_root)

    train_num = 129 # TODO: you should change this parameter according to the precessed number
    test_num = 15   # TODO: you should change this parameter according to the precessed number
    record_checkpoint = train_num + test_num
    bar = progressbar.ProgressBar()
    for i in bar(range(record_checkpoint, total)):
        flow, events = gen_data(i)
        flow = torch.tensor(flow, dtype=torch.float)
        events = torch.tensor(events, dtype=torch.int8)
        if i % 10 == 0:
            name = '{:0=5}'.format(test_num)
            test_num += 1
            torch.save(flow, os.path.join(test_flow_root, name))
            torch.save(events, os.path.join(test_events_root, name))
            print("Saving %s/%s test events and flow data" %(test_num, total))
        else:
            name = '{:0=5}'.format(train_num)
            train_num += 1
            torch.save(flow, os.path.join(train_flow_root, name))
            torch.save(events, os.path.join(train_events_root, name))
            print("Saving %s/%s train events and flow data" % (train_num, total))
