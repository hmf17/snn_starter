import torch
import os
import matplotlib.pyplot as plt
import numpy as np

root = "D:\\Dataset\\train"
flow_list = torch.load(os.path.join(root, "flow.train.data"))
events_list = torch.load(os.path.join(root, "events.train.data"))
root = os.path.join(root, "final")


def show_flow(index):
    plt.imshow(np.transpose(flow_list[index].numpy()[:, :, 0]))
    plt.show()


def show_events(index, window=0):
    plt.imshow(np.transpose(events_list[index].numpy()[:, :, window]), cmap='bwr')
    plt.show()


def save_flow(index):
    plt.imshow(np.transpose(flow_list[index].numpy()[:, :, 0]))
    plt.savefig(os.path.join(root, '{:0=5}flow.png'.format(index)))


def save_events(index, window=0):
    plt.imshow(np.transpose(events_list[index].numpy()[:, :, window]), cmap='bwr')
    plt.savefig(os.path.join(root, '{:0=5}event{:0=1}.png'.format(index, window)))


if __name__ == '__main__':
    for i in range(10):
        save_flow(i)
        save_events(i)
