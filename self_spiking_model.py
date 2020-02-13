# '''
# created by wyj;
# changed by hmf;
# '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.1 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 # decay constants
num_classes = 11
batch_size  = 500
learning_rate = 1e-3
num_epochs = 500 # max epoch



# define approximate firing function
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        #input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()
act_fun = ActFun.apply
# ActFun.apply()
# membrane potential update
# parameter sample decide sample or not
def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer




# 此前学长的定义保持不变，以下对于网络架构进行改变
# cnn_layer(in_planes, out_planes, stride, padding, kernel_size, groups)
cfg_cnn = [(6, 12, 2, 0, 3, 1),
           (12, 252, 2, 0, 4, 2),
           (252, 256, 1, 0, 1, 2),
           (256, 256, 2, 0, 2, 2),
           (256, 512, 1, 1, 3, 32),
           (512, 512, 1, 0, 1, 4),
           (512, 512, 1, 0, 1, 4),
           (512, 512, 1, 0, 1, 4),
           (512, 512, 2, 0, 2, 16),
           (512, 1024, 1, 1, 3, 64),
           (1024, 1024, 1, 0, 1, 8),
           (1024, 1024, 1, 0, 1, 8),
           (1024, 1024, 2, 0, 2, 32),
           (1024, 1024, 1, 0, 1, 8),
           (1024, 968, 1, 0, 1, 8),
           (968, 2640, 1, 0, 1, 8)]
cfg_kernel = [31, 14, 14, 7, 7, 7, 7, 7, 3, 3, 3, 3, 1, 1, 1, 1]
cfg_fc = [128, 11]

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()

        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[3]
        self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[4]
        self.conv5 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[5]
        self.conv6 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[6]
        self.conv7 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[7]
        self.conv8 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[8]
        self.conv9 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[9]
        self.conv10 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[10]
        self.conv11 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[11]
        self.conv12 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[12]
        self.conv13 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[13]
        self.conv14 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[14]
        self.conv15 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               groups=groups)
        in_planes, out_planes, stride, padding, kernel_size, groups = cfg_cnn[15]
        self.conv16 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                groups=groups)
        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])


    def forward(self, input, time_window = 20):
        #conv
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
        c4_mem = c4_spike = torch.zeros(batch_size, cfg_cnn[3][1], cfg_kernel[3], cfg_kernel[3], device=device)
        c5_mem = c5_spike = torch.zeros(batch_size, cfg_cnn[4][1], cfg_kernel[4], cfg_kernel[4], device=device)
        c6_mem = c6_spike = torch.zeros(batch_size, cfg_cnn[5][1], cfg_kernel[5], cfg_kernel[5], device=device)
        c7_mem = c7_spike = torch.zeros(batch_size, cfg_cnn[6][1], cfg_kernel[6], cfg_kernel[6], device=device)
        c8_mem = c8_spike = torch.zeros(batch_size, cfg_cnn[7][1], cfg_kernel[7], cfg_kernel[7], device=device)
        c9_mem = c9_spike = torch.zeros(batch_size, cfg_cnn[8][1], cfg_kernel[8], cfg_kernel[8], device=device)
        c10_mem = c10_spike = torch.zeros(batch_size, cfg_cnn[9][1], cfg_kernel[9], cfg_kernel[9], device=device)
        c11_mem = c11_spike = torch.zeros(batch_size, cfg_cnn[10][1], cfg_kernel[10], cfg_kernel[10], device=device)
        c12_mem = c12_spike = torch.zeros(batch_size, cfg_cnn[11][1], cfg_kernel[11], cfg_kernel[11], device=device)
        c13_mem = c13_spike = torch.zeros(batch_size, cfg_cnn[12][1], cfg_kernel[12], cfg_kernel[12], device=device)
        c14_mem = c14_spike = torch.zeros(batch_size, cfg_cnn[13][1], cfg_kernel[13], cfg_kernel[13], device=device)
        c15_mem = c15_spike = torch.zeros(batch_size, cfg_cnn[14][1], cfg_kernel[14], cfg_kernel[14], device=device)
        c16_mem = c16_spike = torch.zeros(batch_size, cfg_cnn[15][1], cfg_kernel[15], cfg_kernel[15], device=device)
        #fc
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        for step in range(time_window): # simulation time steps
            x = input
            x = x.float()
            c1_mem, c1_spike = mem_update(self.conv1, x, c1_mem, c1_spike)
            x = c1_spike
            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem, c2_spike)
            x = c2_spike
            c3_mem, c3_spike = mem_update(self.conv3, x, c3_mem, c3_spike)
            x = c3_spike
            c4_mem, c4_spike = mem_update(self.conv4, x, c4_mem, c4_spike)
            x = c4_spike
            c5_mem, c5_spike = mem_update(self.conv5, x, c5_mem, c5_spike)
            x = c5_spike
            c6_mem, c6_spike = mem_update(self.conv6, x, c6_mem, c6_spike)
            x = c6_spike
            c7_mem, c7_spike = mem_update(self.conv7, x, c7_mem, c7_spike)
            x = c7_spike
            c8_mem, c8_spike = mem_update(self.conv8, x, c8_mem, c8_spike)
            x = c8_spike
            c9_mem, c9_spike = mem_update(self.conv9, x, c9_mem, c9_spike)
            x = c9_spike
            c10_mem, c10_spike = mem_update(self.conv10, x, c10_mem, c10_spike)
            x = c10_spike
            c11_mem, c11_spike = mem_update(self.conv11, x, c11_mem, c11_spike)
            x = c11_spike
            c12_mem, c12_spike = mem_update(self.conv12, x, c12_mem, c12_spike)
            x = c12_spike
            c13_mem, c13_spike = mem_update(self.conv13, x, c13_mem, c13_spike)
            x = c13_spike
            c14_mem, c14_spike = mem_update(self.conv14, x, c14_mem, c14_spike)
            x = c14_spike
            c15_mem, c15_spike = mem_update(self.conv15, x, c15_mem, c15_spike)
            x = c15_spike
            c16_mem, c16_spike = mem_update(self.conv16, x, c16_mem, c16_spike)
            x = c16_spike
            x = x.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / time_window
        return outputs