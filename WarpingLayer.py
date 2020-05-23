import torch
import torch.nn as nn
import torch.nn.functional as F

# warp operation with none parameter
# do not need to refine to the SNN format
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WarpingLayer(nn.Module):
    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)  # [B, 1, H, W]
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)  # [B, 1, H, W]
        grid = torch.cat((xx, yy), 1).float()  # [B, 2, H, W]

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / (max(W - 1, 1) - 1.0)
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / (max(H - 1, 1) - 1.0)

        # warp layer use nn.functional.grid_sample() to do interpolation for input image with bilinear method
        vgrid = vgrid.permute(0, 2, 3, 1)  # [B, H, W, 2]
        output = F.grid_sample(x, vgrid)
        # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = torch.ones(x.size()).to(device)
        mask = F.grid_sample(mask, vgrid)

        # mask if consist of 0 and 1, and just like non-linear function
        # the format of mask is similar to the spike
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask