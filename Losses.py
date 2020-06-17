import torch
import torch.nn as nn
# from torchvision.transforms import Resize, ToTensor, ToPILImage

alphas = [0.32, 0.08, 0.02, 0.01, 0.005]
epsilon = 0.001
q = 0.8

"""
EPE: end-point error, the common index for optical flow prediction
"""
def EPE(target_flow, input_flow):
    torch.norm(target_flow - input_flow, p=2, dim=1).mean()


'''
This method is hard to calculate the grad correctly
'''
# def TensorScale(prev_img, shape):
#     '''
#     :param prev: [B, 2, H, W]
#     :param shape: target shape
#     :return: shaped image
#     '''
    # transToPTL = ToPILImage()
    # transToTensor = ToTensor()
    # transResize = Resize(shape)
    #
    # batchsize, _, prev_W, prev_H = prev_img.shape
    # prev_img_toCount = prev_img[0, :, :, :]
    # prev_img_toCount = transToPTL(prev_img_toCount)
    # target_img_toCount = transResize(prev_img_toCount)
    # target_img = transToTensor(target_img_toCount)
    # for i in range(1,batchsize):
    #     prev_img_toCount = prev_img[i,:,:,:]
    #     prev_img_toCount = transToPTL(prev_img_toCount)
    #     target_img_toCount = transResize(prev_img_toCount)
    #     target_img = torch.stack((target_img, transToTensor(target_img_toCount)), dim=0)
    # return target_img




class Train_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, flow_pre: list, flow_gt: torch.Tensor):
        """
        :param flow: Optical flow predicted from Pyramid-Optical-Flow-Estimator in format of list
        :param flow_gt: Optical ground truths in format of [B, 2, H, W]
        :return: the training loss
        """
        flow_gt_toCount = flow_gt.permute((0,3,1,2))
        total_loss = 0
        num_of_pyr = len(flow_pre)
        avg_pool = nn.AvgPool2d(kernel_size=2, padding=0, stride=2)
        # model outputs a quarter resolution
        flow_gt_toCount = avg_pool(flow_gt_toCount)
        # use zeropadding to fit the input image size
        Padding_list = [3, 0, 0, 0, 0]

        for level in range(num_of_pyr):
            flow_pre_toCount = flow_pre[level]
            _, _, W_flow_pre, H_flow_pre = flow_gt_toCount.shape
            ZeroLeftPadding = nn.ZeroPad2d(padding=(Padding_list[level], 0, 0, 0))
            flow_gt_toCount = ZeroLeftPadding(flow_gt_toCount)
            level_norn = torch.norm(flow_gt_toCount.cpu() - flow_pre_toCount.cpu(), p=1, dim=1)
            level_loss = torch.mean(level_norn, dim=(0, 1, 2))
            total_loss += alphas[level] * torch.pow(level_loss + epsilon, q)
            flow_gt_toCount = avg_pool(flow_gt_toCount)

        return total_loss
