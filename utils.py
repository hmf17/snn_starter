import torch.nn as nn


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm = False):
    if batch_norm:
        return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1)
                            )
    else:
        return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
                nn.LeakyReLU(0.1)
                            )


def predict_flow(in_planes):# out_planes = 2: output the predicted optic flow
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=0):
    # 逆卷积层，将特征图反向生成图片，应用在光流生成器这一部分中
    # 本质是利用deconv的方法进行upsample，这里在FlowNet的论文中也有提及
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, output_padding, bias=True)