import torch
import torch.nn as nn
import sys
import numpy as np



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


def bilinear_interpolation(src, dst_size, align_corners=False):
    """
    双线性插值高效实现
    :param src: 源图像H*W*C
    :param dst_size: 目标图像大小H*W
    :return: 双线性插值后的图像
    """
    (src_h, src_w, src_c) = src.shape  # 原图像大小 H*W*C
    (dst_h, dst_w), dst_c = dst_size, src_c  # 目标图像大小H*W*C

    if src_h == dst_h and src_w == dst_w:  # 如果大小不变，直接返回copy
        return src.copy()
    # 矩阵方式实现
    h_d = np.arange(dst_h)  # 目标图像H方向坐标
    w_d = np.arange(dst_w)  # 目标图像W方向坐标
    if align_corners:
        h = float(src_h - 1) / (dst_h - 1) * h_d
        w = float(src_w - 1) / (dst_w - 1) * w_d
    else:
        h = float(src_h) / dst_h * (h_d + 0.5) - 0.5  # 将目标图像H坐标映射到源图像上
        w = float(src_w) / dst_w * (w_d + 0.5) - 0.5  # 将目标图像W坐标映射到源图像上

    h = np.clip(h, 0, src_h - 1)  # 防止越界，最上一行映射后是负数，置为0
    w = np.clip(w, 0, src_w - 1)  # 防止越界，最左一行映射后是负数，置为0

    h = np.repeat(h.reshape(dst_h, 1), dst_w, axis=1)  # 同一行映射的h值都相等
    w = np.repeat(w.reshape(dst_w, 1), dst_h, axis=1).T  # 同一列映射的w值都相等

    h0 = np.floor(h).astype(np.int)  # 同一行的h0值都相等
    w0 = np.floor(w).astype(np.int)  # 同一列的w0值都相等

    h0 = np.clip(h0, 0, src_h - 2)  # 最下一行上不大于src_h - 2，相当于padding
    w0 = np.clip(w0, 0, src_w - 2)  # 最右一列左不大于src_w - 2，相当于padding

    h1 = np.clip(h0 + 1, 0, src_h - 1)  # 同一行的h1值都相等，防止越界
    w1 = np.clip(w0 + 1, 0, src_w - 1)  # 同一列的w1值都相等，防止越界

    q00 = src[h0, w0]  # 取每一个像素对应的q00
    q01 = src[h0, w1]  # 取每一个像素对应的q01
    q10 = src[h1, w0]  # 取每一个像素对应的q10
    q11 = src[h1, w1]  # 取每一个像素对应的q11

    h = np.repeat(h[..., np.newaxis], dst_c, axis=2)  # 图像有通道C，所有的计算都增加通道C
    w = np.repeat(w[..., np.newaxis], dst_c, axis=2)
    h0 = np.repeat(h0[..., np.newaxis], dst_c, axis=2)
    w0 = np.repeat(w0[..., np.newaxis], dst_c, axis=2)
    h1 = np.repeat(h1[..., np.newaxis], dst_c, axis=2)
    w1 = np.repeat(w1[..., np.newaxis], dst_c, axis=2)

    r0 = (w1 - w) * q00 + (w - w0) * q01  # 双线性插值的r0
    r1 = (w1 - w) * q10 + (w - w0) * q11  # 双线性差值的r1
    q = (h1 - h) * r0 + (h - h0) * r1  # 双线性差值的q
    dst = q.astype(src.dtype)  # 图像的数据类型
    return dst


"""
optic flow utils
"""


def load_flow(path):
    with open(path, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count = 1)[0])
        if magic == 202021.25:
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data
        return None

def save_flow(path, flow):
    magic = np.array([202021.25], np.float32)
    h, w = flow.shape[:2]
    h, w = np.array([h], np.int32), np.array([w], np.int32)

    with open(path, 'wb') as f:
        magic.tofile(f); w.tofile(f); h.tofile(f); flow.tofile(f)



def makeColorwheel():

    #  color encoding scheme

    #   adapted from the color circle idea described at
    #   http://members.shaw.ca/quadibloc/other/colint.html

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3]) # r g b

    col = 0
    #RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY

    #YG
    colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col:YG+col, 1] = 255
    col += YG

    #GC
    colorwheel[col:GC+col, 1]= 255
    colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC

    #CB
    colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col:CB+col, 2] = 255
    col += CB

    #BM
    colorwheel[col:BM+col, 2]= 255
    colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM

    #MR
    colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col:MR+col, 0] = 255
    return 	colorwheel


def computeColor(u, v):

    colorwheel = makeColorwheel()
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v)

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
    k1 = k0+1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1],3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius
        col[~idx] *= 0.75 # out of range
        img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

    return img.astype(np.uint8)


def vis_flow(flow):
    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10

    u = flow[:,:,0]
    v = flow[:,:,1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999

    maxrad = -1
    #fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0
    v[greater_v] = 0

    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])

    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v))
    maxrad = max([maxrad, np.amax(rad)])
    # print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

    u = u/(maxrad+eps)
    v = v/(maxrad+eps)
    img = computeColor(u, v)
    return img[:,:,[2,1,0]]


if __name__ == '__main__':
    flow = load_flow('dataset/training/flow/alley_1/frame_0001.flo')
    img = vis_flow(flow)
    import  imageio
    imageio.imsave('predict_result/test.png', img)