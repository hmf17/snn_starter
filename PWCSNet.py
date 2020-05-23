"""
PWC-Net redefined and Used for TiaoZhanBei Contest by hmf, wsh, lyf
"""
import os
from CostVolumeLayer import *
from SpikingModel import *
from WarpingLayer import *
from utils import *
import numpy as np
from torchsummary import summary

# parameter
os.environ['PYTHON_EGG_CACHE'] = 'tmp/' # a writable directory
warp_parameter = [0, 5, 2.5, 1.25, 0.625, 0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4 # 由于SCNN定义的问题，可能需要用到小批量的训练集合
time_windows = 5

# net


class PWCSNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """
    def __init__(self, md=4, training=True):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCSNet,self).__init__()
        self.training = training

        self.conv1a  = conv(1,   16, kernel_size=3, stride=2) # 1 channels input
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa = conv(32,  32, kernel_size=3, stride=1)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3aa = conv(64,  64, kernel_size=3, stride=1)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4aa = conv(96,  96, kernel_size=3, stride=1)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128,128, kernel_size=3, stride=1)
        self.conv5b  = conv(128,128, kernel_size=3, stride=1)
        self.conv6a = conv(128,196, kernel_size=3, stride=2)
        self.conv6aa  = conv(196,196, kernel_size=3, stride=1)
        self.conv6b  = conv(196,196, kernel_size=3, stride=1)

        # define correlation layer CostVolumeLayer or Correlation
        self.corr = CostVolumeLayer()
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.warp = WarpingLayer()

        nd = (2*md+1)**2 # 81 for max displayme1nt=4
        dd = np.cumsum([128,128,96,64,32]) # [128, 256, 352, 416, 448]

        od = nd #
        # convx_x is the conv layer for optic flow estimation
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow6 = predict_flow(od+dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd+128+4
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od+dd[4])
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd+96+4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od+dd[4])
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd+64+4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od+dd[4])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd+32+4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        # dilation convolution for contextNet to extract features of optic flow
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32) # dilation=1

        # parameters initialization for weight and bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,x):
        im1 = x[:,:1,:,:]
        im2 = x[:,1:,:,:]
        # the pyramid extractor use SCNN model
        ######
        #1
        im11a_mem = im11a_spike = torch.zeros(batch_size, 16, 512, 218, device=device)
        im11aa_mem = im11aa_spike = torch.zeros(batch_size, 16, 512, 218, device=device)
        c11_sumspike = c11_mem = c11_spike = torch.zeros(batch_size, 16, 512, 218, device=device)
        im21a_mem = im21a_spike = torch.zeros(batch_size, 16, 512, 218, device=device)
        im21aa_mem = im21aa_spike = torch.zeros(batch_size, 16, 512, 218, device=device)
        c21_sumspike = c21_mem = c21_spike = torch.zeros(batch_size, 16, 512, 218, device=device)

        #2
        im12a_mem = im12a_spike = torch.zeros(batch_size, 32, 256, 109, device=device)
        im12aa_mem = im12aa_spike = torch.zeros(batch_size, 32, 256, 109, device=device)
        c12_sumspike = c12_mem = c12_spike = torch.zeros(batch_size, 32, 256, 109, device=device)
        im22a_mem = im22a_spike = torch.zeros(batch_size, 32, 256, 109, device=device)
        im22aa_mem = im22aa_spike = torch.zeros(batch_size, 32, 256, 109, device=device)
        c22_sumspike = c22_mem = c22_spike = torch.zeros(batch_size, 32, 256, 109, device=device)

        #3
        im13a_mem = im13a_spike = torch.zeros(batch_size, 64, 128, 55, device=device)
        im13aa_mem = im13aa_spike = torch.zeros(batch_size, 64, 128, 55, device=device)
        c13_sumspike = c13_mem = c13_spike = torch.zeros(batch_size, 64, 128, 55, device=device)
        im23a_mem = im23a_spike = torch.zeros(batch_size, 64, 128, 55, device=device)
        im23aa_mem = im23aa_spike = torch.zeros(batch_size, 64, 128, 55, device=device)
        c23_sumspike = c23_mem = c23_spike = torch.zeros(batch_size, 64, 128, 55, device=device)

        #4
        im14a_mem = im14a_spike = torch.zeros(batch_size, 96, 64, 28, device=device)
        im14aa_mem = im14aa_spike = torch.zeros(batch_size, 96, 64, 28, device=device)
        c14_sumspike = c14_mem = c14_spike = torch.zeros(batch_size, 96, 64, 28, device=device)
        im24a_mem = im24a_spike = torch.zeros(batch_size, 96, 64, 28, device=device)
        im24aa_mem = im24aa_spike = torch.zeros(batch_size, 96, 64, 28, device=device)
        c24_sumspike = c24_mem = c24_spike = torch.zeros(batch_size, 96, 64, 28, device=device)

        # 5
        im15a_mem = im15a_spike = torch.zeros(batch_size, 128, 32, 14, device=device)
        im15aa_mem = im15aa_spike = torch.zeros(batch_size, 128, 32, 14, device=device)
        c15_sumspike = c15_mem = c15_spike = torch.zeros(batch_size, 128, 32, 14, device=device)
        im25a_mem = im25a_spike = torch.zeros(batch_size, 128, 32, 14, device=device)
        im25aa_mem = im25aa_spike = torch.zeros(batch_size, 128, 32, 14, device=device)
        c25_sumspike = c25_mem = c25_spike = torch.zeros(batch_size, 128, 32, 14, device=device)

        # 6
        im16a_mem = im16a_spike = torch.zeros(batch_size, 196, 16, 7, device=device)
        im16aa_mem = im16aa_spike = torch.zeros(batch_size, 196, 16, 7, device=device)
        c16_sumspike = c16_mem = c16_spike = torch.zeros(batch_size, 196, 16, 7, device=device)
        im26a_mem = im26a_spike = torch.zeros(batch_size, 196, 16, 7, device=device)
        im26aa_mem = im26aa_spike = torch.zeros(batch_size, 196, 16, 7, device=device)
        c26_sumspike = c26_mem = c26_spike = torch.zeros(batch_size, 196, 16, 7, device=device)

        for step in range(time_windows):
            im11a_mem, im11a_spike = mem_update(self.conv1a, im1, im11a_mem, im11a_spike)
            im11aa_mem, im11aa_spike = mem_update(self.conv1aa, im11a_spike, im11aa_mem, im11aa_spike)
            c11_mem, c11_spike = mem_update(self.conv1b, im11aa_spike, c11_mem, c11_spike)
            im21a_mem, im21a_spike = mem_update(self.conv1a, im2, im21a_mem, im21a_spike)
            im21aa_mem, im21aa_spike = mem_update(self.conv1aa, im21a_spike, im21aa_mem, im21aa_spike)
            c21_mem, c21_spike = mem_update(self.conv1b, im21aa_spike, c21_mem, c21_spike)
            c11_sumspike += c11_spike
            c21_sumspike += c21_spike

            im12a_mem, im12a_spike = mem_update(self.conv2a, c11_spike, im12a_mem, im12a_spike)
            im12aa_mem, im12aa_spike = mem_update(self.conv2aa, im12a_spike, im12aa_mem, im12aa_spike)
            c12_mem, c12_spike = mem_update(self.conv2b, im12aa_spike, c12_mem, c12_spike)
            im22a_mem, im22a_spike = mem_update(self.conv2a, c21_spike, im22a_mem, im22a_spike)
            im22aa_mem, im22aa_spike = mem_update(self.conv2aa, im22a_spike, im22aa_mem, im22aa_spike)
            c22_mem, c22_spike = mem_update(self.conv2b, im22aa_spike, c22_mem, c22_spike)
            c12_sumspike += c12_spike
            c22_sumspike += c22_spike

            im13a_mem, im13a_spike = mem_update(self.conv3a, c12_spike, im13a_mem, im13a_spike)
            im13aa_mem, im13aa_spike = mem_update(self.conv3aa, im13a_spike, im13aa_mem, im13aa_spike)
            c13_mem, c13_spike = mem_update(self.conv3b, im13aa_spike, c13_mem, c13_spike)
            im23a_mem, im23a_spike = mem_update(self.conv3a, c22_spike, im23a_mem, im23a_spike)
            im23aa_mem, im23aa_spike = mem_update(self.conv3aa, im23a_spike, im23aa_mem, im23aa_spike)
            c23_mem, c23_spike = mem_update(self.conv3b, im23aa_spike, c23_mem, c23_spike)
            c13_sumspike += c13_spike
            c23_sumspike += c23_spike

            im14a_mem, im14a_spike = mem_update(self.conv4a, c13_spike, im14a_mem, im14a_spike)
            im14aa_mem, im14aa_spike = mem_update(self.conv4aa, im14a_spike, im14aa_mem, im14aa_spike)
            c14_mem, c14_spike = mem_update(self.conv4b, im14aa_spike, c14_mem, c14_spike)
            im24a_mem, im24a_spike = mem_update(self.conv4a, c23_spike, im24a_mem, im24a_spike)
            im24aa_mem, im24aa_spike = mem_update(self.conv4aa, im24a_spike, im24aa_mem, im24aa_spike)
            c24_mem, c24_spike = mem_update(self.conv4b, im24aa_spike, c24_mem, c24_spike)
            c14_sumspike += c14_spike
            c24_sumspike += c24_spike

            im15a_mem, im15a_spike = mem_update(self.conv5a, c14_spike, im15a_mem, im15a_spike)
            im15aa_mem, im15aa_spike = mem_update(self.conv5aa, im15a_spike, im15aa_mem, im15aa_spike)
            c15_mem, c15_spike = mem_update(self.conv5b, im15aa_spike, c15_mem, c15_spike)
            im25a_mem, im25a_spike = mem_update(self.conv5a, c24_spike, im25a_mem, im25a_spike)
            im25aa_mem, im25aa_spike = mem_update(self.conv5aa, im25a_spike, im25aa_mem, im25aa_spike)
            c25_mem, c25_spike = mem_update(self.conv5b, im25aa_spike, c25_mem, c25_spike)
            c15_sumspike += c15_spike
            c25_sumspike += c25_spike

            im16a_mem, im16a_spike = mem_update(self.conv6a, c15_spike, im16a_mem, im16a_spike)
            im16aa_mem, im16aa_spike = mem_update(self.conv6aa, im16a_spike, im16aa_mem, im16aa_spike)
            c16_mem, c16_spike = mem_update(self.conv6b, im16aa_spike, c16_mem, c16_spike)
            im26a_mem, im26a_spike = mem_update(self.conv6a, c25_spike, im26a_mem, im26a_spike)
            im26aa_mem, im26aa_spike = mem_update(self.conv6aa, im26a_spike, im26aa_mem, im26aa_spike)
            c26_mem, c26_spike = mem_update(self.conv6b, im26aa_spike, c26_mem, c26_spike)
            c16_sumspike += c16_spike
            c26_sumspike += c26_spike
        ######

        c11 = c11_sumspike / time_windows
        c21 = c21_sumspike / time_windows
        c12 = c12_sumspike / time_windows
        c22 = c22_sumspike / time_windows
        c13 = c13_sumspike / time_windows
        c23 = c23_sumspike / time_windows
        c14 = c14_sumspike / time_windows
        c24 = c24_sumspike / time_windows
        c15 = c15_sumspike / time_windows
        c25 = c25_sumspike / time_windows
        c16 = c16_sumspike / time_windows
        c26 = c26_sumspike / time_windows

        # delete the intermediate variable for the convenience of debug
        del im11a_mem, im11a_spike, im11aa_mem, im11aa_spike, c11_mem, c11_spike, c11_sumspike,\
            im21a_mem, im21a_spike, im21aa_mem, im21aa_spike, c21_mem, c21_spike, c21_sumspike, \
            im12a_mem, im12a_spike, im12aa_mem, im12aa_spike, c12_mem, c12_spike, c12_sumspike, \
            im22a_mem, im22a_spike, im22aa_mem, im22aa_spike, c22_mem, c22_spike, c22_sumspike, \
            im13a_mem, im13a_spike, im13aa_mem, im13aa_spike, c13_mem, c13_spike, c13_sumspike, \
            im23a_mem, im23a_spike, im23aa_mem, im23aa_spike, c23_mem, c23_spike, c23_sumspike, \
            im14a_mem, im14a_spike, im14aa_mem, im14aa_spike, c14_mem, c14_spike, c14_sumspike, \
            im24a_mem, im24a_spike, im24aa_mem, im24aa_spike, c24_mem, c24_spike, c24_sumspike, \
            im15a_mem, im15a_spike, im15aa_mem, im15aa_spike, c15_mem, c15_spike, c15_sumspike, \
            im25a_mem, im25a_spike, im25aa_mem, im25aa_spike, c25_mem, c25_spike, c25_sumspike, \
            im16a_mem, im16a_spike, im16aa_mem, im16aa_spike, c16_mem, c16_spike, c16_sumspike, \
            im26a_mem, im26a_spike, im26aa_mem, im26aa_spike, c26_mem, c26_spike, c26_sumspike

        # use densenet structure as the optical flow estimator
        # the way of every estimator layers is ：Warp + CostVolume + DesNet
        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)
        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)


        warp5 = self.warp(c25, up_flow6*0.625)
        corr5 = self.corr(c15, warp5)
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)


        warp4 = self.warp(c24, up_flow5*1.25)
        corr4 = self.corr(c14, warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        ZeroLeftPadding = nn.ZeroPad2d(padding=(1, 0, 0, 0))    # to fit the input image size
        c23 = ZeroLeftPadding(c23)
        c13 = ZeroLeftPadding(c13)
        warp3 = self.warp(c23, up_flow4*2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        ZeroLeftPadding = nn.ZeroPad2d(padding=(3, 0, 0, 0))  # use zeropadding to fit the input image size
        c22 = ZeroLeftPadding(c22)
        c12 = ZeroLeftPadding(c12)
        warp2 = self.warp(c22, up_flow3*5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)

        # context network
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.training:
            return [flow2, flow3, flow4, flow5, flow6]
        else:
            return flow2 # real optical flow

if __name__ == '__main__':
    model = PWCSNet(md=4).to(device)
    # use torchsummary package tool to view the structure of model
    # to run this file, you should set batch_size to 2
    summary(model, (2, 1024, 436))