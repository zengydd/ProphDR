import torch 
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from Models.CCA import CrissCrossAttention

class RCCAModule(nn.Module):
    # channels from C to C'   '2048->512' --> '1->512'
    # 我感觉outchannels随便改改也没什么影响
    # RCCAModule(2048, 512, num_classes)
    def __init__(self, dim, in_channels, out_channels, recurrence):
        super(RCCAModule, self).__init__()
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.recurrence = recurrence
        self.h_dim = dim
        # inter_channels = in_channels // 4
        inter_channels = in_channels
        # self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                            InPlaceABNSync(inter_channels))
        
        # same_padding卷积卷积再进行crisscross，我可以去掉这一层 【看效果吧不行就换掉】
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        # self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            InPlaceABNSync(inter_channels))
        #same_padding卷积
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))
# bottleneck是用来降低计算量的下采样操作,看结果好坏进行取舍实验
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.2),
            # num_classes 是con2vd 最终层的输出维度
            # same_valid卷积, 注意bias=Trus
            nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
            )
# 三组数据压到一组上
        # valid卷积 W:3—>1
        # self.redim = nn.Sequential(nn.Conv2d(out_channels, num_classes, 3, padding=(1,0), bias=False),
        #                            nn.BatchNorm2d(num_classes))

    def forward(self, x):
        m_batchsize, channel, height, width = x.size()
        # reduce channels from C to C'   2048->512
        output = self.conva(x)
        # 这里开始进行循环cca操作
        attn_list = []
        for i in range(self.recurrence):
            output, att_H, att_W = self.cca(output)
            attn_list.append([att_H, att_W])
            # print(i, attn_list)

        output = self.convb(output)
# 不是很懂为什么cca传出来的结果要进行bottlenet和concat
        output = self.bottleneck(torch.cat([x, output], 1))
        # print('out11111', output.shape)
# 三组数据压到一组上
        # output = self.redim(output)
        if self.h_dim == 3:
            output = output.squeeze(-1)
        # b,c,h,w -> b,h,c,w 改bottleneck outchannels = 1
        output = output.permute(0,2,1,3).contiguous().view(m_batchsize, height, width*channel)
        # print('outpppppp', output.shape)
        # output = self.redim(output)
        return output, attn_list

class CCNet(nn.Module):
    def __init__(self, dim, recurrence=2):
        super(CCNet, self).__init__()
        self.ccnet = RCCAModule(dim, in_channels=1, out_channels=512, recurrence=recurrence)
    def forward(self, x):
        output, attn_list = self.ccnet(x)
        return output, attn_list


if __name__ == "__main__":
    x = torch.randn(2, 1, 714, 3)
    d = torch.randn(2,1,768,1)
    model = CCNet(num_classes = 1, recurrence=2)
    out = model(x)
    print(model)
    print('out', out.shape)
