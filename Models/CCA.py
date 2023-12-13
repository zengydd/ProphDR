import sys, os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils.pyt_utils import load_model
from torch.nn import Softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def INF(B,H,W):
    # GPU运行
     return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1).to(device) 
    #  return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_channels):
        super(CrissCrossAttention,self).__init__()
        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
    # v 的通道数不同
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        # [b, c', h, w]
        # print('x', x.shape)
        proj_query = self.query_conv(x)
        # proj_query = x
        # print('proj_query', proj_query.shape)
         # [b, w, c', h] -> [b*w, c', h] -> [b*w, h, c']
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        # print('proj_query_H', proj_query_H.shape)
        # [b, h, c', w] -> [b*h, c', w] -> [b*h, w, c']
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        # print('proj_query_w', proj_query_W.shape)
     
        proj_key = self.key_conv(x)
        # proj_key = x
        # print('projkey', proj_key.shape)
            # [b, w, c', h] -> [b*w, c', h]
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)      
        # print('proj_key_H', proj_key_H.shape)
        # [b, h, c', w] -> [b*h, c', w]
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        # print('proj_key_W', proj_key_W.shape)
        
        proj_value = self.value_conv(x)
        # proj_value = x
        # print('proj_value', proj_value.shape)
        # [b, w, c, h] -> [b*w, c, h]
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        # print('proj_value_H', proj_value_H.shape)
         # [b, h, c, w] -> [b*h, c, w]
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        # print('proj_value_W', proj_value_W.shape)
        # bmm : attention
         # [b*w, h, c']* [b*w, c', h] -> [b*w, h, h] -> [b, h, w, h]
        #  加上自身那个点的信息
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        # print('energy_H \n', energy_H)
        # [b*h, w, c']*[b*h, c', w] -> [b*h, w, w] -> [b, h, w, w]
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        # print('energy_W \n', energy_W)
        # [b,h,w,h+w],对角线上-inf数值置零
        cat = torch.cat([energy_H, energy_W], 3)
        concate = self.softmax(cat)
        # print('concate \n', cat.shape)
# ATTENTION
        # [b, h, w, h] -> [b, w, h, h] -> [b*w, h, h]
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_H2 = concate[:,:,:,0:height].permute(0,2,1,3).contiguous()
        # print('att_H2', att_H2)
        # [b,h,w,w] -> [b*h,w,w]
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        att_W2 = concate[:,:,:,height:height+width].contiguous()
        # print('att_W2', att_W2)
        # print('att_W', att_W.shape)
        # [b*w, h, c]*[b*w, h, h] -> [b, w, c, h]
        # 把attention weight 加载到位点上
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        # print('out', out_H.size(),out_W.size())
        output = self.gamma*(out_H + out_W) + x
        # print('output', output)
        return output, att_H, att_W
 



if __name__ == '__main__':
    model = CrissCrossAttention(1)
    x = torch.randn(1, 1, 5, 3)
    output, att_H, att_W = model(x)
    print('s',output.shape)
    # print(model)
    # print('att_H \n', att_H)
    # print('att_W \n', att_W)
    # torch.cuda.empty_cache()清空gpu缓存