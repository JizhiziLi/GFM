import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from config import *
from util import *


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv_up_psp(in_channels, out_channels, up_sample):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=up_sample, mode='bilinear',align_corners = False))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear',align_corners = True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class e2e_resnet34_2b_gfm_tt(nn.Module):

    def __init__(self, args):
        super().__init__()
        
        self.resnet = models.resnet34(pretrained=True)

        ##########################
        ### Encoder part - RESNET34
        ##########################
        #stage 0
        self.encoder0 = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #stage 1
        self.encoder1 = self.resnet.layer1
        #stage 2
        self.encoder2 = self.resnet.layer2
        #stage 3
        self.encoder3 = self.resnet.layer3
        #stage 4
        self.encoder4 = self.resnet.layer4
        #stage 5
        self.encoder5 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            BasicBlock(512,512),
            BasicBlock(512,512),
            BasicBlock(512,512))
        #stage 6
        self.encoder6 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            BasicBlock(512,512),
            BasicBlock(512,512),
            BasicBlock(512,512))

        ##########################
        ### Decoder part - GLANCE
        ##########################
        ###psp: N, 512, 1/32, 1/32
        self.psp_module = PSPModule(512, 512, (1, 3, 5))
        self.psp6 = conv_up_psp(512, 512, 2)
        self.psp5 = conv_up_psp(512, 512, 4)
        self.psp4 = conv_up_psp(512, 256, 8)
        self.psp3 = conv_up_psp(512, 128, 16)
        self.psp2 = conv_up_psp(512, 64, 32)
        self.psp1 = conv_up_psp(512, 64, 32)
       #stage 6g
        self.decoder6_g = nn.Sequential(
            nn.Conv2d(1024,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners = False))  

        #stage 5g
        self.decoder5_g = nn.Sequential(
            nn.Conv2d(1024,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners = False))       
        #stage 4g
        self.decoder4_g = nn.Sequential(
            nn.Conv2d(1024,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners = False) )
        #stage 3g
        self.decoder3_g = nn.Sequential(
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners = False))       
        #stage 2g
        self.decoder2_g = nn.Sequential(
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners = False))
        #stage 1g
        self.decoder1_g = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #stage 0g
        self.decoder0_g = nn.Sequential(
        nn.Conv2d(64,3,3,padding=1))
        
        ##########################
        ### Decoder part - FOCUS
        ##########################
        self.bridge_block = nn.Sequential(
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        #stage 6f
        self.decoder6_f = nn.Sequential(
            nn.Conv2d(1024,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners = False))
        #stage 5f
        self.decoder5_f = nn.Sequential(
            nn.Conv2d(1024,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners = False))
        #stage 4f
        self.decoder4_f = nn.Sequential(
            nn.Conv2d(1024,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners = False) )
        #stage 3f
        self.decoder3_f = nn.Sequential(
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners = False) )       
        #stage 2f
        self.decoder2_f = nn.Sequential(
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners = False))
        #stage 1f
        self.decoder1_f = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))        
        #stage 0f
        self.decoder0_f = nn.Sequential(
        nn.Conv2d(64,1,3,padding=1))

        
    def forward(self, input):

        glance_sigmoid = torch.zeros(input.shape)
        focus_sigmoid =  torch.zeros(input.shape)
        fusion_sigmoid =  torch.zeros(input.shape)

        ##########################
        ### Encoder part - RESNET34_2b
        ##########################
        e0 = self.encoder0(input)
        #e0: N, 64, H, W
        e1 = self.encoder1(e0)
        #e1: N, 64, H, W
        e2 = self.encoder2(e1)
        #e2: N, 128, H/2, W/2
        e3 = self.encoder3(e2)
        #e3: N, 256, H/4, W/4
        e4 = self.encoder4(e3)
        #e4: N, 512, H/8, W/8
        e5 = self.encoder5(e4)
        #e5: N, 512, H/16, W/16
        e6 = self.encoder6(e5)
        #e6: N, 512, H/32, W/32

        ##########################
        ### Decoder part - GLANCE
        ##########################
        psp = self.psp_module(e6) 
        #psp: N, 512, H/32, W/32
        d6_g = self.decoder6_g(torch.cat((psp, e6),1))
        #d6_g: N, 512, H/16, W/16
        d5_g = self.decoder5_g(torch.cat((self.psp6(psp),d6_g),1))
        #d5_g: N, 512, H/8, W/8
        d4_g = self.decoder4_g(torch.cat((self.psp5(psp),d5_g),1))
        #d4_g: N, 256, H/4, W/4
        d3_g = self.decoder3_g(torch.cat((self.psp4(psp),d4_g),1))
        #d4_g: N, 128, H/2, W/2
        d2_g = self.decoder2_g(torch.cat((self.psp3(psp),d3_g),1))
        #d2_g: N, 64, H, W
        d1_g = self.decoder1_g(torch.cat((self.psp2(psp),d2_g),1))
        #d1_g: N, 64, H, W
        d0_g = self.decoder0_g(d1_g)
        #d0_g: N, 3, H, W
        glance_sigmoid = torch.sigmoid(d0_g)

        ##########################
        ### Decoder part - FOCUS
        ##########################
        bb = self.bridge_block(e6)
        #bg: N, 512, H/32, W/32
        d6_f = self.decoder6_f(torch.cat((bb, e6),1))
        #d6_f: N, 512, H/16, W/16
        d5_f = self.decoder5_f(torch.cat((d6_f, e5),1))
        #d5_f: N, 512, H/8, W/8
        d4_f = self.decoder4_f(torch.cat((d5_f, e4),1))
        #d4_f: N, 256, H/4, W/4
        d3_f = self.decoder3_f(torch.cat((d4_f, e3),1))    
        #d3_f: N, 128, H/2, W/2
        d2_f = self.decoder2_f(torch.cat((d3_f, e2),1))
        #d2_f: N, 64, H, W
        d1_f = self.decoder1_f(torch.cat((d2_f, e1),1))
        #d1_f: N, 64, H, W
        d0_f = self.decoder0_f(d1_f)
        #d0_f: N, 1, H, W
        focus_sigmoid = torch.sigmoid(d0_f)

        ##########################
        ### Fusion net - G/F
        ##########################
        fusion_sigmoid = get_masked_local_from_global(glance_sigmoid, focus_sigmoid)       
        
        return glance_sigmoid, focus_sigmoid, fusion_sigmoid
        