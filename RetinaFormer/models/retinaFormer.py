import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_

from torchvision.ops.deform_conv import DeformConv2d



#mlp layer
class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)

#layer norm:from dehazeformer
class LN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(LN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std
        """ 去掉了均值 方差的反传"""
        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias
    
"""attention related code"""
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )  # embedding

        self.softmax = nn.Softmax(dim=-1)
        self.w = nn.Parameter(torch.ones(2)) 
        self.relu = nn.ReLU()

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 5, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        qon, kon, qoff, koff, von = qkv[0], qkv[1], qkv[2], qkv[3], qkv[
            4]  # make torchscript happy (cannot use tensor as tuple)

        qon = qon * self.scale
        attnon = (qon @ kon.transpose(-2, -1))

        qoff = qoff * self.scale
        attnoff = (qoff @ koff.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        B = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attnon = attnon + B.unsqueeze(0)
        attnoff = attnoff + B.unsqueeze(0)

        ONA = self.softmax(attnon)
        OFFA = self.softmax(attnoff)

        won = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        woff = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))

        attn = ONA * won + OFFA * woff

        outA = (attn @ von).transpose(1, 2).reshape(B_, N, self.dim)
        return outA


class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size #多了
        self.network_depth = network_depth

        self.conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
                                  )

        self.V = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        self.QK = nn.Conv2d(dim, dim * 2, 1)
        self.QKoff = nn.Conv2d(dim, dim * 2, 1)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # 多了shift
        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape
        Xoff = self.maxpool(X) - X
        
        # Q K V
        Von = self.V(X)
        QKon = self.QK(X)
        QKoff = self.QKoff(Xoff)
        QKVon = torch.cat([QKon, QKoff, Von], dim=1)
        
        # shift
        shifted_QKV = self.check_size(QKVon, self.shift_size > 0) 
        Ht, Wt = shifted_QKV.shape[2:]

        # partition windows
        shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
        qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

        attn_windows = self.attn(qkv)

        # merge windows
        shifted_outA = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

        # reverse cyclic shift
        outA = shifted_outA[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]#多了shift-size
        outA= outA.permute(0, 3, 1, 2)

        conv_out = self.conv(Von)
        out = self.proj(conv_out + outA)

        return out


class ONOFFTransformer(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 window_size=8, shift_size=0):
        super().__init__()

        self.norm1 = LN(dim)  
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size) #多了shift——size
        self.relu = nn.ReLU()

        self.norm2 = LN(dim) 
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))  # mlp_ratio用于全连接层

    def forward(self, x):
        identity = x
        x, rescale, rebias = self.norm1(x)
        x = self.attn(x)
        x = x * rescale + rebias
        x = identity + x

        identity = x
        x = self.mlp(x)
        x, rescale, rebias = self.norm2(x)
        x = identity + x * rescale + rebias
        return x


class ONOFFTransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4., window_size=8, trans_num=2):
        super().__init__()
        # build blocks
        """tansformer的个数"""
        TransfBlock = []
        for i in range(trans_num):
            TransfBlock += [ONOFFTransformer(network_depth=network_depth,
                                             dim=dim,
                                             num_heads=num_heads,
                                             mlp_ratio=mlp_ratio,
                                             window_size=window_size,
                                             shift_size=0 if (i % 2 == 0) else window_size // 2)]
        self.transformer = nn.Sequential(*TransfBlock)

    def forward(self, x):
        x = self.transformer(x)
        return x


class PhotoReceptor(nn.Module):
    def __init__(self, patch_size=8, in_chans=4, embed_dim=24, kernel_size=3):
        super().__init__()
        self.patch_size = patch_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
        )

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        return x

    def forward(self, x):
        batch_size = x.shape[0]
        if batch_size == 1:  # 因为后续下采样到8倍，这块先填充
            x = self.check_image_size(x)  # B,C,L,X
        """这块新增视杆信息"""
        L = torch.mean(x, dim=1).unsqueeze(1)
        Photo = torch.cat((x, L), dim=1)
        Photo = self.conv1(Photo)  # 用2个卷积对数据进行初步处理
        return x, Photo




class HoriztontalBlock(nn.Module):
    def __init__(self, embed_dim=24):
        super().__init__()
        # build blocks
        self.norm1 = LN(embed_dim)
        self.block1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim, embed_dim, 1, padding_mode='reflect'),
            nn.LeakyReLU()
        )

        self.norm2 = LN(embed_dim)
        self.block2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim, embed_dim, 1, padding_mode='reflect'),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x1, rescale, rebias = self.norm1(x)
        x1 = self.block1(x1)
        x1 = x1 * rescale + rebias

        x2, rescale, rebias = self.norm2(x1)
        x2 = self.block2(x2)
        x2 = x + x2*rescale+rebias
        return x2


class Bipolar(nn.Module):
    def __init__(self, in_chans, embed_dim, kernel_size=3,
                 network_depth=8, dim=128, num_heads=4, mlp_ratio=2.,
                 window_size=8, trans_num=2):
        super().__init__()
        # 3*3 conv+1*1conv
        self.calculate = nn.Sequential(nn.Conv2d(in_chans, in_chans, kernel_size=1),
                                       nn.ReLU())
        # # down-sample
        self.down1 = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=2, padding=1,
                               padding_mode='reflect')
        # self.patchEmd = Patch_Embed_stage(embed_dim, embed_dim)
        self.transformer = ONOFFTransformerBlock(network_depth=network_depth, dim=embed_dim,
                                      num_heads=num_heads, mlp_ratio=mlp_ratio,
                                      window_size=window_size, trans_num=trans_num)


    def forward(self, x):
        x1_res = self.calculate(x)
        x1 = self.down1(x1_res)
        x2 = self.transformer(x1) + x1
        return x2


class Ganglion(nn.Module):
    def __init__(self, embed_dims=[16, 32, 64, 128], out_chans=3, x_size=[256, 256], network_depth=10, num_heads=6,
                 mlp_ratio=4., window_size=8, trans_num=2):
        super().__init__()
        self.down = nn.Conv2d(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.trans = ONOFFTransformerBlock(network_depth=network_depth, dim=embed_dims[3],
                                num_heads=num_heads, mlp_ratio=mlp_ratio,
                                window_size=window_size, trans_num=trans_num)
        self.skip3 = nn.Conv2d(embed_dims[2], embed_dims[2], 3, padding_mode='reflect', padding=1)
        self.skip4 = nn.Conv2d(embed_dims[2], embed_dims[2], 3, padding_mode='reflect', padding=1)
        self.ganglion2 = RetinaFusion(embed_dims[2])
        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[2], embed_dim=embed_dims[3])  # 上采样

        self.skip5 = nn.Conv2d(embed_dims[1], embed_dims[1], 3, padding_mode='reflect', padding=1)
        self.skip6 = nn.Conv2d(embed_dims[1], embed_dims[1], 3, padding_mode='reflect', padding=1)
        self.ganglion3 = RetinaFusion(embed_dims[1])
        self.patch_split3 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[1], embed_dim=embed_dims[2])  # 上采样
        self.patch_split4 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[0], embed_dim=embed_dims[1])  # 上采样
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[0], kernel_size=3)  # ON-OFF对应的数据
        self.group = 4
        self.conv_on = nn.Sequential(nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, padding=1, groups=embed_dims[0], padding_mode='reflect'),nn.ReLU())
        self.conv_off = nn.Sequential(nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=5, padding=2, groups=embed_dims[0], padding_mode='reflect'),nn.ReLU())

        self.color_scale = nn.Parameter(0.1*torch.randn(int(embed_dims[0]/self.group),3))
        self.softmax = nn.Softmax(dim=1)
        
    """the proposed COM"""
    def color_opponency_Mechanism(self, GC):
        b,c,h,w = GC.size()
        Xcen = self.conv_on(GC)
        Xsur = self.conv_off(GC)
        XCOM = torch.zeros(Xcen.shape).cuda()

        # 黄颜色通道
        for i in range(self.group):
            #红颜色通道
            XCOM[:,i*self.group,:,:] = Xcen[:,i*self.group,:,:] - self.color_scale[i,0] * Xsur[:,i*self.group+1,:,:] #R-G
            XCOM[:,i*self.group+1,:,:] = Xcen[:,i*self.group+1,:,:] - self.color_scale[i,1] * Xsur[:,i*self.group,:,:]#G-R
            XCOM[:,i*self.group+2,:,:] = Xcen[:,i*self.group+2,:,:]-self.color_scale[i,2] * (Xsur[:,i*self.group+1,:,:]+Xsur[:,i*self.group,:,:])*0.5

        XCOM = XCOM + GC
        XCOM = XCOM.reshape(b,-1,self.group,h,w) #求均值 C*4
        XCOM = torch.mean(XCOM,dim=1)

        return XCOM

    def forward(self, amaout, biotransresult):
        down_result = self.down(amaout)  # B 8C H W
        trans_result = self.trans(down_result) + down_result  # B 8C H W
        up1 = self.patch_split2(trans_result)  # 上采样 B,4C,H/2，W/2
        fusion2 = self.ganglion2(self.skip3(up1),self.skip4(amaout))

        up2 = self.patch_split3(fusion2)  # 上采样 B,C,H，W
        fusion3 = self.ganglion3(self.skip5(up2),self.skip6(biotransresult))

        final_out = self.patch_split4(fusion3)

        # 加入颜色机制和同心圆感受野机制
        XCOM = self.color_opponency_Mechanism(final_out)
        # #加入脉冲发放机制
        return XCOM


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x

"""The proposed RetinaFusion"""
class RetinaFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=4):
        super(RetinaFusion, self).__init__()

        d = int(dim / reduction)
        self.d = d

        self.conv = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.convdw1 = nn.Conv2d(d, d, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.convdw2 = nn.Conv2d(d, d, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
        self.convdw3 = nn.Conv2d(d, d, kernel_size=3, stride=1, padding=7, dilation=7, bias=False)
        self.convcat = nn.Conv2d(dim, dim, 1, stride=1, bias=False)
        self.pixelAttn2 = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=False),
            nn.ReLU())

    def forward(self, x1,x2):
        XR = x1+x2
        XR1,XR2,XR3,XR4 = torch.split(XR, (self.d,self.d,self.d,self.d), dim=1)
        XR1 = self.conv(XR1)  # B,d,H,W
        XR2 = self.convdw1(XR2)  # B,d,H,W-+
        XR3 = self.convdw2(XR3)  # B,d,H,W
        XR4 = self.convdw3(XR4)  # B,d,H,W

        XRO = self.convcat(torch.cat([XR1, XR2, XR3, XR4], dim=1))  # B,C,H,W to B,C,H,W
        XRFM = self.pixelAttn2(XRO) * XR + XR  # B,1,H,W to B,C,H,W
        return XRFM



class RetinaFormer(nn.Module):
    def __init__(self, in_chans=4, out_chans=4, window_size=8,
                 embed_dims=[24, 48, 96, 192],
                 mlp_ratios=[2., 4.],
                 depths=[1, 3, 4, 8, 12],
                 num_heads=[4, 4, 4]):
        super(RetinaFormer, self).__init__()

        self.down_sample_size = 8
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        self.photo = PhotoReceptor(patch_size=self.down_sample_size, in_chans=in_chans, embed_dim=embed_dims[0],
                                   kernel_size=3)

        # # backbone
        # self.horizontal = HoriztontalBlock(embed_dim=embed_dims[0],
        #                                    network_depth=sum(depths), num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
        #                                    window_size=window_size, trans_num=depths[0])
        # backbone
        self.horizontal = HoriztontalBlock(embed_dim=embed_dims[0])

        # split image into non-overlapping patches
        self.bipolar = Bipolar(
            in_chans=embed_dims[0] * 2, embed_dim=embed_dims[1], kernel_size=3,
            network_depth=sum(depths), dim=embed_dims[1],
            num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
            window_size=window_size, trans_num=depths[1])

        self.down2 = nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1, padding_mode='reflect')

        # self.amacrine_patchEmd = Patch_Embed_stage(embed_dims[1], embed_dims[1])
        self.amacrine = ONOFFTransformerBlock(network_depth=sum(depths), dim=embed_dims[2],
                                   num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                   window_size=window_size, trans_num=depths[2])

        self.ganglion = Ganglion(embed_dims, out_chans, network_depth=sum(depths),
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 window_size=window_size, trans_num=depths[3])


    def forward(self, x):
        H, W = x.shape[2:]
        # 视网膜层  photo有L,维度B,C,H,W
        I, photo = self.photo(x)
        # 水平细胞层 卷积
        horizontal = self.horizontal(photo)  # 维度B,C,H W
        #双极细胞层
        bioInput = torch.cat((photo, horizontal), dim=1)  # B，C，H,W
        biotransresult = self.bipolar(bioInput)  # B，C，H，W
        #无长突细胞层
        amaout_down = self.down2(biotransresult) # B，2C，H，W
        amaout = self.amacrine(amaout_down) + amaout_down  # B，2C，H，W
        #神经节细胞层
        ganglion = self.ganglion(amaout, biotransresult)
        #将大气散射模型 直接改为端到端输出
        Icolor, L = torch.split(ganglion, (3, 1), dim=1)
        Iout = L * Icolor +I
        Iout = Iout[:, :, :H, :W]
        return Iout


def retinaformer_l():
    return RetinaFormer(
        embed_dims=[32,64,128,256],
        mlp_ratios=[0., 4., 4.,4.0],
        depths=[0,4, 8, 12],
        num_heads=[0,2, 4, 8])

def retinaformer_m():
    return RetinaFormer(
        embed_dims=[32,64,128,256],
        mlp_ratios=[0., 4., 4.,4.0],
        depths=[0,3, 6, 9],
        num_heads=[0,2, 4, 8])

def retinaformer_s():
    return RetinaFormer(
        embed_dims=[24,48,96,192],
        mlp_ratios=[0., 4., 4.,4.],
        depths=[0,2, 4, 8],
        num_heads=[0,2, 4, 8])


