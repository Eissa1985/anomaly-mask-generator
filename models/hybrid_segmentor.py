import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt
from torchvision.models import resnet50, ResNet50_Weights
# --- Utility Layers ---
class LayerNorm2d(nn.LayerNorm):

    def forward(self, x):

        x = rearrange(x, "b c h w -> b h w c")

        x = super().forward(x)

        x = rearrange(x, "b h w c -> b c h w")

        return x

class DepthWiseConv(nn.Module):

    def __init__(self, in_dim, out_dim, kernel, padding, stride=1, bias=True):

        super(DepthWiseConv, self).__init__()

        self.DW_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,

                                 kernel_size=kernel, stride=stride,

                                 padding=padding, groups=in_dim, bias=bias)

        self.PW_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,

                                 kernel_size=1, bias=bias)

    def forward(self, x):

        return self.PW_conv(self.DW_conv(x))

class DoubleConv(nn.Module):

    def __init__(self, in_dim, out_dim):

        super(DoubleConv, self).__init__()

        # Ensure hidden_dim is at least 1

        hidden_dim = max(1, int((in_dim + out_dim)/2))

        self.conv_block = nn.Sequential(

            nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(hidden_dim),

            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(out_dim),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.conv_block(x)

# --- Transformer Components ---
class OverlapPatchEmbedding(nn.Module):

    def __init__(self, kernel, stride, padding, in_dim, out_dim):

        super(OverlapPatchEmbedding, self).__init__()

        # Use Conv2d directly for better compatibility with dynamic sizes

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding)



    def forward(self, x):

        return self.conv(x)

class EfficientMSA(nn.Module):

    def __init__(self, dim, n_heads, reduction_ratio):

        super(EfficientMSA, self).__init__()

        self.reshaping_k = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)

        self.reshaping_v = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)

        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

        self.norm = LayerNorm2d(dim)



    def forward(self, x):

        n, c, h, w = x.shape

        x_norm = self.norm(x)

        reshaped_k = rearrange(self.reshaping_k(x_norm), "b c h w -> b (h w) c")

        reshaped_v = rearrange(self.reshaping_v(x_norm), "b c h w -> b (h w) c")

        q = rearrange(x_norm, "b c h w -> b (h w) c")

        output, _ = self.attention(q, reshaped_k, reshaped_v)

        return rearrange(output, "b (h w) c -> b c h w", h=h, w=w)

class MixFFN(nn.Module):

    def __init__(self, dim, expansion_factor):

        super(MixFFN, self).__init__()

        latent_dim = dim * expansion_factor

        self.norm = LayerNorm2d(dim)

        self.mixffn = nn.Sequential(

            nn.Conv2d(dim, latent_dim, 1),

            DepthWiseConv(latent_dim, latent_dim, kernel=3, padding=1),

            nn.GELU(),

            nn.Conv2d(latent_dim, dim, 1)

        )

    def forward(self, x):

        return self.mixffn(self.norm(x))

class MiT(nn.Module):

    def __init__(self, channels, dims, n_heads, expansion, reduction_ratio, n_layers):

        super(MiT, self).__init__()

        # Optimized overlapping parameters for 448/512 compatibility

        kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)

        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])



        for (in_dim, out_dim), (k, s, p), layers_count, exp, heads, red in zip(dim_pairs, kernel_stride_pad, n_layers, expansion, n_heads, reduction_ratio):

            overlapping = OverlapPatchEmbedding(k, s, p, in_dim, out_dim)

            blocks = nn.ModuleList([])

            for _ in range(layers_count):

                blocks.append(nn.ModuleList([

                    EfficientMSA(dim=out_dim, n_heads=heads, reduction_ratio=red),

                    MixFFN(dim=out_dim, expansion_factor=exp)

                ]))

            self.stages.append(nn.ModuleList([overlapping, blocks]))



    def forward(self, x):

        layer_outputs = []

        for overlapping, blocks in self.stages:

            x = overlapping(x)

            for (attention, ffn) in blocks:

                x = attention(x) + x

                x = ffn(x) + x

            layer_outputs.append(x)

        return layer_outputs

# --- ResNet Component ---
class ResNetEncoder(nn.Module):

    def __init__(self):

        super(ResNetEncoder, self).__init__()

        encoder = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.encoder1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu) # 1/2

        self.mp = encoder.maxpool # 1/4

        self.encoder2 = encoder.layer1 # 1/4

        self.encoder3 = encoder.layer2 # 1/8

        self.encoder4 = encoder.layer3 # 1/16

        self.encoder5 = encoder.layer4 # 1/32



    def forward(self, x):

        output1 = self.encoder1(x)

        output2 = self.encoder2(self.mp(output1))

        output3 = self.encoder3(output2)

        output4 = self.encoder4(output3)

        output5 = self.encoder5(output4)

        return output1, output2, output3, output4, output5

# --- Main Hybrid Model ---
class HybridSegmentor(nn.Module):

    def __init__(self, in_channels=3, num_classes=1):

        super(HybridSegmentor, self).__init__()

        # MiT features: 64, 128, 320, 512

        # ResNet50 features: 64, 256, 512, 1024, 2048

        mit_dims = (64, 128, 320, 512)

        resnet_dims = (64, 256, 512, 1024, 2048)

       

        self.mix_transformer = MiT(in_channels, mit_dims, n_heads=(1, 2, 5, 8),

                                   expansion=(8, 8, 4, 4), reduction_ratio=(8, 4, 2, 1),

                                   n_layers=(2, 2, 2, 2))

       

        self.cnn_encoder = ResNetEncoder()

       

        # FIXED: Input channels match the cat() operation in forward()

        self.reduce_channels = nn.ModuleList([

            DoubleConv(mit_dims[0] + resnet_dims[1], mit_dims[0]),   # 64 + 256

            DoubleConv(mit_dims[1] + resnet_dims[2], mit_dims[1]),   # 128 + 512

            DoubleConv(mit_dims[2] + resnet_dims[3], mit_dims[2]),   # 320 + 1024

            DoubleConv(mit_dims[3] + resnet_dims[4], mit_dims[3]),   # 512 + 2048

            DoubleConv(resnet_dims[4], mit_dims[3])                  # 2048 (Single input bridge)

        ])



        self.upsampling = nn.ModuleList([

            nn.Sequential(DoubleConv(mit_dims[0], 32), nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)),

            nn.Sequential(DoubleConv(mit_dims[1], 32), nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)),

            nn.Sequential(DoubleConv(mit_dims[2], 32), nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)),

            nn.Sequential(DoubleConv(mit_dims[3], 32), nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)),

            nn.Sequential(DoubleConv(mit_dims[3], 32), nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True))

        ])



        self.to_segment_conv = nn.Conv2d(5 * 32, num_classes, 1)



    def forward(self, x):

        mit_feats = self.mix_transformer(x) # [1/4, 1/8, 1/16, 1/32]

        cnn_feats = self.cnn_encoder(x)     # [1/2, 1/4, 1/8, 1/16, 1/32]



        up_sides = []

       

        # Loop over the 4 shared scales (1/4 to 1/32)

        # We pair mit_feats[i] with cnn_feats[i+1] to ensure spatial sizes match

        for i in range(4):

            fused = torch.cat((mit_feats[i], cnn_feats[i+1]), dim=1)

            up_sides.append(self.upsampling[i](self.reduce_channels[i](fused)))

       

        # 5th level using only the deepest ResNet bottleneck (1/32 scale)

        # No cat() here to avoid the 4096 channel error

        f5 = self.reduce_channels[4](cnn_feats[4])

        up_sides.append(self.upsampling[4](f5))



        return self.to_segment_conv(torch.cat(up_sides, dim=1))


