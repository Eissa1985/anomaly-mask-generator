import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. وحدة CBAM (صيادة الشذوذ الدقيق - بديل CoordAtt)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM_Mask(nn.Module):
    """
    تُستخرج قناع انتباه (Attention Mask) حاد جداً
    يحافظ على إشارات الشذوذ الدقيقة بفضل الـ Max Pooling.
    """
    def __init__(self, inp, reduction=16):
        super(CBAM_Mask, self).__init__()
        self.ca = ChannelAttention(inp, reduction)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        c_mask = self.ca(x)
        s_mask = self.sa(x * c_mask)
        return c_mask * s_mask

# ==========================================
# 2. وحدات الـ ASPP (Atrous Spatial Pyramid Pooling)
# ==========================================
class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.relu(self.bn(self.conv(x)))
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU())
        # استخدام تباعدات (2, 4, 6) لتناسب حجم الخرائط الصغيرة في قاع الشبكة
        self.branch2 = ASPPConv(in_channels, out_channels, dilation=2)
        self.branch3 = ASPPConv(in_channels, out_channels, dilation=4)
        self.branch4 = ASPPConv(in_channels, out_channels, dilation=6)
        self.branch5 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        return self.project(torch.cat([x1, x2, x3, x4, x5], dim=1))

# ==========================================
# 3. وحدات المفكك الأساسية (UpConv & DBBlock)
# ==========================================
class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=nn.BatchNorm2d):
        super(UpConvBlock, self).__init__()
        # استخدام Bilinear Interpolation لتجنب رقع الشطرنج
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn = norm_layer(out_channel)
        self.relu = nn.GELU()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(x)))

class DBBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=nn.BatchNorm2d):
        super(DBBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.depthwise_norm = norm_layer(in_channel)
        self.depthwise_activation = nn.LeakyReLU(0.01)
        self.pointwise_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.norm1 = norm_layer(out_channel)
        self.activation1 = nn.LeakyReLU(0.01)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.norm2 = norm_layer(out_channel)
        self.activation2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_activation(self.depthwise_norm(self.depthwise_conv(x)))
        x = self.activation1(self.norm1(self.pointwise_conv(x)))
        x = self.activation2(self.norm2(self.conv2(x)))
        return x

# ==========================================
# 4. المفكك الكامل (Decoder)
# ==========================================
class Decoder(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.num_layers = len(in_channels)
        
        # وحدة ASPP في عنق الزجاجة (Bottleneck)
        bottleneck_ch = in_channels[-1]
        self.aspp = ASPP(in_channels=bottleneck_ch, out_channels=bottleneck_ch)
        
        self.up_convs = nn.ModuleList()
        self.db_blocks = nn.ModuleList()
        self.cbam_layers = nn.ModuleList()

        # بناء مسارات الرفع والدمج
        for i in range(self.num_layers - 1, 0, -1):
            current_ch = in_channels[i]
            target_ch = in_channels[i-1] 
            
            up = UpConvBlock(current_ch, current_ch // 2, norm_layer)
            self.up_convs.append(up)
            
            db = DBBlock(current_ch // 2 + target_ch, target_ch, norm_layer)
            self.db_blocks.append(db)
            
            # بوابات الانتباه CBAM لمسارات التخطي (Attention U-Net)
            self.cbam_layers.append(CBAM_Mask(target_ch))

        final_ch = in_channels[0]

        if self.num_layers >= 5:
            self.extra_up = UpConvBlock(final_ch, 48, norm_layer)
            self.extra_db = DBBlock(48, 24, norm_layer)
            head_in = 24
        else:
            self.extra_up = None
            self.extra_db = None
            head_in = final_ch

        # الطبقة النهائية مع GroupNorm و GELU وتهيئة الأوزان لمنع الـ Logits الميتة
        self.final_out = nn.Sequential(
            nn.Conv2d(head_in, 48, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=48), 
            nn.GELU(),
            nn.Conv2d(48, 2, kernel_size=3, padding=1),
        )

        nn.init.normal_(self.final_out[-1].weight, mean=0.0, std=0.05)
        if self.final_out[-1].bias is not None:
            nn.init.constant_(self.final_out[-1].bias, 0.0)

    def forward(self, encoder_output, concat_features):
        # 1. التقاط السياق الواسع في نقطة عنق الزجاجة
        x = self.aspp(encoder_output)
        
        # 2. عكس قائمة مسارات التخطي لمطابقة ترتيب الصعود
        features_to_fuse = concat_features[::-1] 

        # 3. فك التشفير مع بوابات الانتباه
        for i, (up_layer, db_layer) in enumerate(zip(self.up_convs, self.db_blocks)):
            # الرفع
            x = up_layer(x)
            
            # جلب الخصائص من الـ Encoder وتنقيتها عبر CBAM
            skip_feat = features_to_fuse[i]
            attn_mask = self.cbam_layers[i](skip_feat)
            skip_feat = skip_feat * attn_mask
            
            # الدمج
            x = torch.cat([x, skip_feat], dim=1)
            x = db_layer(x)

        if self.extra_up is not None:
            x = self.extra_up(x)
            x = self.extra_db(x)

        # 4. التنبؤ النهائي
        x_mask = self.final_out(x)
        return x_mask
