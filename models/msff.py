import torch
import torch.nn as nn
import torch.nn.functional as F
from coordatt import CoordAtt # تأكد من وجود ملف coordatt.py

# 1. بلوك الانتباه القياسي (الأصح والأكثر استقراراً)
class StandardCoordBlock(nn.Module):
    def __init__(self, in_channels):
        super(StandardCoordBlock, self).__init__()
        
        # استخلاص الخصائص المكانية (Local Features)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        
        # وحدة الانتباه (CoordAtt) في النهاية (Global Weighting)
        self.attn = CoordAtt(in_channels, in_channels)
        
    def forward(self, x):
        feat = self.conv(x)
        
        # أوزان الانتباه تضرب مباشرة في الخصائص
        attn_mask = self.attn(feat)
        out = feat * attn_mask
        
        # الاتصال المتبقي للحفاظ على طاقة الخصائص (Residual)
        return x + out


# 2. وحدة كشف الحواف الترددية الرشيقة
class LeanFrequencyEdgeDetection(nn.Module):
    def __init__(self, input_channels, output_channels=1):
        super(LeanFrequencyEdgeDetection, self).__init__()
        
        # الفرع المكاني (Spatial)
        mid_ch = max(1, input_channels // 4)
        self.spatial_edge = nn.Sequential(
            nn.Conv2d(input_channels, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, output_channels, kernel_size=3, padding=1, bias=False)
        )
        
        # الفرع الترددي (Spectral) - فلترة آمنة لا تصفر الترددات
        self.spectral_filter = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.GELU() 
        )
        
        # إسقاط خفيف لقناة الحواف الترددية
        self.spectral_proj = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        
        spatial_out = self.spatial_edge(x)
        
        x_fft = torch.fft.rfft2(x.to(torch.float32), norm='ortho')
        amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        amp_filtered = self.spectral_filter(amp)
        x_spatial_mod = torch.fft.irfft2(torch.polar(amp_filtered, phase), s=(h, w), norm='ortho')
        
        spectral_out = self.spectral_proj(x_spatial_mod.to(x.dtype))
        
        return torch.sigmoid(spatial_out + spectral_out)
    

# 3. وحدة الدمج الكلية (خالية من الالتفاف العكسي والانخماص)
class MSFF(nn.Module):
    def __init__(self, in_channels_list, norm_layer=nn.BatchNorm2d):
        super(MSFF, self).__init__()
        
        # استخدام البلوك القياسي الأصح
        self.blocks = nn.ModuleList([StandardCoordBlock(ch) for ch in in_channels_list])
        self.edges = nn.ModuleList([LeanFrequencyEdgeDetection(ch) for ch in in_channels_list])
        
        # استخدام Conv1x1 للمواءمة بدلاً من ConvTranspose2d الثقيلة
        self.align_convs = nn.ModuleList()
        for i in range(len(in_channels_list) - 1):
            align_layer = nn.Sequential(
                nn.Conv2d(in_channels_list[i+1], in_channels_list[i], kernel_size=1, bias=False),
                norm_layer(in_channels_list[i]),
                nn.GELU()
            )
            self.align_convs.append(align_layer)

    def forward(self, features, debug=False):
        # 1. الأقنعة (Edges)
        m_list = [edge(f) for edge, f in zip(self.edges, features)]
        
        # 2. الخصائص المكانية (Blocks)
        f_k_list = [blk(f) for blk, f in zip(self.blocks, features)]
        
        # 3. الدمج الهرمي الخطي (Linear Upsampling Fusion)
        f_f_list = [None] * len(f_k_list)
        f_f_list[-1] = f_k_list[-1]
        
        for i in range(len(f_k_list) - 2, -1, -1):
            up_feat = F.interpolate(f_f_list[i+1], size=f_k_list[i].shape[2:], mode='bilinear', align_corners=False)
            f_f_list[i] = f_k_list[i] + self.align_convs[i](up_feat)

        # 4. تراكم الأقنعة الخطي O(M) لمنع انخماص القيم
        cum_masks = [None] * len(m_list)
        cum_masks[-1] = m_list[-1]
        
        for i in range(len(m_list) - 2, -1, -1):
            up_mask = F.interpolate(cum_masks[i+1], size=m_list[i].shape[2:], mode='bilinear', align_corners=False)
            cum_masks[i] = torch.max(m_list[i], up_mask)

        # 5. ضرب الخصائص في (1 + القناع) للحفاظ على طاقة الخريطة وتضخيم الشذوذ
        f_out_list = [f * (1 + m) for f, m in zip(f_f_list, cum_masks)]
        return f_out_list
