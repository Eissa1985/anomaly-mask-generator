import torch
import torch.nn as nn
import torch.nn.functional as F
from .coordatt import CoordAtt # تأكد من وجود ملف coordatt.py

# 1. بلوك الانتباه القياسي (بقاءه كما هو لأنه ممتاز لاستخلاص السياق)
class StandardCoordBlock(nn.Module):
    def __init__(self, in_channels):
        super(StandardCoordBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.attn = CoordAtt(in_channels, in_channels)
        
    def forward(self, x):
        feat = self.conv(x)
        attn_mask = self.attn(feat)
        out = feat * attn_mask
        return x + out


# 2. وحدة كشف الحواف الترددية الرشيقة
class LeanFrequencyEdgeDetection(nn.Module):
    def __init__(self, input_channels, output_channels=1):
        super(LeanFrequencyEdgeDetection, self).__init__()
        mid_ch = max(1, input_channels // 4)
        self.spatial_edge = nn.Sequential(
            nn.Conv2d(input_channels, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, output_channels, kernel_size=3, padding=1, bias=False)
        )
        
        self.spectral_filter = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.GELU() 
        )
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
    

# 3. وحدة الـ MSFF المحسنة (IoU Booster)
class MSFF(nn.Module):
    def __init__(self, in_channels_list, norm_layer=nn.BatchNorm2d):
        super(MSFF, self).__init__()
        
        self.blocks = nn.ModuleList([StandardCoordBlock(ch) for ch in in_channels_list])
        self.edges = nn.ModuleList([LeanFrequencyEdgeDetection(ch) for ch in in_channels_list])
        
        # --- التعديل الجوهري لرفع الـ IoU ---
        # استخدام دمج قابل للتعلم بدلاً من الضرب المباشر (1+m).
        # نضيف +1 لعدد القنوات لاستيعاب قناع الحافة المدمج (Concatenation).
        self.learned_edge_fusers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch + 1, ch, kernel_size=1, bias=False),
                norm_layer(ch),
                nn.GELU()
            ) for ch in in_channels_list
        ])

    def forward(self, features, debug=False):
        # 1. استخراج الأقنعة المكانية والترددية للحواف
        m_list = [edge(f) for edge, f in zip(self.edges, features)]
        
        # 2. تعزيز الخصائص باستخدام الانتباه الإحداثي (بشكل مستقل لكل مقياس)
        f_k_list = [blk(f) for blk, f in zip(self.blocks, features)]
        
        # 3. تراكم الأقنعة من الطبقات العميقة للسطحية (يوفر حواف هيكلية قوية للـ IoU)
        cum_masks = [None] * len(m_list)
        cum_masks[-1] = m_list[-1]
        for i in range(len(m_list) - 2, -1, -1):
            up_mask = F.interpolate(cum_masks[i+1], size=m_list[i].shape[2:], mode='bilinear', align_corners=False)
            cum_masks[i] = torch.max(m_list[i], up_mask)

        # 4. الدمج الذكي (Learned Edge Prompting)
        # هذا هو السر لرفع IoU: نجعل الشبكة "تتعلم" كيف تدمج الحافة مع الخصائص
        # بدلاً من طمس الخصائص بدمج هرمي مكرر (الذي يتكفل به المفكك لاحقاً).
        f_out_list = []
        for i in range(len(f_k_list)):
            # لصق خريطة الخصائص مع قناع الحافة (قناة إضافية)
            f_and_m = torch.cat([f_k_list[i], cum_masks[i]], dim=1)
            # دمجهم برمجياً عبر طبقة 1x1 لضبط الأوزان بدقة لكل قناة
            f_out = self.learned_edge_fusers[i](f_and_m)
            f_out_list.append(f_out)

        return f_out_list
