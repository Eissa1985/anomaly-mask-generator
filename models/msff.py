import torch
import torch.nn as nn
import torch.nn.functional as F
from .coordatt import CoordAtt

# (SA Code remains the same)
class SA(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(SA, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(in_channel)
        self.act1 = nn.ReLU(inplace=True)
        self.attn = CoordAtt(in_channel, in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_layer(in_channel)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channel, 2*in_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(2*in_channel)
        self.act3 = nn.ReLU(inplace=True)

    def forward(self, x, use_attn=True):
        x_conv = self.conv1(x)
        x_conv = self.bn1(x_conv)
        x_conv = self.act1(x_conv)        

        if use_attn:
            x_att = self.attn(x)
            out1 = x_conv * x_att
        else:
            out1 = x_conv 
        out2 = self.act2(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        w, b = out3[:, :self.in_channel, :, :], out3[:, self.in_channel:, :, :]
        out3 = self.act3(w * out2 + b)
        return out3

# (LearnableEdgeDetection Code remains the same)
# class LearnableEdgeDetection(nn.Module):
#     def __init__(self, input_channels, output_channels=1):
#         super(LearnableEdgeDetection, self).__init__()
#         self.edge_conv = nn.Sequential(
#             nn.Conv2d(input_channels, max(1, input_channels // 4), kernel_size=1),
#             nn.BatchNorm2d(max(1, input_channels // 4)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(max(1, input_channels // 4), output_channels, kernel_size=3, padding=1, bias=False)
#         )

#     def forward(self, x):
#         edge_map = self.edge_conv(x)
#         return torch.sigmoid(edge_map)

# class FrequencyAwareBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super().__init__()
#         self.in_channels = in_channels
        
#         # طبقة انتباه ترددي تتعلم فلترة الترددات غير الطبيعية
#         self.freq_weight = nn.Parameter(torch.ones(1, in_channels, 1, 1))
#         self.spatial_conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
#             nn.BatchNorm2d(in_channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction, in_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, h, w = x.shape
        
#         # 1. تحويل الميزة إلى الفضاء الترددي
#         x_fft = torch.fft.rfft2(x.to(torch.float32), norm='ortho')
        
#         # 2. تعديل الأطياف الترددية بالأوزان القابلة للتعلم لتعزيز ترددات الشذوذ
#         weighted_fft = x_fft * self.freq_weight
        
#         # 3. العودة إلى الفضاء المكاني (Spatial Domain)
#         x_freq_enhanced = torch.fft.irfft2(weighted_fft, s=(h, w), norm='ortho')
#         x_freq_enhanced = x_freq_enhanced.to(x.dtype)
        
#         # 4. دمج الانتباه المكاني والترددي عبر اتصال متبقي (Residual Connection)
#         attn_mask = self.spatial_conv(x - x_freq_enhanced)
        
#         return x * (1 + attn_mask)

class FrequencyAwareEdgeDetection(nn.Module):
    """
    وحدة كشف حواف هجينة (مكاني + ترددي) مصممة خصيصاً للعيوب البنيوية الدقيقة في المنسوجات.
    تكتشف التمزقات والحواف الواضحة عبر الفرع المكاني، وتكتشف غياب الخيوط والتشوهات الدورية عبر فرع فورييه.
    """
    def __init__(self, input_channels, output_channels=1):
        super(FrequencyAwareEdgeDetection, self).__init__()
        
        # 1. الفرع المكاني (Spatial Branch - نفس خوارزميتك الأصلية للحفاظ على استقرار كشف الحواف المباشرة)
        self.spatial_edge = nn.Sequential(
            nn.Conv2d(input_channels, max(1, input_channels // 4), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(1, input_channels // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, input_channels // 4), output_channels, kernel_size=3, padding=1, bias=False)
        )
        
        # 2. الفرع الترددي (Spectral Branch - لاقتناص شذوذ الترددات مثل Single Mispick)
        # نستخدم InstanceNorm لتطبيع طاقة فورييه لكل صورة على حدة لإبراز التردد الشاذ
        self.spectral_filter = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, groups=input_channels, bias=False),
            nn.InstanceNorm2d(input_channels, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.spectral_projector = nn.Sequential(
            nn.Conv2d(input_channels, max(1, input_channels // 4), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(1, input_channels // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, input_channels // 4), output_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        
        # --- 1. التمرير المكاني ---
        spatial_out = self.spatial_edge(x)
        
        # --- 2. التمرير الترددي في الفضاء الكامن (Latent Frequency Domain) ---
        # تحويل إجباري إلى float32 لضمان استقرار حسابات فورييه عند التدريب بالدقة النصفية FP16 (AMP)
        x_fp32 = x.to(torch.float32)
        x_fft = torch.fft.rfft2(x_fp32, norm='ortho')
        
        # فصل طيف السعة (Amplitude) عن طيف الطور (Phase)
        amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # فلترة السعة الترددية لإبراز التشوهات في الدورية النسيجية
        amp_filtered = self.spectral_filter(amp)
        
        # إعادة بناء العدد المركب ثم العودة إلى المجال المكاني عبر IFFT
        x_complex_mod = torch.polar(amp_filtered, phase)
        x_spatial_mod = torch.fft.irfft2(x_complex_mod, s=(h, w), norm='ortho')
        
        # إعادة التنسيق لمطابقة دقة المدخلات الأصلي (FP32 أو FP16)
        x_spatial_mod = x_spatial_mod.to(x.dtype)
        spectral_out = self.spectral_projector(x_spatial_mod)
        
        # --- 3. دمج الحواف المكانية والترددية ---
        combined_edge_map = spatial_out + spectral_out
        return torch.sigmoid(combined_edge_map)
    
class MSFF(nn.Module):
    def __init__(self, in_channels_list, norm_layer=nn.BatchNorm2d):
        super(MSFF, self).__init__()
        self.blocks = nn.ModuleList([SA(ch) for ch in in_channels_list])
        # self.edges = nn.ModuleList([LearnableEdgeDetection(ch) for ch in in_channels_list])
        # self.freq_blocks = nn.ModuleList([FrequencyAwareBlock(ch) for ch in in_channels_list])
        self.edges = nn.ModuleList([FrequencyAwareEdgeDetection(ch) for ch in in_channels_list])
        
        self.upconvs = nn.ModuleList()
        for i in range(len(in_channels_list) - 1):
            up_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels_list[i+1],
                    in_channels_list[i],  
                    kernel_size=2, stride=2
                ),
                norm_layer(in_channels_list[i]),
                nn.ReLU(inplace=True)            
            )
            self.upconvs.append(up_layer)

    def forward(self, features, debug=True):
        m_list = [edge(f) for edge, f in zip(self.edges, features)]
        
        f_k_list = [blk(f) for blk, f in zip(self.blocks, features)]
        f_f_list = [None] * len(f_k_list)
        f_f_list[-1] = f_k_list[-1]

        for i in range(len(f_k_list) - 2, -1, -1):
            up_feat = self.upconvs[i](f_f_list[i+1])
            f_f_list[i] = f_k_list[i] + up_feat

        f_out_list = []
        for i in range(len(f_f_list)):
            combined_mask = m_list[i]
            for j in range(i + 1, len(m_list)):
                scale = 2 ** (j - i)
                upsampled_m = F.interpolate(
                    m_list[j], scale_factor=scale, mode='bilinear', align_corners=True
                )
                # combined_mask = combined_mask * upsampled_m
                combined_mask = torch.max(combined_mask, upsampled_m)

            
            # f_out = f_f_list[i] * combined_mask
            f_out = f_f_list[i] * (1 + combined_mask)
            f_out_list.append(f_out)

        return f_out_list
