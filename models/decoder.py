import torch
import torch.nn as nn
import torch.nn.functional as F

class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=nn.BatchNorm2d):
        super(UpConvBlock, self).__init__()
        self.blk = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.blk(x)

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

class Decoder(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.num_layers = len(in_channels)
        self.up_convs = nn.ModuleList()
        self.db_blocks = nn.ModuleList()

        for i in range(self.num_layers - 1, 0, -1):
            current_ch = in_channels[i]
            target_ch = in_channels[i-1]
            up = UpConvBlock(current_ch, current_ch // 2, norm_layer)
            self.up_convs.append(up)
            db = DBBlock(current_ch // 2 + target_ch, target_ch, norm_layer)
            self.db_blocks.append(db)

        final_ch = in_channels[0]

        if self.num_layers >= 5:
            self.extra_up = UpConvBlock(final_ch, 48, norm_layer)
            self.extra_db = DBBlock(48, 24, norm_layer)
            head_in = 24

        else:
            self.extra_up = None
            self.extra_db = None
            head_in = final_ch

        # --- التعديلات المطلوبة تبدأ هنا ---
        # استخدام GroupNorm و GELU للسماح بتباين أعلى وتجنب إخماد التدرج
        self.final_out = nn.Sequential(
            nn.Conv2d(head_in, 48, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=48), 
            nn.GELU(),
            nn.Conv2d(48, 2, kernel_size=3, padding=1),
        )

        # تهيئة أوزان الطبقة الأخيرة لتوسيع نطاق الـ Logits وتجنب النقطة الميتة
        nn.init.normal_(self.final_out[-1].weight, mean=0.0, std=0.05)
        if self.final_out[-1].bias is not None:
            nn.init.constant_(self.final_out[-1].bias, 0.0)
        # --- نهاية التعديلات ---

    def forward(self, encoder_output, concat_features):
        x = encoder_output
        features_to_fuse = concat_features[::-1] # عكس القائمة

        for i, (up_layer, db_layer) in enumerate(zip(self.up_convs, self.db_blocks)):
            x = up_layer(x)
            skip_feat = features_to_fuse[i]
            x = torch.cat([x, skip_feat], dim=1)
            x = db_layer(x)

        if self.extra_up is not None:
            x = self.extra_up(x)
            x = self.extra_db(x)

        x_mask = self.final_out(x)

        return x_mask
