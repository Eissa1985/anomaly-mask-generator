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

        # Depthwise convolution

        self.depthwise_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)

        self.depthwise_norm = norm_layer(in_channel)

        self.depthwise_activation = nn.LeakyReLU(0.01)



        # Pointwise convolution

        self.pointwise_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

        self.norm1 = norm_layer(out_channel)

        self.activation1 = nn.LeakyReLU(0.01)



        # Additional convolution

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

       

        # ترتيب القنوات: [c1, c2, c3, c4, c5] (من الضحل للعميق)

        # نحن نحتاج للتحرك عكسياً من العميق للضحل

        self.in_channels = in_channels

        self.num_layers = len(in_channels)

       

        # قوائم لتخزين الطبقات بشكل ديناميكي

        self.up_convs = nn.ModuleList()

        self.db_blocks = nn.ModuleList()

       

        # نبدأ من القناة الأخيرة (الأعمق) ونتحرك للخلف

        # Logic: Deepest -> Up -> Concat with (Deepest-1) -> DBBlock

       

        for i in range(self.num_layers - 1, 0, -1):

            # القناة الحالية (العميقة)

            current_ch = in_channels[i]

            # القناة المستهدفة (الأقل عمقاً التي سندمج معها)

            target_ch = in_channels[i-1]

           

            # 1. تكبير الطبقة العميقة للنصف (حجم القناة)

            up = UpConvBlock(current_ch, current_ch // 2, norm_layer)

            self.up_convs.append(up)

           

            # 2. الدمج والمعالجة

            # الدخل لـ DBBlock سيكون: (نصف القناة العميقة) + (القناة المستهدفة كاملة)

            db = DBBlock(current_ch // 2 + target_ch, target_ch, norm_layer)

            self.db_blocks.append(db)

           

        # --- الطبقة النهائية (Head) ---

        # بعد انتهاء الحلقة، نكون وصلنا لآخر طبقة (in_channels[0])

        final_ch = in_channels[0]

       

        # إذا كان لدينا 5 طبقات (ResNet)، عادة نحتاج لخطوة إضافية للوصول للدقة العالية

        # إذا كان 4 (Swin)، نخرج مباشرة

       

        if self.num_layers >= 5:

            self.extra_up = UpConvBlock(final_ch, 48, norm_layer)

            self.extra_db = DBBlock(48, 24, norm_layer)

            head_in = 24

        else:

            self.extra_up = None

            self.extra_db = None

            head_in = final_ch



        # رأس الإخراج النهائي (2 Channels: Background, Anomaly)

        self.final_out = nn.Sequential(

            nn.Conv2d(head_in, 48, kernel_size=3, padding=1),

            norm_layer(48),

            nn.ReLU(inplace=True),

            nn.Conv2d(48, 2, kernel_size=3, padding=1),

        )


    def forward(self, encoder_output, concat_features, debug=False):
        """
        encoder_output: أعمق ميزة (التي خرجت من Backbone مباشرة)
        concat_features: قائمة الميزات الأقل عمقاً (بما في ذلك MSFF outputs)
        """
        if debug:
            print("\n" + "="*50)
            print("🏗️ [Decoder Module] Started")
            print(f"  [Input] Encoder Deepest Output: {encoder_output.shape}")
            for idx, f in enumerate(concat_features):
                print(f"  [Input] Skip Feature {idx} (Shallow -> Deep): {f.shape}")
            print("-" * 50)

        x = encoder_output
        features_to_fuse = concat_features[::-1] # عكس القائمة

        for i, (up_layer, db_layer) in enumerate(zip(self.up_convs, self.db_blocks)):
            # 1. تكبير الحجم
            x_up = up_layer(x)
            
            # 2. جلب الميزة المقابلة للدمج
            skip_feat = features_to_fuse[i]
            
            # 3. التأكد من تطابق الأحجام (Interpolation إذا لزم الأمر)
            if x_up.shape[2:] != skip_feat.shape[2:]:
                x_up = F.interpolate(x_up, size=skip_feat.shape[2:], mode='bilinear', align_corners=True)
            
            # 4. الدمج (Concatenation)
            x_cat = torch.cat([x_up, skip_feat], dim=1)
            
            # 5. المعالجة بـ DBBlock
            x = db_layer(x_cat)

            if debug:
                print(f"  [Loop {i}] Upconv: {x_up.shape} | Skip Feat: {skip_feat.shape}")
                print(f"            -> Concat: {x_cat.shape} -> DBBlock Output: {x.shape}")

        if debug:
            print("-" * 50)

        # معالجة إضافية في حالة 5 طبقات
        if self.extra_up is not None:
            x = self.extra_up(x)
            x = self.extra_db(x)
            if debug:
                print(f"  [Extra] Head Upconv & DBBlock -> Shape: {x.shape}")

        # الخرج النهائي
        x_mask = self.final_out(x)
        
        if debug:
            print(f"  [Final] Prediction Mask Output: {x_mask.shape}")
            print("="*50 + "\n")

        return x_mask

    # def forward(self, encoder_output, concat_features):

    #     """

    #     encoder_output: أعمق ميزة (التي خرجت من Backbone مباشرة)

    #     concat_features: قائمة الميزات الأقل عمقاً (بما في ذلك MSFF outputs)

    #     """

       

    #     # نبدأ بالخرج العميق

    #     x = encoder_output

       

    #     # نعكس ترتيب الميزات للدمج (لأننا نبدأ من العميق ونصعد للضحل)

    #     # concat_features تأتي مرتبة [shallow, ..., deep-1]

    #     # نحتاج لدمجها بالعكس

    #     features_to_fuse = concat_features[::-1] # عكس القائمة

       

    #     # الحلقة الديناميكية للدمج

    #     for i, (up_layer, db_layer) in enumerate(zip(self.up_convs, self.db_blocks)):

    #         # 1. تكبير الحجم

    #         x = up_layer(x)

           

    #         # 2. جلب الميزة المقابلة للدمج

    #         skip_feat = features_to_fuse[i]

           

    #         # 3. التأكد من تطابق الأحجام (Interpolation إذا لزم الأمر)

    #         if x.shape[2:] != skip_feat.shape[2:]:

    #             x = F.interpolate(x, size=skip_feat.shape[2:], mode='bilinear', align_corners=True)

           

    #         # 4. الدمج (Concatenation)

    #         x = torch.cat([x, skip_feat], dim=1)

           

    #         # 5. المعالجة بـ DBBlock

    #         x = db_layer(x)

           

    #     # معالجة إضافية في حالة 5 طبقات

    #     if self.extra_up is not None:

    #         x = self.extra_up(x)

    #         x = self.extra_db(x)

           

    #     # الخرج النهائي

    #     x_mask = self.final_out(x)

       

    #     return x_mask
