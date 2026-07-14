from .decoder import Decoder
# from .msff import MSFF
from utils.metrics import AnomalyEvaluator
import torch
import torch.nn as nn
import torch.nn.functional as F
# from losses import CompositeLoss, FocalLoss, SpectralLoss
from losses import IoUOptimizedLoss
import torch.optim as optim
from tqdm import tqdm
import time
import os
import numpy as np
import gc
import torchvision
import math

import time
import json
import os
import logging
from typing import List
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.metrics import precision_score, f1_score, precision_recall_curve, roc_curve
from lion_pytorch import Lion
from scheduler import CosineAnnealingWarmupRestarts
import wandb
from timm import create_model
from .hybrid_segmentor import MiT, DoubleConv
import random
from glob import glob
import math
from PIL import Image
import torchvision.transforms.functional as TF
import warnings
from .u2net import U2NET
from torch.autograd import Variable

warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')

logger = logging.getLogger(__name__)

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

# # ==========================================
# # 1. Metrics & Losses (المقاييس ودوال الخسارة)
# # ==========================================
# def dice_coefficient(y_true, y_pred, smooth=100.0):
#     # تسطيح المصفوفات
#     y_true_f = y_true.reshape(-1)
#     y_pred_f = y_pred.reshape(-1)
    
#     intersection = torch.sum(y_true_f * y_pred_f)
#     union = torch.sum(y_true_f) + torch.sum(y_pred_f)
#     return (2. * intersection + smooth) / (union + smooth)

# def dice_coefficient_loss(y_true, y_pred, smooth=100.0):
#     # في PyTorch نستخدم (1 - dice) لكي نجعل الدالة تصغر نحو الصفر
#     return 1.0 - dice_coefficient(y_true, y_pred, smooth)

# def iou_score(y_true, y_pred, smooth=100.0):
#     y_true_f = y_true.reshape(-1)
#     y_pred_f = y_pred.reshape(-1)
    
#     intersection = torch.sum(y_true_f * y_pred_f)
#     sum_vals = torch.sum(y_true_f + y_pred_f)
#     return (intersection + smooth) / (sum_vals - intersection + smooth)

# def jaccard_distance(y_true, y_pred, smooth=100.0):
#     return 1.0 - iou_score(y_true, y_pred, smooth)

# # ------- 1. define loss function --------

# bce_loss = nn.BCELoss(size_average=True)

# def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

#     # 1. أضف هذا السطر لحل مشكلة الأبعاد
#     if labels_v.dim() == 3:
#         labels_v = labels_v.unsqueeze(1)
        
#     # 2. تأكد أن الهدف من نوع Float 
#     labels_v = labels_v.float()

#     loss0 = bce_loss(d0,labels_v)
#     loss1 = bce_loss(d1,labels_v)
#     loss2 = bce_loss(d2,labels_v)
#     loss3 = bce_loss(d3,labels_v)
#     loss4 = bce_loss(d4,labels_v)
#     loss5 = bce_loss(d5,labels_v)
#     loss6 = bce_loss(d6,labels_v)

#     loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
#     # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

#     return loss0, loss
       
# class ChannelProjector(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ChannelProjector, self).__init__()
#         self.projector = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.projector(x)

class AnomalyTransplanter(nn.Module):
    def __init__(self, anomaly_root_dir, target, img_size=224, p_anomaly=0.5, p_blur=0.3, p_illum=0.4):
        """
        نسخة صناعية مطورة تمنع الألوان الفسفورية وتولد عيوب نسيج وأقمشة واقعية 100%
        """
        super().__init__()
        self.img_size = img_size
        self.p_anomaly = p_anomaly
        self.p_blur = p_blur
        self.p_illum = p_illum

        # 1. تحميل كافة الأقنعة الجاهزة في الذاكرة لتسريع الوصول
        self.anomaly_source_path = []
        sub_dir = target
        full_dir = os.path.join(anomaly_root_dir, sub_dir)
        print(full_dir)
        if os.path.exists(full_dir):
            paths = glob(os.path.join(full_dir, "*/*.*"))
            self.anomaly_source_path.extend([p for p in paths if p.endswith(('.png', '.jpg', '.bmp', '.tif'))])
        
        if len(self.anomaly_source_path) == 0:
            raise RuntimeError(f"لم يتم العثور على أي أقنعة في المسار: {full_dir}!")
            
        print(f"--> [GPU Augmenter - Textile Edition] Successfully loaded {len(self.anomaly_source_path)} offline mask paths.")

    def transplant_anomaly(self, images, inject_mask, device):
        b, c, h, w = images.shape
        
        # إنشاء مصفوفات النتائج مبدئياً كنسخ نظيفة
        result_images = images.clone()
        final_masks = torch.zeros((b, 1, h, w), device=device)
        
        # --- الفلترة الذكية ---
        # استخراج فهارس (Indices) الصور التي تم اختيارها للتشويه فقط
        inject_indices = [i for i in range(b) if inject_mask[i, 0, 0, 0].item() > 0]
        
        # إذا لم يتم اختيار أي صورة، نرجع المصفوفات النظيفة فوراً
        if len(inject_indices) == 0:
            return result_images, final_masks
            
        # سحب مسارات عشوائية بعدد الصور المطلوبة فقط!
        sampled_paths = random.choices(self.anomaly_source_path, k=len(inject_indices))
        
        # المرور فقط على الصور التي سيتم تشويهها
        for idx, path in zip(inject_indices, sampled_paths):
            
            # 1. قراءة صورة العيب
            with Image.open(path) as a_img:
                a_img = a_img.convert("RGB")
                if a_img.size != (self.img_size, self.img_size):
                    a_img = a_img.resize((self.img_size, self.img_size))
                t_img = TF.to_tensor(np.array(a_img)).to(device, non_blocking=True)
                    
            # 2. قراءة قناع العيب
            mask_path = path.replace('images', 'masks')
            base_name, ext = os.path.splitext(mask_path)
            mask_path = base_name + '_mask' + ext
            with Image.open(mask_path) as msk:
                msk = msk.convert("L")
                if msk.size != (self.img_size, self.img_size):
                    # استخدام NEAREST للحفاظ على القناع حاداً وثنائياً كما في الكود القديم
                    msk = msk.resize((self.img_size, self.img_size), Image.NEAREST)
                m_img = TF.to_tensor(np.array(msk)).to(device, non_blocking=True)[0]
            
            # 3. إيجاد حدود العيب واقتطاعه
            indices = torch.nonzero(m_img > 0)
            if indices.size(0) == 0:
                continue # تخطي إذا كان القناع فارغاً
                
            y_min, x_min = torch.min(indices, dim=0)[0]
            y_max, x_max = torch.max(indices, dim=0)[0]
            
            # --- الإصلاح الأول: إضافة +1 لضمان أخذ كامل مساحة العيب (Slicing Fix) ---
            crop_mask = m_img[y_min:y_max+1, x_min:x_max+1]
            crop_anomaly = t_img[:, y_min:y_max+1, x_min:x_max+1]
            
            # --- الإصلاح الثاني: تحويل القناع إلى ثنائي بصرامة (Binary Mask) ---
            hard_mask = (crop_mask > 0).float()
            crop_anomaly = crop_anomaly * hard_mask
            
            c_h, c_w = crop_mask.shape
            
            # 4. اختيار مكان عشوائي للدمج
            max_h = max(1, h - c_h)
            max_w = max(1, w - c_w)
            
            place_h = random.randint(0, max_h - 1)
            place_w = random.randint(0, max_w - 1)
            
            # 5. دمج العيب في الصورة 
            target_roi = result_images[idx, :, place_h:place_h+c_h, place_w:place_w+c_w]
            
            # --- الإصلاح الثالث: الاستبدال القاسي بدلاً من الدمج الناعم (Hard Replacement) ---
            # هذا يطابق كودك القديم bg_img[result_img > 0] = 0 تماماً
            anomaly_exists = (crop_anomaly > 0).any(dim=0).float() 
            
            result_images[idx, :, place_h:place_h+c_h, place_w:place_w+c_w] = target_roi * (1.0 - anomaly_exists) + crop_anomaly
            final_masks[idx, 0, place_h:place_h+c_h, place_w:place_w+c_w] = hard_mask

        return result_images, final_masks

    def _apply_motion_blur(self, x):
        """
        محاكاة اهتزاز وحركة خط الإنتاج السريعة (Anisotropic Motion Blur)
        """
        if torch.rand(1).item() > self.p_blur:
            return x
        
        kernel_size = int(torch.randint(3, 9, (1,)).item()) | 1  # رقم فردي دائماً
        kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=x.device)
        
        if torch.rand(1).item() > 0.5:
            kernel[0, 0, kernel_size // 2, :] = 1.0 / kernel_size  # حركة أفقية (سحب القماش)
        else:
            kernel[0, 0, :, kernel_size // 2] = 1.0 / kernel_size  # حركة رأسية
            
        kernel = kernel.repeat(x.shape[1], 1, 1, 1)
        x_blurred = F.conv2d(x, kernel, padding=kernel_size//2, groups=x.shape[1])
        return x_blurred

    def _apply_illumination_gradient(self, x):
        """
        محاكاة الظلال وتغير الإضاءة على القماش على خط الإنتاج.
        """
        if torch.rand(1).item() > self.p_illum:
            return x
            
        b, c, h, w = x.shape
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing='ij'
        )
        
        angle = torch.rand(1, device=x.device) * 2 * math.pi
        gradient = (x_grid * torch.cos(angle) + y_grid * torch.sin(angle))
        
        gradient = 0.7 + 0.6 * ((gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8))
        gradient = gradient.unsqueeze(0).unsqueeze(0).expand_as(x)
        
        return torch.clamp(x * gradient, 0.0, 1.0)
        
    def forward(self, images, anomaly_textures=None):
        batch_size = images.shape[0]
        device = images.device
        # print(images.shape)

        # 1. تطبيق تأثيرات البيئة الصناعية (ظلال واهتزاز) على الصورة النظيفة أولاً
        # images = self._apply_illumination_gradient(images)
        # images = self._apply_motion_blur(images)
        
        # 2. قناع اختيار الصور التي سيحقن بها عيوب
        inject_mask = (torch.rand(batch_size, device=device) < self.p_anomaly).float().view(batch_size, 1, 1, 1)
        
        # print(inject_mask)
        
        if inject_mask.sum() == 0:
            zeros_mask = torch.zeros((batch_size, 1, self.img_size, self.img_size), device=device)

            images = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            return images, zeros_mask, torch.zeros(batch_size, device=device, dtype=torch.long)
            
        # 3. سحب الأقنعة الجاهزة (تصبح ناعمة الحواف الآن)
        a_imgs, fault_masks = self.transplant_anomaly(images, inject_mask, device)
        
        targets = (fault_masks.view(batch_size, -1).max(dim=1)[0] > 0.1).long()
        
        # إعادة تقريب القناع ليكون ثنايياً لحسابات الخسارة الدقيقة في التدريب
        train_masks = torch.where(fault_masks > 0.2, 1.0, 0.0)
        a_imgs = TF.normalize(a_imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return a_imgs, train_masks, targets

########################
# # ==========================================
# # 2. Network Blocks (وحدات بناء الشبكة)
# # ==========================================
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout_rate=0.2):
#         super(DoubleConv, self).__init__()
#         self.step1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.Dropout2d(dropout_rate),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.step2 = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.Dropout2d(dropout_rate),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.step1(x)
#         x = self.step2(x)
#         return x

# # ملاحظة: تم حذف EncoderBlock لأننا سنستخدم cnn_backbone بدلاً منه

# class SpatialAttentionBlock(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         max_pool = torch.max(x, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x, dim=1, keepdim=True)
#         concat = torch.cat([max_pool, avg_pool], dim=1)
#         attention = self.sigmoid(self.conv(concat))
#         return x * attention

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, skip_channels, num_filters, dropout_rate=0.2):
#         super(DecoderBlock, self).__init__()
#         self.up = nn.ConvTranspose2d(in_channels, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.conv = DoubleConv(num_filters + skip_channels, num_filters, dropout_rate)

#     def forward(self, x, skip_features):
#         x = self.up(x)
        
#         # مطابقة الأبعاد مكانياً في حال كان هناك اختلاف طفيف (تأمين إضافي)
#         if x.shape[2:] != skip_features.shape[2:]:
#             x = F.interpolate(x, size=skip_features.shape[2:], mode='bilinear', align_corners=False)
            
#         x = torch.cat([x, skip_features], dim=1)
#         x = self.conv(x)
#         return x
# ###################################

class EEMFNet(nn.Module):
    # def __init__(self, device='cpu', config=None):
    def __init__(self, device='cpu', config=None, in_channels=3, num_classes=2, img_size=224):

        super(EEMFNet, self).__init__()
        self.opts_list = {
            "adamw": optim.AdamW,
            "adam": optim.Adam,
            "lion": Lion,
            }

        self.device = device
        self.config = config

        self.augmenter = AnomalyTransplanter(
            anomaly_root_dir="datasets/anomaly_generation_datasets/images",
            target = "carpet",
            img_size=224, 
            p_anomaly=0.5,  # 1.0 لضمان ظهور الشذوذ في كل صور الباتش للتجربة
            p_blur=1.0,     
            p_illum=1.0     
        ).to(device)

        self.evaluator = AnomalyEvaluator(pro_integration_limit=0.3)
        backbone_name = config.backbone_name if config else "resnet18"

        #################
        self.cnn_backbone = create_model(
            backbone_name,
            pretrained=True,
            in_chans=in_channels,
            features_only=True # استخراج مسارات التخطي تلقائياً
        )
        
        # تجميد الأوزان بالكامل للـ Encoder
        for p in self.cnn_backbone.parameters():
            p.requires_grad = False
            
        # استخراج قنوات الـ Backbone ديناميكياً (مثل [64, 64, 128, 256, 512] للـ ResNet34)
        cnn_channels = self.cnn_backbone.feature_info.channels()

        # # 2. Spatial Attention يطبق على أعمق خريطة
        # self.sa = SpatialAttentionBlock()

        # # 3. بناء الـ Decoder ديناميكياً ليتكيف مع أي موديل
        # self.decoders = nn.ModuleList()
        # decoder_in_ch = cnn_channels[-1]
        
        # # بناء الـ Decoder عكسياً من القنوات العميقة إلى السطحية
        # for i in range(len(cnn_channels) - 2, -1, -1):
        #     skip_ch = cnn_channels[i]
        #     num_filters = skip_ch 
        #     self.decoders.append(
        #         DecoderBlock(decoder_in_ch, skip_ch, num_filters, dropout_rate=0.2)
        #     )
        #     decoder_in_ch = num_filters

        # # 4. التكبير النهائي (للرجوع إلى دقة الصورة الأصلية 1/1)
        # self.final_up = nn.ConvTranspose2d(decoder_in_ch, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.final_conv = nn.Sequential(
        #     DoubleConv(32, 16),
        #     nn.Conv2d(16, 2, kernel_size=1)
        # )
        # # self.sigmoid = nn.Sigmoid()
        #####################
        # self.net = U2NET(3, 1)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        
        # logger.info(f"--> Building Backbone: {backbone_name}")

        # try:
        #     self.cnn_backbone = create_model(
        #         backbone_name,
        #         pretrained=True,
        #         features_only=True
        #     )
        #     for p in self.cnn_backbone.parameters():
        #             p.requires_grad = False
        #     # for name, param in self.cnn_backbone.named_parameters():
        #     #     if "blocks.4" in name or "blocks.5" in name or "conv_head" in name:
        #     #         param.requires_grad = True
        #     #     else:
        #     #         param.requires_grad = False

        # except RuntimeError as e:
        #     logger.warning(f"Error.... Default indices failed for {backbone_name}, trying default behavior. c: {e}")

        # cnn_channels = self.cnn_backbone.feature_info.channels()
        
        

        # mit_dims = (64, 128, 320, 512)
        # self.trans_backbone = MiT(channels=3, dims=mit_dims, n_heads=(1, 2, 5, 8),
        #                           expansion=(8, 8, 4, 4), reduction_ratio=(8, 4, 2, 1),
        #                           n_layers=(2, 2, 2, 2))
        
        # # (يفترض وجود كود تحميل الأوزان هنا كما في كودك الأصلي... تم تخطيه للاختصار)

        # # ⚠️ الحل العلمي: لا تقم بتجميد الـ Transformer بالكامل!
        # # نفتح آخر طبقتين (Block 3 & 4) للتعلم الدقيق لكي يتأقلم مع أنسجة السجاد
        # for name, param in self.trans_backbone.named_parameters():
        #     if "block3" in name or "block4" in name or "norm3" in name or "norm4" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # # --- ج. وحدات الدمج الهجين (Attentional Feature Fusion) بدلاً من CrossAttention ---
        # self.fusion_blocks = nn.ModuleList([
        #     AttentionalFeatureFusion(cnn_dim=cnn_channels[1], trans_dim=mit_dims[0]),
        #     AttentionalFeatureFusion(cnn_dim=cnn_channels[2], trans_dim=mit_dims[1]),
        #     AttentionalFeatureFusion(cnn_dim=cnn_channels[3], trans_dim=mit_dims[2]),
        #     AttentionalFeatureFusion(cnn_dim=cnn_channels[4], trans_dim=mit_dims[3])
        # ])

        # # (تم حذف self.upsampling لأنه كود ميت لا يُستخدم)

        # # --- د. تجهيز الأبعاد والمفكك ---
        # self.base_dim = 48
        # self.target_channels = [self.base_dim * (2 ** i) for i in range(len(cnn_channels))]

        # self.projections = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(src, tgt, kernel_size=1, bias=False),
        #         nn.GroupNorm(num_groups=8, num_channels=tgt),
        #         nn.ReLU(inplace=True)
        #     ) for src, tgt in zip(cnn_channels, self.target_channels)
        # ])
        
        # mit_dims = (64, 128, 320, 512)
        # self.trans_backbone = MiT(channels=3, dims=mit_dims, n_heads=(1, 2, 5, 8),
        #                           expansion=(8, 8, 4, 4), reduction_ratio=(8, 4, 2, 1),
        #                           n_layers=(2, 2, 2, 2))

        # try:
        #     logger.info("--> Attempting to download official MiT-B2 weights...")
            
        #     weight_urls = [
        #         "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth",
        #         "https://huggingface.co/jishi/SegFormer-mit-b2-imagenet-1k/resolve/main/mit_b2.pth"
        #     ]
            
        #     state_dict = None
        #     for url in weight_urls:
        #         try:
        #             logger.info(f"Downloading from: {url}")
        #             checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu', progress=True)
        #             state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        #             break 
        #         except Exception as dl_err:
        #             logger.warning(f"Failed to download from {url}. Trying next source...")
        #             continue
            
        #     if state_dict is None:
        #         raise RuntimeError("All weight servers failed or are unreachable.")

        #     clean_state_dict = {}
        #     for k, v in state_dict.items():
        #         clean_key = k.replace('backbone.', '').replace('encoder.', '')
        #         clean_state_dict[clean_key] = v

        #     missing_keys, unexpected_keys = self.trans_backbone.load_state_dict(clean_state_dict, strict=False)
        #     logger.info("--> SUCCESS: Pre-trained MiT-B2 weights successfully injected!")
        #     if missing_keys:
        #         logger.debug(f"Expected unmapped keys (e.g. classification head): {len(missing_keys)} keys.")

        # except Exception as e:
        #     logger.warning(f"--> Could not inject pre-trained weights: {e}. Model will train from scratch.")

        # # for p in self.trans_backbone.parameters():
        # #     p.requires_grad = False
        # for name, param in self.trans_backbone.named_parameters():
        #     if "block3" in name or "block4" in name or "norm3" in name or "norm4" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # # --- ج. وحدات الدمج الهجين (Attentional Feature Fusion) بدلاً من CrossAttention ---
        # self.fusion_blocks = nn.ModuleList([
        #     AttentionalFeatureFusion(cnn_dim=cnn_channels[1], trans_dim=mit_dims[0]),
        #     AttentionalFeatureFusion(cnn_dim=cnn_channels[2], trans_dim=mit_dims[1]),
        #     AttentionalFeatureFusion(cnn_dim=cnn_channels[3], trans_dim=mit_dims[2]),
        #     AttentionalFeatureFusion(cnn_dim=cnn_channels[4], trans_dim=mit_dims[3])
        # ])

        # # (تم حذف self.upsampling لأنه كود ميت لا يُستخدم)

        # # --- د. تجهيز الأبعاد والمفكك ---
        # self.base_dim = 48
        # self.target_channels = [self.base_dim * (2 ** i) for i in range(len(cnn_channels))]

        # self.projections = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(src, tgt, kernel_size=1, bias=False),
        #         nn.GroupNorm(num_groups=8, num_channels=tgt),
        #         nn.ReLU(inplace=True)
        #     ) for src, tgt in zip(cnn_channels, self.target_channels)
        # ])
        # # self.msff = MSFF(in_channels[1:-1]).to(self.device)
        # # self.msff = MSFF(self.target_channels[1:-1]).to(self.device)
        # # self.decoder = Decoder(in_channels).to(self.device)
        self.decoder = Decoder(cnn_channels).to(self.device)
        # self.decoder = Decoder(self.target_channels).to(self.device)
        # self.cnn_backbone.to(self.device)
        self.to(self.device)

    # def forward(self, x):
    #     # استخراج الخصائص من الـ Encoder المجمد
    #     features = self.cnn_backbone(x)
        
    #     # تطبيق الانتباه المكاني على آخر خريطة (أعمق واحدة)
    #     sa = self.sa(features[-1])
        
    #     # فك التشفير (Decoder)
    #     dec_out = sa
    #     # استبعاد الخريطة الأخيرة، وعكس الباقي لعمل مسارات التخطي
    #     skip_features = features[:-1][::-1] 
        
    #     for i, decoder in enumerate(self.decoders):
    #         dec_out = decoder(dec_out, skip_features[i])
            
    #     # المخرجات النهائية
    #     final_out = self.final_up(dec_out)
    #     final_out = self.final_conv(final_out)
        
    #     # return self.sigmoid(final_out)
    #     return final_out

    def forward(self, x):  

        input_size = x.shape[2:]
        cnn_feats = self.cnn_backbone(x)

        # trans_feats = self.trans_backbone(x)
        # hybrid_raw_features = [cnn_feats[0]]  # الطبقة الأولى تبقى CNN نقية
        # for i in range(4):
        #     # دمج مستقر عبر التركيبة المحدبة
        #     fused = self.fusion_blocks[i](cnn_x=cnn_feats[i+1], trans_x=trans_feats[i])
        #     hybrid_raw_features.append(fused)
        
        # 3. توحيد القنوات (Projections)
        features = []
        # for i, (proj, feat) in enumerate(zip(self.projections, cnn_feats)):
        for i, (proj, feat) in enumerate(zip(self.projections, hybrid_raw_features)):
            features.append(proj(feat))
        
        f_in = features[0]
        f_out = features[-1]
        f_ii = features[1:-1]

        # # 2. MSFF
        # enable_msff = getattr(self.config, 'enable_msff', True)
        # if enable_msff:
        #     msff_outputs = self.msff(features=f_ii)
        # else:
        #     msff_outputs = f_ii # تخطي عملية الدمج والـ Attention

        # 3. Decoder
        outputs = self.decoder(
            encoder_output=f_out,
            concat_features=[f_in] + f_ii
        )

        if outputs.shape[2:] != input_size:
            outputs = F.interpolate(
                outputs,
                size=input_size, 
                mode='bilinear',
                align_corners=True
            )
        
        return outputs
        # refined_features = self.cbam(outputs)
        
        # out = self.final_conv(refined_features)
        # return out

    def fit(self, train_loader, test_loader=None, save_dir=None):
        num_training_steps = self.config.num_epochs

        # optimizer = self.opts_list[self.config.opt_name](
        #     params       = filter(lambda p: p.requires_grad, self.parameters()),
        #     lr           = self.config.learning_rate,  
        #     weight_decay = self.config.weight_decay
        #     )

        # scheduler = CosineAnnealingWarmupRestarts(
        #         optimizer,
        #         first_cycle_steps = num_training_steps,
        #         max_lr = self.config.learning_rate,
        #         min_lr = self.config.min_lr,
        #         gamma= 1.0,
        #         warmup_steps   = int(num_training_steps * self.config.warmup_ratio)
        #         )

        # focal_criterion = FocalLoss(
        #     smooth= self.config.focal_smooth,
        #     gamma = self.config.focal_gamma,
        #     alpha = self.config.focal_alpha
        # )
        # pc_criterion = CompositeLoss()
        # spectral_criterion = SpectralLoss(loss_weight=self.config.spectral_weight)
        
        # composite_weight = self.config.composite_weight
        # focal_weight = self.config.focal_weight

        criterion = IoUOptimizedLoss(dice_weight=0.6, focal_weight=0.4).to(self.device)
        # criterion = EEMFNetLoss(focal_weight=0.6, dice_weight=0.4).to(self.device)
        
        ##############################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # تهيئة النموذج ومُحسِّن Adam
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        
        ######################################

        best_score = -1.0
        best_AP = -1.0
        best_epoch = 0

        logger.info(f"--> Starting Training for {num_training_steps} epochs...")
        train_mode = True
        epoch = 0
        
        # for epoch in range(num_training_steps):
        while train_mode:  
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            # logger.info(f"Epoch {epoch}: Difficulty Level {train_loader.dataset.difficulty_level:.2f}")    
            self.train() 
            # self.net.train()
            
            self.cnn_backbone.eval()
            # self.trans_backbone.eval()
            # self.cnn_backbone.layer4.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_training_steps}", leave=False)

            for raw_images, _, _,_ in pbar:
                
                if raw_images.shape[-1] == 3 or raw_images.shape[-1] == 1:
                    raw_images = raw_images.permute(0, 3, 1, 2).contiguous()

                # end = time.time()
                # images, masks, targets = images.to(self.device), masks.to(self.device), targets.to(self.device)
                raw_images = raw_images.to(self.device, non_blocking=True)
                
                with torch.no_grad():
                    images, masks, targets = self.augmenter(raw_images)

                if masks.dim() == 4:
                    masks = masks.squeeze(1)

                # optimizer.zero_grad(set_to_none=True)
                
                outputs = self(images)
                
                # if torch.cuda.is_available():
                #     images, masks = Variable(images.cuda(), requires_grad=False), Variable(masks.cuda(),
                #                                                                                 requires_grad=False)
                # else:
                #     images, masks = Variable(images, requires_grad=False), Variable(masks, requires_grad=False)

                # d0, d1, d2, d3, d4, d5, d6 = self.net(images)
                # loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, masks)

                # # loss_f = focal_criterion(outputs, masks)
                # # outputs = F.softmax(outputs, dim=1)
                # # if isinstance(outputs, (list, tuple)): outputs = outputs[0]
                # # masks = masks.unsqueeze(1).float() if masks.dim() == 3 else masks.float()
                # # # loss_c = pc_criterion(outputs[:, 1, :, :], masks)
                # # loss_c = pc_criterion(outputs[:, 1:2, :, :], masks)
                # # # loss_s = spectral_criterion(outputs[:, 1, :, :], masks)
                # # # loss_s = spectral_criterion(outputs[:, 1:2, :, :], masks)
                # # # loss =(composite_weight * loss_c) + (focal_weight * loss_f) + loss_s
                # # loss =(composite_weight * loss_c) + (focal_weight * loss_f)

                loss = criterion(outputs, masks)
                # # anomaly_preds = outputs[:, 1, :, :] # استخراج قناة الشذوذ
    
                # # loss = criterion(anomaly_preds, masks)
                # loss = dice_coefficient_loss(masks, outputs[:, 0, :, :])
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                

                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

                # del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            pbar.close()

            avg_loss = total_loss / len(train_loader)

            log_payload = {
                "train/epoch_loss": avg_loss,
                'lr': optimizer.param_groups[0]['lr']
            }
            
            if (test_loader and ((epoch+1) % self.config.val_interval == 0)) or (epoch==0):
                logger.info(f"\n[Epoch {epoch+1}] Validating...")
                eval_metrics, fps, optimal_threshold = self.predict(test_loader)
                log_payload.update({
                    "val/img_auc": eval_metrics['AUROC-image'],
                    "val/img_AP": eval_metrics['AP-image'],
                    "val/pixel_auc": eval_metrics['AUROC-pixel'],
                    "val/pixel_AP": eval_metrics['AP-pixel'],
                    "val/pro_score": eval_metrics['AUPRO-pixel'],
                    "val/pixel_f1": eval_metrics['F1-pixel'],
                    "val/pixel_IoU1": eval_metrics['IoU-pixel'],
                    "val/pixel_Dice": eval_metrics['Dice-pixel'],
                    "val/FPR@95TPR": eval_metrics['FPR@95TPR'],
                    "val/optimal_threshold": optimal_threshold
                })

                current_AP = eval_metrics["AP-pixel"]
                # current_score = eval_metrics["AUPRO-pixel"]

                current_score = np.mean([
                    eval_metrics['AUROC-pixel'],
                    eval_metrics['AUROC-image'],
                    eval_metrics["F1-pixel"],
                    eval_metrics["AUPRO-pixel"],
                    eval_metrics["AP-pixel"]
                    ])

                if best_score < current_score:
                    if  num_training_steps-epoch < 10:
                        num_training_steps = num_training_steps + 10
                    # best_score = np.mean(list(eval_metrics.values()))
                    best_score = current_score
                    best_epoch = epoch

                    if save_dir:
                        save_path = os.path.join(save_dir, "best_model.pth")                        
                        eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])

                        state = {
                                'best_epoch': best_epoch+1,
                                'inference_speed': f"{fps:.4f} s",
                                'optimal_threshold': f"{optimal_threshold:.4f}",
                                'metrics': eval_log
                            }

                        json.dump(state, open(os.path.join(save_dir, 'best_score.json'),'w'), indent='\t')

                        state_dict_cpu = {k: v.cpu() for k, v in self.state_dict().items()}
                        torch.save(state_dict_cpu, save_path)

                        logger.info(f"Epoch {epoch+1}:Img-AUC: {eval_metrics['AUROC-image']:.4f} | Px-AUC: {eval_metrics['AUROC-pixel']:.4f} | PRO: {eval_metrics['AUPRO-pixel']:.4f} | F1-Score: {eval_metrics['F1-pixel']:.4f} | Optimal-Threshold: {optimal_threshold:.4f} | inference speed: {fps:.4f} s")
                        logger.info(f"Epoch {epoch+1}: finished. Avg Loss: {avg_loss:.6f}")
                        logger.info(f"   >> New Best Model Saved!")

                    if self.config.use_wandb:
                        wandb.run.summary["best_img_auc"] = eval_metrics['AUROC-image']
                        wandb.run.summary["best_pixel_auc"] = eval_metrics['AUROC-pixel']
                        wandb.run.summary["best_aupro"] = eval_metrics['AUPRO-pixel']
                        wandb.run.summary["best_pixel_ap"] = eval_metrics['AP-pixel']
                        wandb.run.summary["best_IoU-pixel"] = eval_metrics['IoU-pixel']
                        wandb.run.summary["best_F1-pixel"] = eval_metrics['F1-pixel']
                        wandb.run.summary["best_epoch"] = epoch + 1
                else:
                    if best_AP < current_AP:
                        best_AP = current_AP
                        if  num_training_steps-epoch < 10:
                            num_training_steps = num_training_steps + 10

            else:
                logger.info(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.6f}")

            # if epoch < self.config.num_epochs:
            #     scheduler.step()

            if self.config.use_wandb:      
                wandb.log(log_payload, step=epoch+1)

            epoch += 1
            if epoch == num_training_steps:
                train_mode = False
                break

    def find_optimal_threshold(self, labels, scores):
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        a = 2 * precision[:-1] * recall[:-1]
        b = precision[:-1] + recall[:-1]
        f1_scores = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
       
        return best_threshold, best_f1


    def predict(self, test_loader):
        self.eval() 
        # self.net.eval()
        anomaly_maps = []
        image_scores = []
        gt_labels = []
        gt_masks = []

        with torch.no_grad():
            total_inference_time = 0.0
            total_images = 0
            for images, masks, labels, paths in tqdm(test_loader, desc="Testing"):
                images, masks, labels = images.to(self.device), masks.to(self.device), labels.to(self.device)
                
                start_t = time.time()    
                outputs = self(images)
                # pred = self(images)
                # d1,d2,d3,d4,d5,d6,d7= self.net(images)

                # pred = outputs[:,0,:,:]
                pred = outputs[:,1,:]
                anomaly_score_i = torch.topk(torch.flatten(pred, start_dim=1), 100)[0].mean(dim=1)
                
                image_scores.extend(anomaly_score_i.cpu())
                anomaly_maps.extend(pred.cpu().data.numpy())
                
                gt_labels.extend(labels.cpu().numpy())
                gt_masks.extend(masks.cpu().data.numpy())

                
                
                total_inference_time += (time.time() - start_t)
                total_images += images.size(0)
                # # outputs = F.softmax(outputs, dim=1)
                # anomaly_score_i = torch.topk(torch.flatten(outputs[:,1,:], start_dim=1), 100)[0].mean(dim=1)
                
                # image_scores.extend(anomaly_score_i.cpu())
                # anomaly_maps.extend(outputs[:,1,:].cpu().numpy())
                
                # gt_labels.extend(labels.cpu().numpy())
                # gt_masks.extend(masks.cpu().numpy())
        
        inference_speed = total_inference_time / total_images
        if len(anomaly_maps) > 0:
            anomaly_maps = np.array(anomaly_maps, dtype=np.float32)
        else:
            anomaly_maps = np.array([])

        image_scores = np.array(image_scores)
        gt_labels = np.array(gt_labels)
        gt_masks = np.array(gt_masks)
        
        flat_scores = anomaly_maps.flatten()
        flat_masks = gt_masks.flatten().astype(int)
        optimal_threshold, pixel_f1 = self.find_optimal_threshold(flat_masks, flat_scores)

        if len(np.unique(gt_labels)) > 1:
            img_auc = roc_auc_score(gt_labels, image_scores)
            img_ap = average_precision_score(gt_labels, image_scores)
        else:
            img_auc, img_ap = 0.5, 0.0
        if len(np.unique(flat_masks)) > 1:
            pixel_auc = roc_auc_score(flat_masks, flat_scores)
            pixel_ap = average_precision_score(flat_masks, flat_scores)
        else:
            pixel_auc, pixel_ap = 0.5, 0.0

        binary_preds = (anomaly_maps > optimal_threshold).astype(np.uint8)
        binary_preds = binary_preds.astype(bool)
        gt_masks_bool = gt_masks.astype(bool)
        intersection = np.logical_and(binary_preds, gt_masks_bool).sum(axis=(1, 2))
        union = np.logical_or(binary_preds, gt_masks_bool).sum(axis=(1, 2))
        ious = intersection / (union + 1e-6)
        dices = 2 * intersection / (binary_preds.sum(axis=(1, 2)) + gt_masks_bool.sum(axis=(1, 2)) + 1e-6)
        seg_iou = np.mean(ious)
        seg_dice = np.mean(dices)
        
        pro_score = self.evaluator.compute_pro_score(
            anomaly_maps, gt_masks, return_curve=False)
        
        if self.config.use_wandb:
            wandb.log({"Final_Anomaly_Map": [wandb.Image(anomaly_maps[-4], caption="Ablation Map Result")]})

        fpr, tpr, _ = roc_curve(flat_masks, flat_scores)
        fpr_95 = fpr[np.argmin(np.abs(tpr - 0.95))]

        metrics = {
            "AUROC-image": img_auc,
            "AP-image": img_ap,
            "AUROC-pixel": pixel_auc,
            "AP-pixel": pixel_ap,
            "AUPRO-pixel": pro_score,
            # "AUPRO-pixel": pro_score_value,
            "F1-pixel": pixel_f1,
            "IoU-pixel": seg_iou,
            "Dice-pixel": seg_dice,
            "FPR@95TPR": fpr_95
        }

        logger.info(
            f"[Img-AUC {img_auc:.4f} | Img-AP {img_ap:.4f} | "
            f"Px-AUC {pixel_auc:.4f} | Px-AP {pixel_ap:.4f} | "
            f"PRO {pro_score:.4f} | F1 {pixel_f1:.4f} | "
            f"IoU-pixel {seg_iou:.4f} | "
            f"Dice-pixel {seg_dice:.4f} | "
            f"FPR@95TPR {fpr_95:.4f} | "
            f"Inference {inference_speed:.4f}s]"
        )

        del anomaly_maps
        del image_scores
        del gt_labels
        del gt_masks
        del flat_scores
        del flat_masks
        del binary_preds
        del gt_masks_bool
        gc.collect()
        
        torch.cuda.empty_cache()
        
        return metrics, inference_speed, optimal_threshold
