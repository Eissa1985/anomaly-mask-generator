from .decoder import Decoder
from .msff import MSFF
from utils.metrics import AnomalyEvaluator
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import CompositeLoss, FocalLoss, SpectralLoss
import torch.optim as optim
from tqdm import tqdm
import time
import os
import numpy as np
import gc

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

warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')

logger = logging.getLogger(__name__)

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
    
class IndustrialAugmenter(nn.Module):
    def __init__(self, masks_root_dir, img_size=224, p_anomaly=0.5, p_blur=0.3, p_illum=0.4):
        """
        masks_root_dir: المسار الأساسي لمجلدات الأقنعة (مثلاً: "dataset/masks")
        يجب أن يحتوي على المجلدات الفرعية: freeform, perlin, scratch
        """
        super().__init__()
        self.img_size = img_size
        self.p_anomaly = p_anomaly
        self.p_blur = p_blur
        self.p_illum = p_illum

        # 1. فهرسة كافة الأقنعة الجاهزة في الذاكرة لتسريع الوصول (Zero I/O Bottleneck)
        self.mask_paths = []
        for sub_dir in ["freeform", "perlin", "scratch"]:
            full_dir = os.path.join(masks_root_dir, sub_dir)
            if os.path.exists(full_dir):
                # قراءة كل صيغ الصور المحتملة
                paths = glob(os.path.join(full_dir, "*.*"))
                self.mask_paths.extend([p for p in paths if p.endswith(('.png', '.jpg', '.bmp', '.tif'))])
        
        if len(self.mask_paths) == 0:
            raise RuntimeError(f"لم يتم العثور على أي أقنعة في المسار: {masks_root_dir}. تأكد من وجود المجلدات الفرعية!")
            
        print(f"--> [GPU Augmenter] Successfully loaded {len(self.mask_paths)} offline mask paths.")

    def _sample_offline_masks(self, batch_size, device):
        """
        سحب أقنعة عشوائية من الـ 3000 قناع وتحميلها كـ Tensors على الـ GPU بسرعة فائقة
        """
        sampled_paths = random.choices(self.mask_paths, k=batch_size)
        mask_tensors = []
        
        for path in sampled_paths:
            # قراءة سريعة على قناة واحدة (Grayscale)
            with Image.open(path) as img:
                img_gray = img.convert("L")
                # إعادة تحجيم سريع إذا كان مقاس القناع الجاهز يختلف عن img_size
                if img_gray.size != (self.img_size, self.img_size):
                    img_gray = img_gray.resize((self.img_size, self.img_size), Image.NEAREST)
                
                # تحويل إلى Tensor وتطبيع من [0, 255] إلى [0.0, 1.0]
                t_mask = TF.to_tensor(img_gray)
                mask_tensors.append(t_mask)
        
        # تجميع الدفعة بالكامل ونقلها للـ GPU دفعة واحدة
        batch_masks = torch.stack(mask_tensors, dim=0).to(device, non_blocking=True)
        
        # التأكد من الثنائية (Binarization) لتجنب قيم الاستيفاء الرمادية
        return torch.where(batch_masks > 0.5, 1.0, 0.0)

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
        """
        images: صور الأقمشة النظيفة من الـ DataLoader بأبعاد (B, C, H, W)
        anomaly_textures: (اختياري) خامات عيوب خارجية لنقلها داخل القناع
        """
        batch_size = images.shape[0]
        device = images.device
        
        # 1. تطبيق تأثيرات البيئة الصناعية (ظلال واهتزاز) على الصورة النظيفة أولاً
        images = self._apply_illumination_gradient(images)
        images = self._apply_motion_blur(images)
        
        # 2. تحديد الصور التي سيتم حقن العيوب بها (50% من الدفعة مثلاً)
        inject_mask = (torch.rand(batch_size, device=device) < self.p_anomaly).float().view(batch_size, 1, 1, 1)
        
        if inject_mask.sum() == 0:
            zeros_mask = torch.zeros((batch_size, 1, self.img_size, self.img_size), device=device)
            return images, zeros_mask, torch.zeros(batch_size, device=device, dtype=torch.long)
            
        # 3. سحب الأقنعة الجاهزة (Offline Masks) من الـ 3000 قناع وتطبيق قناع الحقن عليها
        fault_masks = self._sample_offline_masks(batch_size, device) * inject_mask
        
        # 4. حقن الشذوذ الفعلي داخل القماش
        if anomaly_textures is not None and anomaly_textures.shape[0] == batch_size:
            # دمج نسيج خارجي (مثل صورة صدأ، تمزق، أو خيط لون مختلف) في منطقة القناع
            corrupted_images = images * (1 - fault_masks) + anomaly_textures * fault_masks
        else:
            # طريقة ذكية: إذا لم تمرر نسيج خارجي، نقوم بإنشاء "تغير تبايني/لوني شاذ" 
            # في نفس النسيج الأصلي لمحاكاة بقع الزيت، بلل القماش، أو الاحتراق الطفيف
            color_shift = torch.randn((batch_size, images.shape[1], 1, 1), device=device) * 0.4
            intensity_factor = torch.empty((batch_size, 1, 1, 1), device=device).uniform_(0.3, 1.7)
            
            modified_fabric = torch.clamp((images * intensity_factor) + color_shift, 0.0, 1.0)
            corrupted_images = images * (1 - fault_masks) + modified_fabric * fault_masks
            
        # 5. إعداد التسميات (Labels): 0 للنسيج السليم، 1 إذا احتوت الصورة على بكسل واحد معيب على الأقل
        targets = (fault_masks.view(batch_size, -1).max(dim=1)[0] > 0).long()
        
        return corrupted_images, fault_masks, targets
    
class ChannelProjector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelProjector, self).__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            # استخدام GroupNorm بدلاً من BatchNorm لاستقرار التدرجات مع الباتشات الصغيرة
            nn.GroupNorm(num_groups=8, num_channels=out_channels), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.projector(x)
        
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

class EEMFNet(nn.Module):
    def __init__(self, device='cpu', config=None):
        super(EEMFNet, self).__init__()
        self.opts_list = {
            "adamw": optim.AdamW,
            "adam": optim.Adam,
            "lion": Lion,
            }

        self.device = device
        self.config = config
        backbone_name = config.backbone_name if config else "resnet18"
        logger.info(f"--> Building Backbone: {backbone_name}")

        try:
            self.cnn_backbone = create_model(
                backbone_name,
                pretrained=True,
                features_only=True
            )
            # for p in self.cnn_backbone.parameters():
            #         p.requires_grad = False
            for name, param in self.cnn_backbone.named_parameters():
                if "blocks.4" in name or "blocks.5" in name or "conv_head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        except RuntimeError as e:
            logger.warning(f"Error.... Default indices failed for {backbone_name}, trying default behavior. c: {e}")

        cnn_channels = self.cnn_backbone.feature_info.channels()
        
        self.augmenter = IndustrialAugmenter(
            masks_root_dir="dataset/masks",  # ضع هنا المسار الدقيق لمجلد masks لديك
            img_size=self.config.img_size,           # 224 أو 256 أو حسب إعداداتك
            p_anomaly=0.5,                   # احتمال حقن عيب في الصورة
            p_blur=0.3,                      # احتمال اهتزاز حزام ماكينة النسيج
            p_illum=0.4                      # احتمال تغير الظلال والإضاءة
        ).to(device)
        
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

        # for p in self.trans_backbone.parameters():
        #     p.requires_grad = True

        # self.fusion_blocks = nn.ModuleList([
        #     DoubleConv(mit_dims[0] + cnn_channels[1], cnn_channels[1]), 
        #     DoubleConv(mit_dims[1] + cnn_channels[2], cnn_channels[2]),  
        #     DoubleConv(mit_dims[2] + cnn_channels[3], cnn_channels[3]),  
        #     DoubleConv(mit_dims[3] + cnn_channels[4], cnn_channels[4]),  
        #     DoubleConv(cnn_channels[4], mit_dims[3])                 
        # ])

        # self.upsampling = nn.ModuleList([
        #     nn.Sequential(DoubleConv(mit_dims[0], 32), nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)),
        #     nn.Sequential(DoubleConv(mit_dims[1], 32), nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)),
        #     nn.Sequential(DoubleConv(mit_dims[2], 32), nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)),
        #     nn.Sequential(DoubleConv(mit_dims[3], 32), nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)),
        #     nn.Sequential(DoubleConv(mit_dims[3], 32), nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True))
        # ])

        self.base_dim = 48 # 64
        self.target_channels = [self.base_dim * (2 ** i) for i in range(len(cnn_channels))]

        logger.info(f"Raw Channels: {cnn_channels}")
        logger.info(f"Projected Target Channels: {self.target_channels}")

        self.projections = nn.ModuleList([
            ChannelProjector(src, tgt)
            for src, tgt in zip(cnn_channels, self.target_channels)
        ])

        # self.msff = MSFF(in_channels[1:-1]).to(self.device)
        self.msff = MSFF(self.target_channels[1:-1]).to(self.device)
        # self.decoder = Decoder(in_channels).to(self.device)
        self.decoder = Decoder(self.target_channels).to(self.device)
        self.evaluator = AnomalyEvaluator(pro_integration_limit=0.3)
        self.cnn_backbone.to(self.device)
        self.to(self.device)
        
    def forward(self, x):  

        input_size = x.shape[2:]
        cnn_feats = self.cnn_backbone(x)
        # trans_feats = self.trans_backbone(x)
        # hybrid_raw_features = [cnn_feats[0]]  
        # for i in range(4):
        #     fused = torch.cat((trans_feats[i], cnn_feats[i+1]), dim=1)
        #     reduced = self.fusion_blocks[i](fused)
        #     hybrid_raw_features.append(reduced)

        features = []
        for i, (proj, feat) in enumerate(zip(self.projections, cnn_feats)):
        # for i, (proj, feat) in enumerate(zip(self.projections, hybrid_raw_features)):
            if cnn_feats[0].shape[1] == cnn_feats[0].shape[2] and cnn_feats[1].shape[1] == cnn_feats[1].shape[2]:
                feat = feat.permute(0, 3, 1, 2).contiguous()

            features.append(proj(feat))
        f_in = features[0]
        f_out = features[-1]
        f_ii = features[1:-1]

        # 2. MSFF
        enable_msff = getattr(self.config, 'enable_msff', True)
        if enable_msff:
            msff_outputs = self.msff(features=f_ii)
        else:
            msff_outputs = f_ii # تخطي عملية الدمج والـ Attention

        # 3. Decoder
        outputs = self.decoder(
            encoder_output=f_out,
            concat_features=[f_in] + msff_outputs
        )

        if outputs.shape[2:] != input_size:
            outputs = F.interpolate(
                outputs,
                size=input_size, 
                mode='bilinear',
                align_corners=True
            )

        return outputs

    def fit(self, train_loader, test_loader=None, save_dir=None):
        num_training_steps = self.config.num_epochs

        optimizer = self.opts_list[self.config.opt_name](
            params       = filter(lambda p: p.requires_grad, self.parameters()),
            lr           = self.config.learning_rate,  
            weight_decay = self.config.weight_decay
            )

        scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps = num_training_steps,
                max_lr = self.config.learning_rate,
                min_lr = self.config.min_lr,
                gamma= 1.0,
                warmup_steps   = int(num_training_steps * self.config.warmup_ratio)
                )

        focal_criterion = FocalLoss(
            smooth= self.config.focal_smooth,
            gamma = self.config.focal_gamma,
            alpha = self.config.focal_alpha
        )
        pc_criterion = CompositeLoss()
        spectral_criterion = SpectralLoss(loss_weight=self.config.spectral_weight)
        
        composite_weight = self.config.composite_weight
        focal_weight = self.config.focal_weight
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
            self.cnn_backbone.eval()
            # self.cnn_backbone.layer4.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_training_steps}", leave=False)

            for raw_images, _, _,_ in pbar:
                # end = time.time()
                # images, masks, targets = images.to(self.device), masks.to(self.device), targets.to(self.device)
                raw_images = raw_images.to(self.device, non_blocking=True)
                
                with torch.no_grad():
                    images, masks, targets = self.augmenter(raw_images)

                if masks.dim() == 4:
                    masks = masks.squeeze(1)

                optimizer.zero_grad(set_to_none=True)
                
                outputs = self(images)

                loss_f = focal_criterion(outputs, masks)
                # outputs = F.softmax(outputs, dim=1)

                if isinstance(outputs, (list, tuple)): outputs = outputs[0]

                masks = masks.float()
                loss_c = pc_criterion(outputs[:, 1, :, :], masks)
                # loss_c = pc_criterion(outputs[:, 1:2, :, :], masks)
                loss_s = spectral_criterion(outputs[:, 1, :, :], masks)
                # loss_s = spectral_criterion(outputs[:, 1:2, :, :], masks)
                loss =(composite_weight * loss_c) + (focal_weight * loss_f) + loss_s

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

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

            if epoch < self.config.num_epochs:
                scheduler.step()

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
                total_inference_time += (time.time() - start_t)
                total_images += images.size(0)
                outputs = F.softmax(outputs, dim=1)
                anomaly_score_i = torch.topk(torch.flatten(outputs[:,1,:], start_dim=1), 100)[0].mean(dim=1)
                
                image_scores.extend(anomaly_score_i.cpu())
                anomaly_maps.extend(outputs[:,1,:].cpu().numpy())
                
                gt_labels.extend(labels.cpu().numpy())
                gt_masks.extend(masks.cpu().numpy())
        
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
