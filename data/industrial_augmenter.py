import os
import random
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
import torchvision.transforms.functional as TF

class OnGPUIndustrialAugmenter(nn.Module):
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
        print(sampled_paths)
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