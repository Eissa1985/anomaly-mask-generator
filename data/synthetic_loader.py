import os
import cv2
import numpy as np
from glob import glob
from einops import rearrange
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List, Tuple
import random
from PIL import Image
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F
import math
import logging
from skimage import exposure
from torchvision import tv_tensors

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


# class AnomalyTransplanter:
#     def __init__(self):
#         pass
    
#     def match_illumination(self, source_anomaly, target_bg, mask):
#         matched_anomaly = exposure.match_histograms(source_anomaly, target_bg, channel_axis=-1)
#         matched_anomaly = np.clip(matched_anomaly, 0, 255).astype(np.uint8)
        
#         matched_anomaly[mask == 0] = 0 
#         return matched_anomaly
    
#     def extract_anomaly_roi(self, a_img: np.ndarray, mask: np.ndarray):
#         # extract the mask region
#         if mask.ndim == 3:
#             y, x = np.where(mask[:, :, 0] > 0)
#         else:
#             y, x = np.where(mask > 0)
            
#         if len(y) == 0:
#             return None, None
            
#         y0, x0, y1, x1 = y.min(), x.min(), y.max(), x.max()
#         crop_img, crop_mask = a_img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
#         if crop_mask.ndim == 2:
#             crop_mask = crop_mask[..., np.newaxis]
#         if crop_img.ndim == 2:
#             crop_img = crop_img[..., np.newaxis]        
#         crop_img = crop_img * (crop_mask > 0)
#         return crop_img, crop_mask
    
#     def transplant_anomaly(self, a_img: np.ndarray, mask: np.ndarray, bg_img: np.ndarray):

#         H, W, C = bg_img.shape

#         result_img = np.zeros_like(bg_img)
#         final_mask = np.zeros((H, W), dtype=np.float32)
        
#         # extract the mask region
#         crop_anomaly, crop_mask = self.extract_anomaly_roi(a_img, mask)
        
#         if crop_mask.ndim == 3:
#             crop_mask = crop_mask[:, :, 0]
       
#         # checkLTPoint
#         Ha, Wa, _ = crop_anomaly.shape
#         place_h = random.choice(range(H - Ha))
#         place_w = random.choice(range(W - Wa))
        
#         m_sum = crop_anomaly > 0
#         m_sum = m_sum.sum()
#         result_img[
#             place_h : place_h + Ha,
#             place_w : place_w + Wa,
#         ] = crop_anomaly
#         fro_img = (np.ones((H, W, 3), np.uint8) * 255 )
#         bg_img[result_img > 0] = 0
#         fusion_sum = (result_img > 0) * (fro_img > 0)
#         fusion_sum = fusion_sum > 0
#         fusion_sum = fusion_sum.sum()
#         # is covered?
#         if fusion_sum == m_sum:
#             bg_img = bg_img + result_img
#             result_img = bg_img
#             final_mask[
#                 place_h : place_h + Ha,
#                 place_w : place_w + Wa,
#             ] = (
#                 crop_mask * 255
#             )
#             final_mask[final_mask > 0] = 255
#             fro_img = fro_img.sum(axis=2)
#             fro_img[fro_img > 0] = 127
#             fro_img[final_mask > 0] = 0
#             # fro_img += final_mask
#             fro_img = fro_img + final_mask
#             fro_img = np.stack([fro_img] * 3, axis=2)
#         return result_img, final_mask / 255.0
    
class EEMFNetDataset(Dataset):
    def __init__(
        self,
        datadir: str,
        anomaly_source_path:str=None,
        target: str=None,
        is_train: bool=True,
        resize: Tuple[int, int] = (224, 224),
        file_list: list = None,
        texture_source_dir: str = None,
        structure_grid_size: int = 8,
        transparency_range: List[float] = [0.35, 1.0],
        perlin_scale: int = 6,
        min_perlin_scale: int = 0,
        perlin_noise_threshold: float = 0.5,
        save_dir="./experiments"
    ):
   
        # self.transplanter = AnomalyTransplanter()
        
        self.folder_dict = {}
        self.save_dir = save_dir
        self.ablation_type = None
        self.structual_mvtec_set = ["toothbrush", "transistor", "zipper", "cable",
        "bottle", "capsule", "hazelnut", "metal_nut", "pill", "screw"]
        self.bg_reverses = {
            "toothbrush": True,
            "transistor": False,
            "zipper": False,
            "cable": True,
            "bottle": False,
            "capsule": False,
            "hazelnut": True,
            "metal_nut": True,
            "pill": True,
            "screw": False,
            "grid": None,
            "carpet": None,
            "leather": None,
            "tile": None,
            "wood": None,
            }
        self.bg_thresholds = {
            "toothbrush": 30,
            "transistor": 90,
            "zipper": 100,
            "cable": 150,
            "bottle": 250,
            "capsule": 120,
            "hazelnut": 50,
            "metal_nut": 40,
            "pill": 100,
            "screw": 110,
            "grid": None,
            "carpet": None,
            "leather": None,
            "tile": None,
            "wood": None,
            }
        self.count = 0
        self.target = target
        self.current_epoch = 0
        self.max_epoch = 100
        self.difficulty_level = 0.0
        # sythetic anomaly switch
        self.anomaly_switch = False
        self.datadir = datadir
        self.is_train = is_train
        self.resize = list(resize)
        self.imagesize = resize[0]

        if file_list is not None:
            self.file_list = file_list
        elif self.is_train:
            self.file_list = glob(os.path.join(self.datadir, target, 'train/*/*'))
        else:
            self.file_list = glob(os.path.join(self.datadir, target, 'test/*/*'))

        if anomaly_source_path is not None:
            self.f_list = glob(os.path.join(anomaly_source_path, '*/*/*'))

        if self.is_train:
            self.texture_source_file_list = glob(os.path.join(texture_source_dir, '*/*')) if texture_source_dir else []
            self.perlin_scale = perlin_scale
            self.min_perlin_scale = min_perlin_scale
            self.perlin_noise_threshold = perlin_noise_threshold
            self.structure_grid_size = structure_grid_size
            self.transparency_range = transparency_range

        self.transform_img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(self.imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(self.imagesize),
            transforms.ToTensor(),
        ])

        # [Index 0: Perlin, Index 1: Scratches, Index 2: External]
        # -----------------------------------------------------------
        self.type_counts = np.array([0, 0], dtype=np.float32)

        self.forced_weights = None
        self.saved_best_weights = None
        self.current_w = np.array([0.5, 0.5], dtype=np.float32)
        self.exploration_fail_count = 0
        self.patience_limit = 5  
        self.exploration_fail_count_g = 0
        self.patience_limit_g = 10

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        target_size = (int(self.resize[0]), int(self.resize[1])) if isinstance(self.resize, (tuple, list)) else self.resize
        try:
            img = Image.open(file_path).convert("RGB").resize(target_size)
            # img = cv2.imread(file_path)
            
        except OSError:
            new_idx = np.random.randint(0, len(self.file_list))
            return self.__getitem__(new_idx)
        
        img = np.array(img)
        
        if "YDFID" in file_path:
            target = 0 if 'defect-free' in file_path else 1
        else:
            target = 0 if 'good' in file_path else 1
        
        if target == 0:
            mask = np.zeros(target_size, dtype=np.float32)
        else:
            if "YDFID" in file_path:
                mask_path = file_path.replace('test', 'GroundTruth').replace('.png', '_mask.png')
            else:
                mask_path = file_path.replace('test', 'ground_truth').replace('.png', '_mask.png')

            if not os.path.exists(mask_path):
                 mask_path = mask_path.replace('_mask.png', '_mask.jpg')

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L').resize(self.resize) # convert L for grayscale
                if mask is None:
                    new_idx = np.random.randint(0, len(self.file_list))
                    return self.__getitem__(new_idx)
                mask = np.array(mask) / 255.0 
            else:
                mask = np.zeros(self.resize, dtype=np.float32)

        # if self.is_train:
        #     if np.random.rand() < 0.5:
        #         k = np.random.choice([1, 2, 3]) #  90, 180, 270 
        #         img = np.ascontiguousarray(np.rot90(img, k))
        #         mask = np.ascontiguousarray(np.rot90(mask, k))

        #     if self.anomaly_switch:
        #         img, aug_mask = self.generate_anomaly(img=img, texture_img_list=self.texture_source_file_list)

        #         if np.max(aug_mask) > 0:
        #             mask = aug_mask
        #             target = 1
        #         # self.anomaly_switch = False
        #     else:
        #         self.anomaly_switch = True
        if self.is_train:
            img_tensor = self.transform_mask(img.copy())
            if isinstance(mask, np.ndarray):
                mask = (mask > 0.5).astype(np.float32) 
            mask = self.transform_mask(mask).squeeze()

        else:

            img_tensor = self.transform_img(img.copy())
            if isinstance(mask, np.ndarray):
                mask = (mask > 0.5).astype(np.float32) 
            mask = self.transform_mask(mask).squeeze()

        return img_tensor, mask, target, file_path

    # def generate_anomaly(self, img: np.ndarray, texture_img_list: list = None) -> Tuple[np.ndarray, np.ndarray]:

    #     if self.ablation_type is not None:
    #         anomaly_type = self.ablation_type
    #     else:
            
    #         if self.forced_weights is not None:
    #             self.current_w = self.forced_weights
    #         else:
    #             self.current_w = self.get_current_inverse_weights()
            
    #         anomaly_type = np.random.choice([0, 1], p=self.current_w)
    #         self.type_counts[anomaly_type] += 1
            
    #     img_h, img_w = img.shape[:2]
        
    #     # (0: PNM, 1: PSM, 2: FFM)
    #     # mask_type
    #     # 0: Perlin Noise Masks (PNM)
    #     # 1: Polyline Scratch Masks (PSM)
    #     # 2: Free-Form Masks (FFM)
    #     mask_type = np.random.choice([0, 1, 2])
    #     if mask_type == 0:
    #         mask = self.generate_perlin_noise_mask(img_h, img_w)
    #     elif mask_type == 1:
    #         mask = self.generate_scratch_mask(img_h, img_w)
    #     else:
    #         mask = self.generate_free_form_mask(img_h, img_w)
            
    #     # ==========================================
    #     # 0: PSA: Procedural Synthetic Anomalies
    #     # 1: TSA: Transplanted Synthetic Anomalies
        
    #     anomaly_type = 1
    #     # 'PSA'
    #     if anomaly_type == 0:
    #         idx = np.random.choice(len(texture_img_list))
    #         a_img = self._texture_source(texture_img_path=texture_img_list[idx])
            
    #         a_img = self.transplanter.match_illumination(a_img, img, mask)
            
    #         img_float = img.astype(np.float32)
    #         mask_scale = 1.0 if mask.max() <= 1.0 else 1.0 / 255.0
    #         mask_norm = np.expand_dims(mask, axis=2).astype(np.float32) * mask_scale
    #         beta = np.random.uniform(*self.transparency_range)
    #         inner_fusion = (beta * a_img) + ((1.0 - beta) * img_float)
    #         a_img = (mask_norm * inner_fusion) + ((1.0 - mask_norm) * img_float)
    #         anomaly_img_uint8 = np.clip(a_img, 0, 255).astype(np.uint8)
        
    #     # 'TSA'
    #     else:
    #         idx = np.random.randint(len(self.f_list))
    #         file_path = self.f_list[idx]
    #         while self.target in file_path:
    #             idx = np.random.randint(len(self.f_list))
    #             file_path = self.f_list[idx]
                
    #         target_size = (int(self.resize[0]), int(self.resize[1])) if isinstance(self.resize, (tuple, list)) else self.resize
    #         anomaly_source_img = Image.open(file_path).convert("RGB").resize(target_size)
    #         a_img = np.array(anomaly_source_img)

    #         mask_path_dir = file_path.replace('images', 'masks')
    #         base_name, ext = os.path.splitext(mask_path_dir)
    #         final_mask_path = base_name + '_mask' + ext
    #         mask = Image.open(final_mask_path).convert("L").resize(target_size, resample=Image.NEAREST)
    #         mask = np.array(mask)
            
    #         a_img = self.transplanter.match_illumination(a_img, img, mask)
    #         a_img, mask = self.transplanter.transplant_anomaly(a_img, mask, img.copy())
    #         anomaly_img_uint8 = np.clip(a_img, 0, 255).astype(np.uint8)
            
    #     return anomaly_img_uint8, mask
    
    # def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     bg_threshold = self.bg_thresholds[self.target]
    #     bg_reverse = self.bg_reverses[self.target]
    #     _, mask = cv2.threshold(img_gray, bg_threshold, 255, cv2.THRESH_BINARY)
    #     if bg_reverse:
    #         return (mask / 255).astype(np.uint8)
    #     else:
    #         target_foreground_mask = cv2.bitwise_not(mask)
    #         return (target_foreground_mask / 255).astype(np.uint8)

    # def _texture_source(self, texture_img_path: str) -> np.ndarray:
    #     target_size = (int(self.resize[0]), int(self.resize[1])) if isinstance(self.resize, (tuple, list)) else self.resize
    #     texture_source_img = Image.open(texture_img_path).convert("RGB").resize(target_size)
    #     texture_source_img = np.array(texture_source_img)
    #     return texture_source_img.astype(np.float32)

    # def generate_scratch_mask(self, height, width):
    #     mask = np.zeros((height, width), dtype=np.float32)
    #     ref_dim = min(height, width)
    #     num_scratches = np.random.randint(1, 4)
    #     for _ in range(num_scratches):
    #         margin = int(ref_dim * 0.1)
    #         x = np.random.randint(margin, width - margin)
    #         y = np.random.randint(margin, height - margin)
    #         current_angle = np.random.uniform(0, 2 * np.pi)
    #         num_segments = np.random.randint(4, 10)
    #         min_thick = max(1, int(ref_dim * 0.004))
    #         max_thick = max(2, int(ref_dim * 0.015))
    #         base_thickness = np.random.randint(min_thick, max_thick + 1)
            
    #         for _ in range(num_segments):
    #             min_len = ref_dim * 0.04
    #             max_len = ref_dim * 0.15
    #             length = np.random.uniform(min_len, max_len)
    #             current_angle += np.random.uniform(-np.pi/6, np.pi/6)
    #             nx = int(x + length * np.cos(current_angle))
    #             ny = int(y + length * np.sin(current_angle))
    #             thickness = max(1, base_thickness + np.random.randint(-1, 2))
    #             cv2.line(mask, (int(x), int(y)), (nx, ny), 1.0, thickness)
    #             x, y = nx, ny
                
    #     k_small = max(3, int(ref_dim * 0.01) | 1)
    #     k_large = max(5, int(ref_dim * 0.02) | 1)
    #     ksize = int(np.random.choice([k_small, k_large]))
    #     mask = cv2.GaussianBlur(mask, (ksize, ksize), 0.8)
    #     mask = np.clip(mask, 0.0, 1.0)
        
    #     return mask

    # def _interpolant(self, t):
    #     """Smoothstep fade function for Perlin noise interpolation."""
    #     return t * t * t * (t * (t * 6 - 15) + 10)
    
    # def _generate_single_layer(self, shape, res):
    #     """Generates a structured 2D Perlin noise lattice."""
    #     delta = (res[0] / shape[0], res[1] / shape[1])
    #     d = (shape[0] // res[0], shape[1] // res[1])
    #     grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
        
    #     # Calculate Gradients
    #     angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    #     gradients = np.dstack((np.cos(angles), np.sin(angles)))
    #     g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    #     g10 = gradients[1:  , 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    #     g01 = gradients[0:-1, 1:  ].repeat(d[0], 0).repeat(d[1], 1)
    #     g11 = gradients[1:  , 1:  ].repeat(d[0], 0).repeat(d[1], 1)

    #     # Calculate Ramps
    #     n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    #     n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    #     n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    #     n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
        
    #     # Apply Interpolation
    #     t = self._interpolant(grid)
    #     n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    #     n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        
    #     return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

    # def generate_perlin_noise_2d(self, shape, res):
    #     noise_sum = np.zeros(shape, dtype=np.float32)
    #     amplitude = 1.0

    #     for octave in range(3):
    #         current_res = (res[0] * (2**octave), res[1] * (2**octave))
    #         layer = self._generate_single_layer(shape, current_res)
    #         noise_sum += layer * amplitude
    #         amplitude *= 0.5 
            
    #     return noise_sum / np.sum([0.5**i for i in range(3)])

    # def generate_perlin_noise_mask(self, height=256, width=256) -> np.ndarray:
    #     min_perlin_scale=1
    #     perlin_scale=4
        
    #     # 1. Stochastically select anomaly morphology (spot/corrosion or scratch)
    #     anomaly_type = np.random.choice(['spot', 'scratch'])
    #     if anomaly_type == 'spot':
    #         # Symmetrical scales for blob-like, organic cloud anomalies
    #         perlin_scale_x = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
    #         perlin_scale_y = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
    #     else:
    #         # Asymmetrical scales to stretch the noise into elongated scratch structures
    #         perlin_scale_x = 2 ** np.random.randint(1, 3) 
    #         perlin_scale_y = 2 ** np.random.randint(4, 7) 

    #     # Generate base 2D Perlin noise
    #     gen_shape = (256, 256)
    #     perlin_noise = self.generate_perlin_noise_2d(gen_shape, (perlin_scale_x, perlin_scale_y))
    #     perlin_noise = cv2.resize(perlin_noise, (width, height))

    #     # Apply stochastic spatial rotation
    #     rot = np.random.choice([0, 90, 180, 270])
    #     if rot > 0:
    #         perlin_noise = np.ascontiguousarray(np.rot90(perlin_noise, rot // 90))

    #     # =====================================================================
    #     # Spatial Localization: Apply a Gaussian Envelope
    #     # Prevents the anomaly from sprawling globally across the entire image
    #     # =====================================================================
    #     h, w = perlin_noise.shape
    #     cy = np.random.randint(h // 4, 3 * h // 4)
    #     cx = np.random.randint(w // 4, 3 * w // 4)
    #     radius = np.random.randint(min(h, w) // 8, min(h, w) // 3)
        
    #     y, x = np.ogrid[:h, :w]
    #     gaussian_envelope = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (radius**2)))
    #     perlin_noise = perlin_noise * gaussian_envelope
    #     # =====================================================================
    #     # Binarization: Dynamic Thresholding
    #     # Adapts to the maximum intensity of the localized noise
    #     # =====================================================================
    #     dynamic_threshold = np.max(perlin_noise) * 0.5 
    #     mask_noise = np.where(perlin_noise > dynamic_threshold, 1.0, 0.0).astype(np.float32)
    #     # =====================================================================
    #     # Morphological Refinement: Noise cleaning and gap bridging
    #     # =====================================================================
    #     mask_noise_uint8 = (mask_noise * 255).astype(np.uint8)
        
    #     # 1. Opening: Removes isolated pixel-dust (small artifacts)
    #     kernel_open = np.ones((3, 3), np.uint8)
    #     mask_noise_uint8 = cv2.morphologyEx(mask_noise_uint8, cv2.MORPH_OPEN, kernel_open)
        
    #     # 2. Closing: Bridges small internal gaps to form a continuous solid blob
    #     kernel_close = np.ones((5, 5), np.uint8)
    #     mask_noise_uint8 = cv2.morphologyEx(mask_noise_uint8, cv2.MORPH_CLOSE, kernel_close)
        
    #     ksize = int(radius / 5) | 1  # ضمان أن يكون الرقم فردياً
    #     ksize = max(3, min(ksize, 11)) # حصر التمويه بين 3 و 11
        
    #     final_mask = cv2.GaussianBlur(mask_noise_uint8, (ksize, ksize), 0)

    #     return (final_mask / 255.0).astype(np.float32)
        
    # def generate_free_form_mask(self, height=256, width=256):
    #     """
    #     Generates a procedural anomaly mask (Free-Form Mask) using random convex polygons.
    #     Dynamically switches between Macro-defects (large/fuzzy) and Micro-defects (small/precise).
    #     """
    #     mask = np.zeros((height, width), dtype=np.float32)
    #     ref_dim = min(height, width)
    #     center_x, center_y = width // 2, height // 2
    #     defect_scale = np.random.choice(['macro', 'micro'])

    #     if defect_scale == 'macro':
    #         num_points = np.random.randint(5, 12)
    #         max_radius = ref_dim // 5
    #         radius = np.random.randint(max_radius // 2, max_radius)
            
    #         env_multiplier = 1.5  
    #         blur_ksize = int(ref_dim * 0.05) | 1  
            
    #     else:
    #         num_points = np.random.randint(4, 9)
    #         min_radius = max(3, int(ref_dim * 0.02))
    #         max_radius = max(8, int(ref_dim * 0.08))
    #         radius = np.random.randint(min_radius, max_radius)
            
    #         env_multiplier = 0.9  
    #         blur_ksize = max(3, int(radius * 0.4) | 1)  

    #     points = []
    #     for _ in range(num_points):
    #         angle = np.random.uniform(0, 2 * np.pi)
            
    #         if defect_scale == 'micro':
    #             r = np.random.uniform(radius * 0.3, radius)
    #         else:
    #             r = np.random.uniform(radius * 0.4, radius)
                
    #         x = int(center_x + r * np.cos(angle))
    #         y = int(center_y + r * np.sin(angle))
    #         points.append([x, y])

    #     points = np.array(points, dtype=np.int32)
    #     hull = cv2.convexHull(points)
    #     cv2.fillConvexPoly(mask, hull, 1.0)

    #     angle = np.random.uniform(-90, 90)
    #     scale = np.random.uniform(0.7, 1.3)
    #     tx = np.random.uniform(-ref_dim // 3, ref_dim // 3)
    #     ty = np.random.uniform(-ref_dim // 3, ref_dim // 3)

    #     M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
    #     M[0, 2] += tx
    #     M[1, 2] += ty
    #     mask = cv2.warpAffine(mask, M, (width, height))

    #     cx_env = int(np.clip(center_x + tx, 0, width - 1))
    #     cy_env = int(np.clip(center_y + ty, 0, height - 1))
        
    #     env_radius = radius * scale * env_multiplier
    #     y, x = np.ogrid[:height, :width]
    #     gaussian_envelope = np.exp(-((x - cx_env)**2 + (y - cy_env)**2) / (2 * (env_radius**2)))
        
    #     mask = mask * gaussian_envelope

    #     mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)

    #     if np.max(mask) > 0:
    #         mask = mask / np.max(mask)

    #     return np.clip(mask, 0.0, 1.0).astype(np.float32)

    def __len__(self):
        return len(self.file_list)
    
