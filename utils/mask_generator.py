import numpy as np
import cv2

class PerlinMaskGenerator:
    def __init__(self, min_scale=1, max_scale=4, octaves=3, persistence=0.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.octaves = octaves
        self.persistence = persistence

    def _smoothstep(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _generate_layer(self, shape, res):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        
        angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        
        g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:  , 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1, 1:  ].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:  , 1:  ].repeat(d[0], 0).repeat(d[1], 1)

        n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
        
        t = self._smoothstep(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        
        return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

    def _generate_fbm_noise(self, shape, res):
        noise_sum = np.zeros(shape, dtype=np.float32)
        amplitude = 1.0
        max_amplitude = 0.0

        for octave in range(self.octaves):
            current_res = (int(res[0] * (2**octave)), int(res[1] * (2**octave)))
            current_res = (max(1, min(current_res[0], shape[0])), max(1, min(current_res[1], shape[1])))
            
            layer = self._generate_layer(shape, current_res)
            noise_sum += layer * amplitude
            max_amplitude += amplitude
            amplitude *= self.persistence 
            
        return noise_sum / max_amplitude

    def generate_mask(self, height=256, width=256, anomaly_type=None) -> np.ndarray:
        
        if anomaly_type not in ['spot', 'scratch']:
            anomaly_type = np.random.choice(['spot', 'scratch'])

        if anomaly_type == 'spot':
            scale_x = 2 ** np.random.randint(self.min_scale, self.max_scale)
            scale_y = 2 ** np.random.randint(self.min_scale, self.max_scale)
        else:
            scale_x = 2 ** np.random.randint(1, 3) 
            scale_y = 2 ** np.random.randint(4, 7) 

        base_size = 256
        noise = self._generate_fbm_noise((base_size, base_size), (scale_x, scale_y))
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)

        rot = np.random.choice([0, 90, 180, 270])
        if rot > 0:
            noise = np.ascontiguousarray(np.rot90(noise, rot // 90))

        cy = np.random.randint(height // 4, 3 * height // 4)
        cx = np.random.randint(width // 4, 3 * width // 4)
        radius = np.random.randint(min(height, width) // 8, min(height, width) // 3)
        
        y, x = np.ogrid[:height, :width]
        envelope = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (radius**2)))
        localized_noise = noise * envelope

        threshold = np.max(localized_noise) * np.random.uniform(0.45, 0.6)
        binary_mask = np.where(localized_noise > threshold, 1.0, 0.0).astype(np.float32)

        mask_uint8 = (binary_mask * 255).astype(np.uint8)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
        
        ksize = int(radius / 4) | 1 
        ksize = max(3, min(ksize, 15))
        final_mask = cv2.GaussianBlur(mask_uint8, (ksize, ksize), 0)

        return (final_mask / 255.0).astype(np.float32)
    
class FreeFormMaskGenerator:
    """
    كلاس لتوليد أقنعة شذوذ عضوية (Free-Form Mask) باستخدام المضلعات المحدبة العشوائية.
    يسمح بالتبديل بين العيوب الكبيرة (Macro) والدقيقة (Micro).
    """
    def __init__(self):
        # يمكنك هنا إضافة أي معاملات افتراضية في المستقبل إذا احتجت
        pass

    def generate_mask(self, height=256, width=256, defect_scale=None) -> np.ndarray:
        """
        الدالة الرئيسية لتوليد قناع مضلع عشوائي.
        
        Parameters:
            height (int): ارتفاع القناع.
            width (int): عرض القناع.
            defect_scale (str): 'macro' أو 'micro' أو None للاختيار العشوائي.
        """
        mask = np.zeros((height, width), dtype=np.float32)
        ref_dim = min(height, width)
        center_x, center_y = width // 2, height // 2
        
        if defect_scale not in ['macro', 'micro']:
            defect_scale = np.random.choice(['macro', 'micro'])

        # 1. إعداد المعاملات بناءً على حجم العيب
        if defect_scale == 'macro':
            num_points = np.random.randint(5, 12)
            max_radius = ref_dim // 5
            radius = np.random.randint(max_radius // 2, max_radius)
            env_multiplier = 1.5  
            blur_ksize = int(ref_dim * 0.05) | 1  
        else:
            num_points = np.random.randint(4, 9)
            min_radius = max(3, int(ref_dim * 0.02))
            max_radius = max(8, int(ref_dim * 0.08))
            radius = np.random.randint(min_radius, max_radius)
            env_multiplier = 0.9  
            blur_ksize = max(3, int(radius * 0.4) | 1)  

        # 2. توليد نقاط المضلع العشوائية
        points = []
        for _ in range(num_points):
            angle = np.random.uniform(0, 2 * np.pi)
            
            if defect_scale == 'micro':
                r = np.random.uniform(radius * 0.3, radius)
            else:
                r = np.random.uniform(radius * 0.4, radius)
                
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            points.append([x, y])

        points = np.array(points, dtype=np.int32)
        
        # 3. رسم المضلع المحدب (Convex Hull)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 1.0)

        # 4. تطبيق تحويلات هندسية (دوران، تمدد، إزاحة)
        angle = np.random.uniform(-90, 90)
        scale = np.random.uniform(0.7, 1.3)
        tx = np.random.uniform(-ref_dim // 3, ref_dim // 3)
        ty = np.random.uniform(-ref_dim // 3, ref_dim // 3)

        M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        mask = cv2.warpAffine(mask, M, (width, height))

        # 5. تطبيق الغلاف الجاوسي (Gaussian Envelope)
        cx_env = int(np.clip(center_x + tx, 0, width - 1))
        cy_env = int(np.clip(center_y + ty, 0, height - 1))
        
        env_radius = radius * scale * env_multiplier
        y, x = np.ogrid[:height, :width]
        gaussian_envelope = np.exp(-((x - cx_env)**2 + (y - cy_env)**2) / (2 * (env_radius**2)))
        
        mask = mask * gaussian_envelope

        # 6. التنعيم النهائي والتطبيع
        mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)

        if np.max(mask) > 0:
            mask = mask / np.max(mask)

        return np.clip(mask, 0.0, 1.0).astype(np.float32)

class ScratchMaskGenerator:
    """
    كلاس لتوليد أقنعة الخدوش (Scratches) عبر إنشاء خطوط مقطعة ومتعرجة
    تحاكي خدوش الآلات الحادة على الأسطح.
    """
    def __init__(self, min_scratches=1, max_scratches=4):
        self.min_scratches = min_scratches
        self.max_scratches = max_scratches

    def generate_mask(self, height=256, width=256) -> np.ndarray:
        """
        الدالة الرئيسية لتوليد قناع الخدوش.
        """
        mask = np.zeros((height, width), dtype=np.float32)
        ref_dim = min(height, width)
        
        # اختيار عدد الخدوش العشوائي
        num_scratches = np.random.randint(self.min_scratches, self.max_scratches)
        
        for _ in range(num_scratches):
            margin = int(ref_dim * 0.1)
            x = np.random.randint(margin, width - margin)
            y = np.random.randint(margin, height - margin)
            
            current_angle = np.random.uniform(0, 2 * np.pi)
            num_segments = np.random.randint(4, 10)
            
            min_thick = max(1, int(ref_dim * 0.004))
            max_thick = max(2, int(ref_dim * 0.015))
            base_thickness = np.random.randint(min_thick, max_thick + 1)
            
            # رسم أجزاء الخدش (Segments)
            for _ in range(num_segments):
                min_len = ref_dim * 0.04
                max_len = ref_dim * 0.15
                length = np.random.uniform(min_len, max_len)
                
                # تغيير الزاوية بشكل طفيف لكل قطعة للحصول على تعرج طبيعي
                current_angle += np.random.uniform(-np.pi/6, np.pi/6)
                
                nx = int(x + length * np.cos(current_angle))
                ny = int(y + length * np.sin(current_angle))
                
                # تذبذب بسيط في السماكة
                thickness = max(1, base_thickness + np.random.randint(-1, 2))
                
                cv2.line(mask, (int(x), int(y)), (nx, ny), 1.0, thickness)
                x, y = nx, ny
                
        # التنعيم لدمج الخدش مع السطح بشكل واقعي
        k_small = max(3, int(ref_dim * 0.01) | 1)
        k_large = max(5, int(ref_dim * 0.02) | 1)
        ksize = int(np.random.choice([k_small, k_large]))
        
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0.8)
        
        return np.clip(mask, 0.0, 1.0).astype(np.float32)