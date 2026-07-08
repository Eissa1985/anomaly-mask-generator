from utils.mask_generator import PerlinMaskGenerator, FreeFormMaskGenerator, ScratchMaskGenerator
import os
import cv2
import numpy as np
from tqdm.notebook import tqdm # استخدام tqdm الخاصة بالجوبيتر
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# ⚙️ إعدادات التوليد (يمكنك تعديلها)
# ==========================================
NUM_MASKS_PER_TYPE = 1000            # عدد الأقنعة المراد توليدها لكل نوع (الإجمالي سيكون 3000)
BASE_OUTPUT_DIR = "datasets/masks"    # المجلد الرئيسي
IMG_HEIGHT = 224                     # ارتفاع القناع
IMG_WIDTH = 224                      # عرض القناع
NUM_WORKERS = os.cpu_count()         # استخدام جميع أنوية المعالج

# تعريف الأنواع والمولدات الخاصة بها
MASK_TYPES = {
    'perlin': (PerlinMaskGenerator, 1),   # (الكلاس, رقم تعريفي للبذرة العشوائية)
    'freeform': (FreeFormMaskGenerator, 2),
    'scratch': (ScratchMaskGenerator, 3)
}

def create_and_save_mask(args):
    """
    دالة تقوم بتوليد قناع واحد وحفظه.
    تستقبل args كـ (فهرس الصورة, اسم نوع القناع)
    """
    index, mask_type = args
    GeneratorClass, type_id = MASK_TYPES[mask_type]
    
    # 1. ضمان عشوائية مستقلة لكل صورة وكل نوع
    np.random.seed((os.getpid() * (int(index) + 1) * type_id) % 123456789)
    
    # 2. تهيئة المولد المناسب بناءً على النوع
    if mask_type == 'perlin':
        generator = GeneratorClass(min_scale=1, max_scale=4, octaves=3)
    else:
        generator = GeneratorClass() # المولدات الأخرى لا تحتاج مدخلات إجبارية حالياً
    
    # 3. توليد القناع
    mask = generator.generate_mask(height=IMG_HEIGHT, width=IMG_WIDTH)
    attempts = 0
    max_attempts = 10
    
    # حلقة تكرارية: طالما القناع أسود (أقصى قيمة فيه تساوي صفر)، أعد التوليد
    while np.max(mask) == 0 and attempts < max_attempts:
        attempts += 1
        mask = generator.generate_mask(height=IMG_HEIGHT, width=IMG_WIDTH)
    
    # 4. تحويل القيم وحفظ الصورة
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # مسار الحفظ (مثال: dataset/masks/perlin/mask_000000.png)
    output_dir = os.path.join(BASE_OUTPUT_DIR, mask_type)
    filename = os.path.join(output_dir, f"{mask_type}_{index:06d}.png")
    cv2.imwrite(filename, mask_uint8)
    
    return True

if __name__ == '__main__':
    print("="*50)
    print(f"🚀 بدء التوليد الشامل (All-in-One Offline Generation)")
    print(f"📁 المجلد الرئيسي: {BASE_OUTPUT_DIR}")
    print(f"🔢 الإجمالي: {NUM_MASKS_PER_TYPE * len(MASK_TYPES)} قناع ({NUM_MASKS_PER_TYPE} لكل نوع)")
    print(f"💻 عدد مسارات المعالجة (Threads): {NUM_WORKERS}")
    print("="*50)

    # تجهيز قائمة المهام (Tasks) وإنشاء المجلدات الفرعية
    tasks = []
    for mask_type in MASK_TYPES.keys():
        os.makedirs(os.path.join(BASE_OUTPUT_DIR, mask_type), exist_ok=True)
        for i in range(NUM_MASKS_PER_TYPE):
            tasks.append((i, mask_type))

    # تنفيذ المهام بالتوازي
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # استخدام إجمالي عدد المهام لشريط التقدم
        list(tqdm(executor.map(create_and_save_mask, tasks), total=len(tasks), desc="توليد الأقنعة"))

    print("\n✅ اكتمل التوليد بنجاح! تم حفظ جميع الأنواع في مجلداتها المستقلة.")
