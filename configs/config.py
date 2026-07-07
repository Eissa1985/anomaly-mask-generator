from dataclasses import dataclass, field
from typing import List

@dataclass
class ExperimentConfig:
    use_wandb: bool = False #True  
    use_synthetic: bool = True
    enable_msff: bool = True
    exp_name: str = "Full_Model"
    # Paths
    dataset_name : str = 'ydfid' # 'mvtec' or 'ydfid' or 'all'
    mvtec_path: str = "./datasets/mvtec_ad"
    ydfid_path : str = "./datasets/YDFID_1"
    aitex_path : str = "./datasets/AITEX"
    tfd_path : str = "./datasets/TFD"

    # MvTec Classes
    mvtec_classes: List[str] = field(default_factory=lambda: [
        "carpet",
        # "leather", "tile", "wood",
        # "grid",
        # "toothbrush",
        # "transistor",
        # "zipper",
        # "cable",
        # "bottle",
        # "capsule",
        # "hazelnut",
        # "metal_nut",
        # "pill",
        # "screw",
    ])

    # AITEX Classes
    aitex_classes: List[str] = field(default_factory=lambda: [
        "0",
        "1",
        "2",
        "3"
    ])

    # TFD Classes
    tfd_classes: List[str] = field(default_factory=lambda: [
        "001",
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
        "008",
        "009",
        "010"
    ])

    # YDFID
    ydfid_classes: List[str] = field(default_factory=lambda: [
        'SL1',
        'SL8', 'SL9', 'SL10', 'SL13', 'SL16',
        'CL1', 'CL2', 'CL3', 'CL4', 'CL10', 'CL12',
        'SP3', 'SP5', 
        'SP19',
        'SP24'
    ])

    texture_source_dir: str = "./datasets/dtd/images"
    # texture_source_dir: str = "./datasets/synthetic_textures/images"
    anomaly_source_path: str = "./datasets/anomaly_generation_datasets/images"
    save_dir: str = "./experiments"
    SEED: int = 42
    structure_grid_size: int = 8
    transparency_range: List[float] = field(default_factory=lambda: [0.40, 1.0])
    perlin_scale: int = 6
    min_perlin_scale: int = 0
    perlin_noise_threshold: float = 0.5

    # --- Training Hyperparameters  ---
    num_epochs: int = 100      
    val_interval: int = 1
    learning_rate: float = 1e-4 
    weight_decay: float = 1e-5
    #####
    focal_alpha: float = None
    focal_gamma: int = 2 #4
    focal_smooth: float = 1e-4
    ######
    spectral_weight=0.2
    ######
    composite_weight: float = 0.6
    focal_weight: float = 0.4
    ##### SCHEDULER
    min_lr: float = 1e-5
    warmup_ratio: float  = 0.1
    # Model
    model_name: str = "eemfnet"
    backbone_name: str = "efficientnet_b4" 
    opt_name: str = "adamw" # adam, lion
    backbones_list: List[str] = field(default_factory=lambda: [
        # "NoBackbone"
        # "resnet18",          
        # "resnet50",          
        # "wide_resnet50_2",    
        # "hrnet_w32",          
        "efficientnet_b4",   
        # "convnext_tiny",      
        # "convnext_base",        
        # # --- Transformers (Hierarchical) ---
        # "swin_tiny_patch4_window7_224",  
        # "swin_base_patch4_window7_224",  
        # "pvt_v2_b2",                    
        # "mit_b2"                        
    ])

    # Data params
    img_size: int = 224
    batch_size: int = 8  

    