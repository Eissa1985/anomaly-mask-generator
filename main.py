import os
import sys
import torch
import logging
import numpy as np
import random
from datetime import datetime
import argparse
from glob import glob
from configs.config import ExperimentConfig
from models import get_model
import wandb
import gc
from data.synthetic_loader import EEMFNetDataset
from torch.utils.data import DataLoader, Subset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
        
    def str2bool(v):
        return str(v).lower() in ("true", "1", "yes")

    parser = argparse.ArgumentParser(description="PhD Anomaly Detection Project")
    parser.add_argument("--dataset_name", type=str, default=None, help="mvtec, ydfid, all")
    parser.add_argument("--model", type=str, default=None, help="اسم الموديل: patchcore, eemfnet")
    parser.add_argument("--backbone_name", type=str, default=None, help="الباكبون: resnet18, wide_resnet50_2")
    parser.add_argument("--epochs", type=int, default=None, help="عدد الحقب للتدريب")
    parser.add_argument("--batch_size", type=int, default=None, help="حجم الدفعة")
    parser.add_argument("--opt_name", type=str, default=None, help="adam, adamw, loin")
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    
    set_seed(cfg.SEED) 

    if args.dataset_name: cfg.dataset_name = args.dataset_name
    if args.model: cfg.model_name = args.model
    if args.backbone_name: cfg.backbone_name = args.backbone_name
    if args.epochs: cfg.num_epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.opt_name: cfg.opt_name = args.opt_name
    if args.use_wandb: cfg.use_wandb = args.use_wandb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cfg.backbone_name in cfg.backbones_list:
        logger.info(f"========== Training with Backbone: {cfg.backbone_name} ==========")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        exp_name = f"{cfg.model_name}_{cfg.backbone_name}_{cfg.opt_name}_{timestamp}"
        exp_dir = os.path.join(cfg.save_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        file_handler = logging.FileHandler(os.path.join(exp_dir, "experiment.log"))
        # logger.addHandler(file_handler)
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Experiment started. Results: {exp_dir}")
        logger.info(f"Device: {device} | Model: {cfg.model_name} | Backbone: {cfg.backbone_name}")

        logger.info("Loading Feature Backbone Extractor...")

        execution_list = []
        if cfg.dataset_name == 'mvtec':
            execution_list = [{'target': cls, 'type': 'mvtec', 'root': cfg.mvtec_path} for cls in cfg.mvtec_classes]
            logger.info("--> Loading MVTec AD Dataset...")

        elif cfg.dataset_name == 'ydfid':
            execution_list = [{'target': cls, 'type': 'ydfid', 'root': cfg.ydfid_path} for cls in cfg.ydfid_classes]
            logger.info("--> Loading YDFID Dataset...")

        elif cfg.dataset_name == 'aitex':
            execution_list = [{'target': cls, 'type': 'aitex', 'root': cfg.aitex_path} for cls in cfg.aitex_classes]
            logger.info("--> Loading AITEX Dataset...")

        elif cfg.dataset_name == 'tfd':
            execution_list = [{'target': cls, 'type': 'tfd', 'root': cfg.tfd_path} for cls in cfg.tfd_classes]
            logger.info("--> Loading TFD Dataset...")

        elif cfg.dataset_name == 'all':
            for cls in cfg.mvtec_classes:
                execution_list.append({'target': cls, 'type': 'mvtec', 'root': cfg.mvtec_path})

            for cls in cfg.ydfid_classes:
                execution_list.append({'target': cls, 'type': 'ydfid', 'root': cfg.ydfid_path})

            for cls in cfg.aitex_classes:
                execution_list.append({'target': cls, 'type': 'aitex', 'root': cfg.aitex_path})

            for cls in cfg.tfd_classes:
                execution_list.append({'target': cls, 'type': 'tfd', 'root': cfg.tfd_path})

            logger.info(f"--> Loading all Dataset..., Total classes to train: {len(execution_list)}")


        scenarios = [
        # {"id": 0, "name": "PSA"},
        # {"id": 1, "name": "TSA"},
        {"id": None, "name": "All"}
    ]

        ablation_group_id = f"Ablation_Study_{wandb.util.generate_id()}"

        for sc in scenarios:
            use_synthetic = True
            enable_msff = True
            cfg.use_synthetic = use_synthetic
            cfg.enable_msff = enable_msff
            # cfg.use_wandb = False

            # for target in cfg.target_names:
            for task in execution_list:
                target = task['target']
                current_root = task['root']

                logger.info(f"\n{'='*20} Processing Class: {target} {'='*20}")
                class_dir = os.path.join(exp_dir, f"{target}")
                os.makedirs(class_dir, exist_ok=True)

                if cfg.use_wandb:
                    run = wandb.init(
                        project="EEMFNet_Ablation_V3",
                        group=ablation_group_id,
                        job_type="ablation_test",
                        name=f"{target}-{sc['name']}",
                        config={
                            "Folder": exp_dir,
                            "Target": target,
                            "scenario": sc['name'],
                            "use_msff": enable_msff,
                            "use_synthetic": use_synthetic,
                            "backbone": cfg.backbone_name
                        },
                        reinit=True
                    )

                print(f"--- Running: {target}-{sc['name']} ---")

                if cfg.use_synthetic:
                    train_dataset = EEMFNetDataset(
                        datadir=current_root,
                        anomaly_source_path=cfg.anomaly_source_path,
                        target=target,
                        is_train=True,
                        resize=(cfg.img_size, cfg.img_size),
                        texture_source_dir=cfg.texture_source_dir,
                        structure_grid_size=cfg.structure_grid_size,
                        transparency_range=cfg.transparency_range,
                        perlin_scale=cfg.perlin_scale,
                        min_perlin_scale=cfg.min_perlin_scale,
                        save_dir=exp_dir
                    )

                    test_dataset = EEMFNetDataset(
                        datadir=current_root,
                        target=target,
                        is_train=False,
                        resize=(cfg.img_size, cfg.img_size),
                    )

                if len(train_dataset) == 0:
                        logger.warning(f"No data found for {target}. Skipping...")
                        if cfg.use_wandb: wandb.finish()
                        continue

                num_workers = 4 if os.name != 'nt' else 0
                # num_workers = 0

                train_dataset.ablation_type = sc['id']

                print(os.path.join(current_root, target, r'test/*/*'))
                all_files = glob(os.path.join(current_root, target, r'test/*/*.png'), recursive=True)
                file_list_test = [f for f in all_files if os.path.isfile(f) and not os.path.basename(f).startswith('.')]
                tmp_labels = [0 if 'good' in file_name or 'defect-free' in file_name  else 1 for file_name in file_list_test]  

                # Separate indices for normal and abnormal images
                normal_indices = [i for i, label in enumerate(tmp_labels) if label == 0]
                abnormal_indices = [i for i, label in enumerate(tmp_labels) if label == 1]

                # Count the number of normal and abnormal images
                num_normal = len(normal_indices)
                num_abnormal = len(abnormal_indices)
                print("...............len file_list test", len(file_list_test))
                print("...............len num_normal test", num_normal)
                print("...............len num_abnormal test", num_abnormal)

                if num_normal > num_abnormal:
                    selected_normal = num_abnormal
                    normal_indices = random.sample(normal_indices, min(selected_normal, num_normal))
                    selected_indices = normal_indices + abnormal_indices

                else:
                    selected_indices = normal_indices + abnormal_indices

                # shuffle the selected indices
                np.random.shuffle(selected_indices)

                # Create a subset of the dataset
                test_dataset = Subset(test_dataset, selected_indices)
                # print("...............len validation_subloader", len(sub_dataset))

                train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                          num_workers=num_workers, 
                                          persistent_workers=True, 
                                          pin_memory=True
                                          )
                test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                                         num_workers=num_workers, 
                                         persistent_workers=True, 
                                         pin_memory=True
                                         )        

                model = get_model(cfg, device)
                logger.info(f"Fitting model for {target}...")

                
                try:
                    model.fit(train_loader, test_loader=test_loader, save_dir=class_dir)
                    logger.info(f"Finished processing {target}.")
                except TypeError:
                    model.fit(train_loader)
                    if cfg.use_wandb:
                        wandb.finish() 
                    continue

                logger.info(f"Memory cleared. Moving to next target...")

                if cfg.use_wandb:
                    run.finish()

        try:
            del model
            del train_loader
            del test_loader
            del train_dataset
            del test_dataset
        except:
            pass

        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()
