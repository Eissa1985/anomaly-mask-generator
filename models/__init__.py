from .coordatt import *
from .decoder import *
from .eemfnet import EEMFNet
from .msff import *

def get_model(cfg, device):
    name = cfg.model_name.lower()
    if name == 'eemfnet':
        return EEMFNet(device=device, config=cfg)
    else:
        raise ValueError(f"Model '{cfg.model_name}' is not found in the model registry.")