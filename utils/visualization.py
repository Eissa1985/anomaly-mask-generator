import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.segmentation import mark_boundaries

def save_anomaly_map(image, anomaly_map, save_path):
    if image.max() > 1: image = image / 255.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(image, cmap='gray', alpha=0.5)
    im = axes[1].imshow(anomaly_map, cmap='jet', alpha=0.5)
    axes[1].set_title("Anomaly Map")
    axes[1].axis('off')
    
    plt.colorbar(im, ax=axes[1])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_pro_curve(fprs, pros, auc_score, save_path, limit=0.3):
    plt.figure(figsize=(8, 6))
    plt.plot(fprs, pros, color='darkorange', lw=2, label=f'PRO curve (AUC = {auc_score:.3f})')
    plt.axvline(x=limit, color='navy', linestyle='--', label=f'Integration limit ({limit})')
    
    plt.xlim([0.0, 1.0]) 
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('Per-Region Overlap (PRO)')
    plt.title('FPR vs PRO Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()