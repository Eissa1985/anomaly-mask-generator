import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from torchvision import transforms
from PIL import Image
import cv2

class ErrorAnalyzer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def find_optimal_threshold(self, labels, scores):
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        
        a = 2 * precision[:-1] * recall[:-1]
        b = precision[:-1] + recall[:-1]
        f1_scores = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        return best_threshold, best_f1

    def analyze_and_save(self, class_name, img_paths, labels, scores, anomaly_maps, gt_masks, model_device):
        best_threshold, f1 = self.find_optimal_threshold(labels, scores)
        print(f"[{class_name}] Optimal Threshold: {best_threshold:.4f} | Max F1-Score: {f1:.4f}")

        base_path = os.path.join(self.save_dir, class_name, "error_analysis")
        categories = ["False_Negative_Missed", "False_Positive_Alarm", "True_Positive_Success", "True_Negative_Success"]
        for cat in categories:
            os.makedirs(os.path.join(base_path, cat), exist_ok=True)

        print(f"[{class_name}] Saving analysis images... (This helps in studying failures)")
        optimal_threshold, _ = self.find_optimal_threshold(gt_masks.flatten(), anomaly_maps.flatten()) 
        for i in range(len(labels)):
            is_anomaly = labels[i] == 1
            pred_anomaly = scores[i] >= best_threshold
            
            if is_anomaly and not pred_anomaly:
                category = "False_Negative_Missed" 
            elif not is_anomaly and pred_anomaly:
                category = "False_Positive_Alarm"
            elif is_anomaly and pred_anomaly:
                category = "True_Positive_Success"
            else:
                category = "True_Negative_Success"
                if np.random.rand() > 0.1: continue 

            self._save_visual_report(
                img_path=img_paths[i],
                anomaly_map=anomaly_maps[i],
                gt_mask=gt_masks[i],
                score=scores[i],
                best_threshold=optimal_threshold,
                save_path=os.path.join(base_path, category, f"{category}_{i}.png")
            )
            
        return best_threshold

    def clean_mask(self, mask_np, kernel_size=3, min_area=20):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(mask_np.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                cleaned[labels == i] = 0
                
        return cleaned

    def _save_visual_report(self, img_path, anomaly_map, gt_mask, score, best_threshold, save_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((anomaly_map.shape[1], anomaly_map.shape[0]))
        img_np = np.array(img) / 255.0

        if np.max(gt_mask) == 0:
            threshold = best_threshold
        else:
            precision, recall, thresholds = precision_recall_curve(gt_mask.reshape(-1), anomaly_map.flatten())
            a = 2 * precision[:-1] * recall[:-1]
            b = precision[:-1] + recall[:-1]
            f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            threshold = thresholds[np.argmax(f1)]

        pred_mask = (anomaly_map > threshold).astype(np.uint8)
        clean = self.clean_mask(pred_mask, kernel_size=3)
        segmentation = np.array(clean).astype(int)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original Image\nScore: {score:.3f}")
        axes[0].axis('off')

        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')

        axes[2].imshow(img_np, alpha=0.5)
        im = axes[2].imshow(anomaly_map, cmap='jet', alpha=0.5)
        axes[2].set_title(f"AI Prediction Map\nMax Val: {anomaly_map.max():.3f}")
        axes[2].axis('off')
        
        axes[3].imshow(segmentation, cmap='gray')
        axes[3].set_title(f"Threshold Result\n(Thresh: {threshold:.3f})")
        axes[3].axis('off')

        plt.suptitle(f"Analysis Report - Score: {score:.4f} vs Threshold: {threshold:.4f}", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()