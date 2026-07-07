import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score
from bisect import bisect
from sklearn.metrics import precision_recall_curve

class AnomalyEvaluator:
    def __init__(self, pro_integration_limit=0.3):
        self.pro_integration_limit = pro_integration_limit

    def find_optimal_threshold(self, labels, scores):
        """حساب أفضل عتبة تفصل بين السليم والمعيب بناء على F1-Score"""
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        
        a = 2 * precision[:-1] * recall[:-1]
        b = precision[:-1] + recall[:-1]
        f1_scores = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        return best_threshold, best_f1
    
    def compute_pixel_auc(self, anomaly_maps, ground_truth_masks):
        """حساب Pixel-Level AUROC القياسي"""
        flat_scores = anomaly_maps.flatten()
        flat_masks = ground_truth_masks.flatten() #.astype(int)

        best_threshold, _ = self.find_optimal_threshold(flat_masks, flat_scores)
        
        if len(np.unique(flat_masks)) < 2:
            return 0.5
        
        return roc_auc_score((flat_masks > best_threshold).astype(int), flat_scores)
        # return roc_auc_score(flat_masks, flat_scores)

    def compute_pro_score(self, anomaly_maps, ground_truth_masks, return_curve=False):
        if anomaly_maps.ndim == 4: anomaly_maps = anomaly_maps.squeeze(1)
        if ground_truth_masks.ndim == 4: ground_truth_masks = ground_truth_masks.squeeze(1)
        
        ground_truth_masks = (ground_truth_masks > 0.5).astype(int)

        all_fprs, all_pros = self._compute_pro_curve_official(anomaly_maps, ground_truth_masks)

        au_pro = self._trapezoid(all_fprs, all_pros, x_max=self.pro_integration_limit)
        
        au_pro /= self.pro_integration_limit

        if return_curve:
            return au_pro, all_fprs, all_pros
        
        return au_pro

    def _compute_pro_curve_official(self, anomaly_maps, ground_truth_maps):
        structure = np.ones((3, 3), dtype=int)
        num_ok_pixels = 0
        num_gt_regions = 0

        shape = (len(anomaly_maps), anomaly_maps[0].shape[0], anomaly_maps[0].shape[1])
        fp_changes = np.zeros(shape, dtype=np.uint32)
        pro_changes = np.zeros(shape, dtype=np.float64)

        for gt_ind, gt_map in enumerate(ground_truth_maps):
            labeled, n_components = label(gt_map, structure)
            num_gt_regions += n_components

            ok_mask = labeled == 0
            num_ok_pixels += np.sum(ok_mask)

            fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
            fp_change[ok_mask] = 1
            fp_changes[gt_ind, :, :] = fp_change

            pro_change = np.zeros_like(gt_map, dtype=np.float64)
            for k in range(n_components):
                region_mask = labeled == (k + 1)
                region_size = np.sum(region_mask)
                pro_change[region_mask] = 1. / region_size

            pro_changes[gt_ind, :, :] = pro_change

        anomaly_scores_flat = np.array(anomaly_maps).ravel()
        fp_changes_flat = fp_changes.ravel()
        pro_changes_flat = pro_changes.ravel()

        sort_idxs = np.argsort(anomaly_scores_flat)[::-1]

        np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
        anomaly_scores_sorted = anomaly_scores_flat
        np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
        fp_changes_sorted = fp_changes_flat
        np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
        pro_changes_sorted = pro_changes_flat

        del sort_idxs

        # Cumsum
        np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
        fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
        np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
        fprs = fp_changes_sorted

        np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
        np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
        pros = pro_changes_sorted

        keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, True)
        del anomaly_scores_sorted

        fprs = fprs[keep_mask]
        pros = pros[keep_mask]
        del keep_mask
        
        # Clip and Pad
        np.clip(fprs, 0, 1, out=fprs)
        np.clip(pros, 0, 1, out=pros)

        zero = np.array([0.])
        one = np.array([1.])
        return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))

    def _trapezoid(self, x, y, x_max=None):
        x = np.asarray(x)
        y = np.asarray(y)
        
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        correction = 0.
        if x_max is not None:
            if x_max not in x:
                ins = bisect(x, x_max)
                if 0 < ins < len(x):
                    # Interpolation
                    y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
                    correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])
            
            mask = x <= x_max
            x = x[mask]
            y = y[mask]

        return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction
   