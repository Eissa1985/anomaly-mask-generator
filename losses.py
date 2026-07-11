import torch.nn as nn
import segmentation_models_pytorch.losses as losses
import torch
import torch.nn.functional as F

# class DiceLoss(nn.Module):
#     """
#     دالة خسارة Dice: هي المعادل الرياضي القابل للاشتقاق لمقياس IoU.
#     كلما قلت هذه الخسارة، زادت دقة تطابق حواف القناع المتوقع مع الحقيقي.
#     """
#     def __init__(self, smooth=1.0, eps=1e-7, log_loss=True):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#         self.eps = eps
#         self.log_loss = log_loss

#     def forward(self, inputs, targets):
#         # inputs: احتمالات من 0 إلى 1 (بعد Softmax)
#         # targets: قناع الحقيقة (0 أو 1)
        
#         # تسطيح المصفوفات لتسهيل حساب التقاطع والاتحاد
#         inputs = inputs.contiguous().view(-1)
#         targets = targets.contiguous().view(-1)
        
#         intersection = (inputs * targets).sum()
        
#         # الفصل بين smooth (لتحسين التدرج) و eps (لمنع القسمة على صفر)
#         dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth).clamp_min(self.eps)
        
#         # تطبيق Log-Dice (مستوحى من مكتبة SMP) لدفع التدرجات بقوة أكبر
#         if self.log_loss:
#             return -torch.log(dice.clamp_min(self.eps))
            
#         return 1 - dice

class IoUOptimizedLoss(nn.Module):
    """
    دالة الخسارة المجمعة: تعطي الأولوية القصوى (80%) لدقة الـ IoU 
    مع الاحتفاظ بـ (20%) لدقة التصنيف البكسلي عبر Focal.
    """
    def __init__(self, dice_weight=0.8, focal_weight=0.2):
        super(IoUOptimizedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(smooth=1e-4, gamma=0)
        self.dice_loss = losses.DiceLoss(mode='binary')
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        logits: الخرج الخام من المفكك بأبعاد [Batch, 2, Height, Width]
        targets: قناع الحقيقة بأبعاد [Batch, 1, Height, Width]
        """
        # 1. تحويل الخرج الخام (Logits) إلى احتمالات (Probabilities)
        # استخدام log_softmax().exp() بدلاً من softmax() للاستقرار العددي (من مكتبة SMP)
        probs = F.softmax(logits, dim=1)[:, 1:2, :, :]
        
        # 2. التأكد من تطابق أبعاد الأهداف لتجنب أخطاء الـ Broadcasting
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        # 3. حساب الخسائر المنفصلة
        bce_val = self.bce_loss(probs, targets)
        dice_val = self.dice_loss(probs, targets)
        focal_val = self.focal_loss(logits, targets)
        
        # 4. الدمج بالأوزان المحددة لرفع الـ IoU
        total_loss = (0.6 * (0.5 * dice_val + 0.5 * bce_val)) + (0.4 * focal_val)
        
        return total_loss
        
class FocalLoss(nn.Module):

    def __init__(self, smooth=1e-5, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth
        
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    
    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))

        target = target.view(-1, 1).long() 
        lprobs = F.log_softmax(input, dim=1)
        
        n_class = lprobs.size(1)
        
        one_hot = torch.zeros(target.size(0), n_class).to(input.device)
        one_hot.scatter_(1, target, 1)

        smooth_target = (1 - self.smooth) * one_hot + self.smooth / n_class
        p_t = torch.exp(lprobs.gather(1, target)) 
        focal_loss = -1 * (1 - p_t) ** self.gamma * torch.log(p_t)
        return focal_loss.mean()


class CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        return 0.5 * self.bce(y_pred, y_true) + 0.5 * self.dice(y_pred, y_true)

class SpectralLoss(nn.Module):
    def __init__(self, loss_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        if pred.shape != target.shape:
            target = F.interpolate(target, size=pred.shape[2:], mode='nearest')
            
        pred_prob = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
        
        fft_pred = torch.fft.rfft2(pred_prob, norm='ortho')
        fft_target = torch.fft.rfft2(target.float(), norm='ortho')
        
        amp_pred = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)
        
        log_amp_pred = torch.log(amp_pred + 1e-8)
        log_amp_target = torch.log(amp_target + 1e-8)
        
        spectral_distance = self.l1_loss(log_amp_pred, log_amp_target)
        
        return self.loss_weight * spectral_distance
