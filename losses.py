import torch.nn as nn
import segmentation_models_pytorch.losses as losses
import torch
import torch.nn.functional as F

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
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        pt = input
        logpt = (pt + 1e-5).log()

        # add label smoothing
        num_class = input.shape[1]
        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != input.device:
            one_hot_key = one_hot_key.to(input.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
            logpt = logpt * one_hot_key

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = (-1 * (1 - pt)**self.gamma * logpt).sum(1)
        if self.size_average: return loss.mean()
        else: return loss.sum()

class DiceLoss(nn.Module):
    def __init__(self, smooth=100.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
            
        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice
            
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         # inputs يجب أن تكون (Logits) قبل الـ Sigmoid/Softmax
#         bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-bce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
#         return focal_loss.mean()

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1.0):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, inputs, targets):
#         # تطبيق Sigmoid للحصول على احتمالات بين 0 و 1
#         inputs = torch.sigmoid(inputs)
        
#         # تسطيح المصفوفات (Flatten)
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
#         return 1 - dice

# class EEMFNetLoss(nn.Module):
#     def __init__(self, focal_weight=0.5, dice_weight=0.5):
#         super(EEMFNetLoss, self).__init__()
#         self.focal_weight = focal_weight
#         self.dice_weight = dice_weight
#         self.focal = FocalLoss(gamma=2.0)
#         self.dice = DiceLoss()

#     def forward(self, preds, masks):
#         # preds: مخرجات الموديل (Logits)
#         # masks: الماسك الحقيقي (Ground Truth)
#         loss_focal = self.focal(preds, masks)
#         loss_dice = self.dice(preds, masks)
        
#         total_loss = (self.focal_weight * loss_focal) + (self.dice_weight * loss_dice)
#         return total_loss

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
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(smooth=1e-4, gamma=0)
        # self.dice_loss = losses.DiceLoss(mode='binary')
        self.bce_loss = nn.L1Loss() #nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        logits: الخرج الخام من المفكك بأبعاد [Batch, 2, Height, Width]
        targets: قناع الحقيقة بأبعاد [Batch, 1, Height, Width]
        """
        # 1. تحويل الخرج الخام (Logits) إلى احتمالات (Probabilities)
        # استخدام log_softmax().exp() بدلاً من softmax() للاستقرار العددي (من مكتبة SMP)
        probs = F.softmax(logits, dim=1)
        focal_val = self.focal_loss(probs, targets)
        
        probs = probs[:, 1:2, :, :]
        
        # 2. التأكد من تطابق أبعاد الأهداف لتجنب أخطاء الـ Broadcasting
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        # 3. حساب الخسائر المنفصلة
        bce_val = self.bce_loss(probs, targets)
        dice_val = self.dice_loss(probs, targets)
        
        # 4. الدمج بالأوزان المحددة لرفع الـ IoU
        total_loss = (0.6 * (0.5 * dice_val + 0.5 * bce_val)) + (0.4 * focal_val)
        
        return total_loss
        
# class FocalLoss(nn.Module):

#     def __init__(self, smooth=1e-5, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.smooth = smooth
        
#         if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
#         if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

    
#     def forward(self, input, target):
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)
#             input = input.transpose(1, 2)
#             input = input.contiguous().view(-1, input.size(2))

#         target = target.view(-1, 1).long() 
#         lprobs = F.log_softmax(input, dim=1)
        
#         n_class = lprobs.size(1)
        
#         one_hot = torch.zeros(target.size(0), n_class).to(input.device)
#         one_hot.scatter_(1, target, 1)

#         smooth_target = (1 - self.smooth) * one_hot + self.smooth / n_class
#         p_t = torch.exp(lprobs.gather(1, target)) 
#         focal_loss = -1 * (1 - p_t) ** self.gamma * torch.log(p_t)
#         return focal_loss.mean()


# class CompositeLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.dice = losses.DiceLoss(mode='binary')
#         self.bce = nn.BCEWithLogitsLoss()

#     def forward(self, y_pred, y_true):
#         return 0.5 * self.bce(y_pred, y_true) + 0.5 * self.dice(y_pred, y_true)

# class SpectralLoss(nn.Module):
#     def __init__(self, loss_weight=0.1):
#         super().__init__()
#         self.loss_weight = loss_weight
#         self.l1_loss = nn.L1Loss()

#     def forward(self, pred, target):
#         if pred.shape != target.shape:
#             target = F.interpolate(target, size=pred.shape[2:], mode='nearest')
            
#         pred_prob = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
        
#         fft_pred = torch.fft.rfft2(pred_prob, norm='ortho')
#         fft_target = torch.fft.rfft2(target.float(), norm='ortho')
        
#         amp_pred = torch.abs(fft_pred)
#         amp_target = torch.abs(fft_target)
        
#         log_amp_pred = torch.log(amp_pred + 1e-8)
#         log_amp_target = torch.log(amp_target + 1e-8)
        
#         spectral_distance = self.l1_loss(log_amp_pred, log_amp_target)
        
#         return self.loss_weight * spectral_distance
