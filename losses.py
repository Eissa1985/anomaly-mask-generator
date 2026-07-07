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