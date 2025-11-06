import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        if target.dtype != torch.float32:
            target = target.float()

        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred, target):
        if target.dtype != torch.float32:
            target = target.float()

        BCE_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        return focal_loss.mean()

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_boundary, target_boundary):
        if target_boundary.dtype != torch.float32:
            target_boundary = target_boundary.float()

        return F.binary_cross_entropy_with_logits(pred_boundary, target_boundary)

class ForgerySegmentationLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=gamma)
        self.boundary_loss = BoundaryLoss()
        
    def forward(self, predictions, targets):
        if targets.dtype != torch.float32:
            targets = targets.float()

        seg_pred = predictions['segmentation']
        boundary_pred = predictions['boundary']
        aux_preds = predictions.get('auxiliary', [])
        
        # Main segmentation loss
        loss_seg = (self.alpha * self.focal_loss(seg_pred, targets) +
                   self.beta * self.dice_loss(seg_pred, targets))
        
        # Boundary loss
        boundary_mask = self._extract_boundary(targets)
        loss_boundary = self.boundary_loss(boundary_pred, boundary_mask)
        
        # Auxiliary losses
        loss_aux = torch.tensor(0.0, device=seg_pred.device)
        for aux_pred in aux_preds:
            loss_aux += self.focal_loss(aux_pred, targets) * 0.4
        
        total_loss = loss_seg + 0.3 * loss_boundary + loss_aux
        
        return {
            'total': total_loss,
            'segmentation': loss_seg,
            'boundary': loss_boundary,
            'auxiliary': loss_aux
        }
    
    def _extract_boundary(self, mask, kernel_size=3):
        """Извлечение границ из маски сегментации"""
        if mask.dim() == 4:  # [B, 1, H, W]
            mask = mask.squeeze(1)  # [B, H, W]
        
        boundary_masks = []
        for i in range(mask.size(0)):
            single_mask = mask[i].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # Create kernel
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
            
            # Erosion and dilation
            padded = F.pad(single_mask, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), 
                          mode='reflect')
            eroded = F.conv2d(padded, kernel) < kernel_size * kernel_size
            
            padded_inv = F.pad(1 - single_mask, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), 
                              mode='reflect')
            dilated = F.conv2d(padded_inv, kernel) < kernel_size * kernel_size
            
            boundary = (eroded > 0) & (dilated > 0)
            boundary_masks.append(boundary.float())
        
        return torch.cat(boundary_masks, dim=0)