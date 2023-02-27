import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_true, y_pred):
        a =  self.bce_loss(y_pred, y_true)
        b =  self.soft_dice_loss(y_true, y_pred)
        return a + b

class partial_l1_loss(nn.Module):
    def __init__(self):
        super(partial_l1_loss, self).__init__()
        self.l1_loss = nn.L1Loss()
    
    def __call__(self, trunk_posi_true, trunk_posi_pred):
        trunk_posi_pseudo_pred = torch.where(trunk_posi_true==-1, trunk_posi_true, trunk_posi_pred)
        return self.l1_loss(trunk_posi_true, trunk_posi_pseudo_pred)

class partial_bce_loss(nn.Module):
    def __init__(self):
        super(partial_bce_loss, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def __call__(self, trunk_link_true, trunk_link_pred):
        trunk_link_pseudo_pred = torch.where(trunk_link_true!=-1, trunk_link_pred, torch.zeros_like(trunk_link_pred))
        trunk_link_pseudo_true = torch.where(trunk_link_true!=-1, trunk_link_true, torch.zeros_like(trunk_link_true))
        return self.bce_loss(trunk_link_pseudo_pred, trunk_link_pseudo_true)