import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np

class MyFrame():
    def __init__(self, net, dice_bce_loss, partial_l1_loss, partial_bce_loss, lr=2e-4, evalmode = False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.dice_bce_loss = dice_bce_loss()
        self.partial_l1_loss = partial_l1_loss()
        self.partial_bce_loss = partial_bce_loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, if_key_points=None, all_key_points_position=None, anchor_link=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.if_key_points = if_key_points
        self.all_key_points_position = all_key_points_position
        self.anchor_link = anchor_link
        self.img_id = img_id

    '''
    def test_one_img(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask
    
    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id
    
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask
    '''

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
        if self.if_key_points is not None:
            self.if_key_points = V(self.if_key_points.cuda(), volatile=volatile)
        if self.all_key_points_position is not None:
            self.all_key_points_position = V(self.all_key_points_position.cuda(), volatile=volatile)
        if self.anchor_link is not None:
            self.anchor_link = V(self.anchor_link.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred, trunk_prob, trunk_posi, trunk_link = self.net.forward(self.img)
        loss_pred = self.dice_bce_loss(self.mask, pred)
        loss_prob = self.dice_bce_loss(self.if_key_points, trunk_prob)
        loss_posi = 14.5 * self.partial_l1_loss(self.all_key_points_position, trunk_posi)
        loss_link = 14.5 * self.partial_bce_loss(self.anchor_link, trunk_link)
        loss = loss_pred + 0.5 * (loss_prob + loss_posi + loss_link) # 权重仍需实验调整
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_pred.item(), loss_prob.item(), loss_posi.item(), loss_link.item()
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))
    
    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file=mylog)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
