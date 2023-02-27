import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np

from time import time

# from networks.unet import Unet
# from networks.dunet import Dunet
# from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from networks.dinknet import DinkNet34_WithBranch
from framework import MyFrame
from loss import dice_bce_loss, partial_l1_loss, partial_bce_loss
from data import ImageFolder

SHAPE = (1024,1024)
ROOT = 'dataset/train/'
imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
trainlist = list(map(lambda x: x[:-8], imagelist))
NAME = 'log04_dink34'
BATCHSIZE_PER_CARD = 4

solver = MyFrame(DinkNet34_WithBranch, dice_bce_loss, partial_l1_loss, partial_bce_loss, 2e-4)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4)

mylog = open('logs/'+NAME+'.log','w')
tic = time()
no_optim = 0
total_epoch = 500
train_epoch_best_loss = 100.
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    train_epoch_loss_pred = 0
    train_epoch_loss_prob = 0
    train_epoch_loss_posi = 0
    train_epoch_loss_link = 0
    for img, mask, if_key_points, all_key_points_position, anchor_link in data_loader_iter:
        solver.set_input(img, mask, if_key_points, all_key_points_position, anchor_link)
        train_loss, train_loss_pred, train_loss_prob, train_loss_posi, train_loss_link = solver.optimize()
        train_epoch_loss += train_loss
        train_epoch_loss_pred += train_loss_pred
        train_epoch_loss_prob += train_loss_prob
        train_epoch_loss_posi += train_loss_posi
        train_epoch_loss_link += train_loss_link
    train_epoch_loss /= len(data_loader_iter)
    train_epoch_loss_pred /= len(data_loader_iter)
    train_epoch_loss_prob /= len(data_loader_iter)
    train_epoch_loss_posi /= len(data_loader_iter)
    train_epoch_loss_link /= len(data_loader_iter)
    print('********', file=mylog)
    print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
    print('train_loss:', train_epoch_loss, file=mylog)
    print('pred_loss:', train_epoch_loss_pred, file=mylog)
    print('prob_loss:', train_epoch_loss_prob, file=mylog)
    print('posi_loss:', train_epoch_loss_posi, file=mylog)
    print('link_loss:', train_epoch_loss_link, file=mylog)
    print('SHAPE:', SHAPE, file=mylog)
    print('********')
    print('epoch:', epoch, '    time:', int(time() - tic))
    print('train_loss:', train_epoch_loss)
    print('pred_loss:', train_epoch_loss_pred)
    print('prob_loss:', train_epoch_loss_prob)
    print('posi_loss:', train_epoch_loss_posi)
    print('link_loss:', train_epoch_loss_link)
    print('SHAPE:', SHAPE)
    
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/'+NAME+'.th')
    if no_optim > 6:
        print('early stop at %d epoch' % epoch, file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
    mylog.flush()
    
print('Finish!', file=mylog)
print('Finish!')
mylog.close()