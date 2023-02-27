"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os
import scipy.io

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

'''
def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask
'''

def randomHorizontalFlip(image, mask, if_key_points, all_key_points_position, anchor_link, u=0.5):
    new_anchor_link = np.zeros((8,64,64))
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

        if_key_points = np.flip(if_key_points, 2)

        all_key_points_position = np.flip(all_key_points_position, 2)
        all_key_points_position[1,:,:] = 1023 - all_key_points_position[1,:,:]
        all_key_points_position[all_key_points_position == 1024] = -1

        anchor_link = np.flip(anchor_link, 2)
        new_anchor_link[0,:,:] = anchor_link[0,:,:]
        new_anchor_link[1,:,:] = anchor_link[7,:,:]
        new_anchor_link[2,:,:] = anchor_link[6,:,:]
        new_anchor_link[3,:,:] = anchor_link[5,:,:]
        new_anchor_link[4,:,:] = anchor_link[4,:,:]
        new_anchor_link[5,:,:] = anchor_link[3,:,:]
        new_anchor_link[6,:,:] = anchor_link[2,:,:]
        new_anchor_link[7,:,:] = anchor_link[1,:,:]
    else:
        new_anchor_link = anchor_link

    return image, mask, if_key_points, all_key_points_position, new_anchor_link

def randomVerticleFlip(image, mask, if_key_points, all_key_points_position, anchor_link, u=0.5):
    new_anchor_link = np.zeros((8,64,64))
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

        if_key_points = np.flip(if_key_points, 1)

        all_key_points_position = np.flip(all_key_points_position, 1)
        all_key_points_position[0,:,:] = 1023 - all_key_points_position[0,:,:]
        all_key_points_position[all_key_points_position == 1024] = -1

        anchor_link = np.flip(anchor_link, 1)
        new_anchor_link[0,:,:] = anchor_link[4,:,:]
        new_anchor_link[1,:,:] = anchor_link[3,:,:]
        new_anchor_link[2,:,:] = anchor_link[2,:,:]
        new_anchor_link[3,:,:] = anchor_link[1,:,:]
        new_anchor_link[4,:,:] = anchor_link[0,:,:]
        new_anchor_link[5,:,:] = anchor_link[7,:,:]
        new_anchor_link[6,:,:] = anchor_link[6,:,:]
        new_anchor_link[7,:,:] = anchor_link[5,:,:]
    else:
        new_anchor_link = anchor_link

    return image, mask, if_key_points, all_key_points_position, new_anchor_link

def randomRotate90(image, mask, if_key_points, all_key_points_position, anchor_link, u=0.5): # 待修改
    new_all_key_points_position = np.zeros((2,64,64))
    new_anchor_link = np.zeros((8,64,64))
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

        if_key_points = if_key_points.transpose(1,2,0)
        if_key_points = np.rot90(if_key_points)
        if_key_points = if_key_points.transpose(2,0,1)

        all_key_points_position = all_key_points_position.transpose(1,2,0)
        all_key_points_position = np.rot90(all_key_points_position)
        all_key_points_position = all_key_points_position.transpose(2,0,1)
        new_all_key_points_position[0,:,:] = 1023 - all_key_points_position[1,:,:]
        new_all_key_points_position[1,:,:] = all_key_points_position[0,:,:]
        new_all_key_points_position[new_all_key_points_position == 1024] = -1

        anchor_link = anchor_link.transpose(1,2,0)
        anchor_link = np.rot90(anchor_link)
        anchor_link = anchor_link.transpose(2,0,1)
        new_anchor_link[0,:,:] = anchor_link[2,:,:]
        new_anchor_link[1,:,:] = anchor_link[3,:,:]
        new_anchor_link[2,:,:] = anchor_link[4,:,:]
        new_anchor_link[3,:,:] = anchor_link[5,:,:]
        new_anchor_link[4,:,:] = anchor_link[6,:,:]
        new_anchor_link[5,:,:] = anchor_link[7,:,:]
        new_anchor_link[6,:,:] = anchor_link[0,:,:]
        new_anchor_link[7,:,:] = anchor_link[1,:,:]
    else:
        new_all_key_points_position = all_key_points_position
        new_anchor_link = anchor_link

    return image, mask, if_key_points, new_all_key_points_position, new_anchor_link

def default_loader(id, root):
    img = cv2.imread(os.path.join(root,'{}_sat.jpg').format(id))
    mask = cv2.imread(os.path.join(root+'{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
    key_points = scipy.io.loadmat(os.path.join(root+'{}_mask.mat').format(id))
    if_key_points = key_points["if_key_points"]
    all_key_points_position = key_points["all_key_points_position"]
    anchor_link = key_points["anchor_link"]

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    '''
    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    '''

    img, mask, if_key_points, all_key_points_position, anchor_link = randomHorizontalFlip(img, mask, if_key_points, all_key_points_position, anchor_link)
    img, mask, if_key_points, all_key_points_position, anchor_link = randomVerticleFlip(img, mask, if_key_points, all_key_points_position, anchor_link)
    img, mask, if_key_points, all_key_points_position, anchor_link = randomRotate90(img, mask, if_key_points, all_key_points_position, anchor_link)
    
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    mask[mask>=0.5] = 1
    mask[mask<=0.5] = 0
    new_all_key_points_position = all_key_points_position % 16
    new_all_key_points_position[all_key_points_position == -1] = -16
    new_all_key_points_position = new_all_key_points_position.astype(np.float64) / 16
    
    return img, mask, if_key_points, new_all_key_points_position, anchor_link

class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask, if_key_points, all_key_points_position, anchor_link = self.loader(id, self.root)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        if_key_points_contiguous = np.ascontiguousarray(if_key_points)
        if_key_points = torch.Tensor(if_key_points_contiguous)
        all_key_points_position_contiguous = np.ascontiguousarray(all_key_points_position)
        all_key_points_position = torch.Tensor(all_key_points_position_contiguous)
        anchor_link_contiguous = np.ascontiguousarray(anchor_link)
        anchor_link = torch.Tensor(anchor_link_contiguous)
        return img, mask, if_key_points, all_key_points_position, anchor_link

    def __len__(self):
        return len(self.ids)