# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :makeup.py
# @Time     :2021/11/26 22:30

import cv2
import os
import numpy as np
from skimage.filters import gaussian
from test import evaluate
import argparse
import keras
import time

# hair
def predict(model, im):
    h, w, _ = im.shape
    # inputs = cv2.resize(im, (512, 512))   # 480
    inputs = im.astype('float32')
    inputs.shape = (1,) + inputs.shape
    inputs = inputs / 255
    mask = model.predict(inputs)
    # ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY_INV)
    mask.shape = mask.shape[1:]
    mask = cv2.resize(mask, (w, h))
    mask.shape = h, w, 1
    return mask


def change_v(v, mask, target):
    # 染发
    epsilon = 1e-7
    x = v / 255                             # 数学化
    target = target / 255
    target = -np.log(epsilon + 1 - target)
    x_mean = np.sum(-np.log(epsilon + 1 - x)  * mask) / np.sum(mask)
    alpha = target / x_mean
    x = 1 - (1 - x) ** alpha
    v[:] = x * 255                          # 二进制化


def recolor(im, mask, color=[]):
    # 工程化
    print("color1:", color)
    print("type of color 1:", type(color))
    color = np.array(color, dtype='uint8', ndmin=3)
    print("color2:", color)
    print("type of color 2:", type(color))
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    color_hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    # 染发
    im_hsv[..., 0] = color_hsv[..., 0]      # 修改颜色
    change_v(im_hsv[..., 2:], mask, color_hsv[..., 2:])
    im_hsv[..., 1] = color_hsv[..., 1]      # 修改饱和度
    x = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
    im = im * (1 - mask) + x * mask
    return im

# hair

def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=1, multichannel=True)   # 减少滤波 lvbo

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[250, 250, 20]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]

    return changed

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--img-path', default='imgs/9.jpg')    # 图片
    return parse.parse_args()

if __name__ == '__main__':
    # 0  background
    # 1  skin/face
    # 2  nose        # 左 眉毛
    # 3  eye_g       # 右 眉毛
    # 4  r_eye       # 左 眼睛
    # 5  l_eye       # 右 眼睛
    # 6  l_brow
    # 7  r_brow      # 左 耳朵
    # 8  l_ear       # 右 耳朵
    # 9  r_ear
    # 10 nose        # 鼻子
    # 11 teeth       # 牙齿
    # 12 upper lip   # 上嘴唇
    # 13 lower lip   # 下嘴唇
    # 14 hat         # 脖子
    # 15 ear_r
    # 16 neck_l      # 衣服
    # 17 hair        # 头发
    # 18 cloth       # 帽子

    args = parse_args()

    table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13,
        'face': 1,
        'teeth': 11
    }

    image_path = args.img_path
    cp = 'weights/79999_iter.pth'

    image = cv2.imread(image_path)
    # image = cv2.resize(image, (1024, 1024))
    ori = image.copy()
    parsing = evaluate(image_path, cp)
    # print(parsing)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    # cv2.imshow('parsing', cv2.resize(parsing, (512, 512)))
    parts = [table['hair']]# , table['teeth'], table['lower_lip']
    colors = [[220, 70, 100]]  # , [20, 70, 180], [120, 170, 180]

    for part, color in zip(parts, colors):
        if part != 17:
            image = hair(image, parsing, part, color)

        if part == 17:
            pass

    #     print(part)
    #     print(color)
    # print(list(zip(parts,colors)))

    cv2.imshow('image', cv2.resize(ori, (512, 512)))
    cv2.imshow('color', cv2.resize(image, (512, 512)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()















