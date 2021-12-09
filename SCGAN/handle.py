# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :handle.py
# @Time     :2021/12/7 0:26
from PIL import Image
import numpy as np
root='./face_mask/'
paths=['mask.png', 'scgan.png']

def label():
    for path in paths:
        path=root+path
        seg = np.array(Image.open(path))
        new = np.zeros_like(seg)
        new[seg == 0] = 0
        new[seg == 1] = 4
        new[seg == 2] = 7
        new[seg == 3] = 2
        new[seg == 4] = 6
        new[seg == 5] = 1
        new[seg == 6] = 8
        new[seg == 7] = 9
        new[seg == 8] = 11
        new[seg == 9] = 13
        new[seg == 10] = 12
        new[seg == 11] = 3
        new[seg == 12] = 5
        new[seg == 13] = 10
        img = Image.fromarray(new)
        img.save(path)

# label()