# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :test.py
# @Time     :2021/12/5 14:50
from options.test_options import TestOptions
from model.models import create_model
from SCDataset import SCDataLoader
import warnings

if __name__ == '__main__':
    warnings.filterwarnings('ignore')      # 取肖 之后 所有警告

    opt = TestOptions().parse()
    # print(opt)
    data_loader = SCDataLoader(opt)
    SCGan = create_model(opt, data_loader)
    SCGan.test()
    print("Finished!!!")




