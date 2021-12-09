# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :makeup_main.py
# @Time     :2021/11/26 22:46

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QMessageBox #(对话框)
from main_ui import Ui_MainWindow
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.Qt import *
import sys
import numpy as np
import cv2
import torch
import os

from model import BiSeNet
from test import evaluate
import argparse
from skimage.filters import gaussian
import json
from torchvision import transforms
from PIL import Image
import keras
import math

sys.path.append('..')
# 妆容迁移
from SCGAN.model.SCGAN import SCGAN
from SCGAN.options.test_options import TestOptions
from SCGAN.model.models import create_model
from SCGAN.options.base_options import BaseOptions
from SCGAN.handle import label
from SCGAN.SCDataset import SCDataLoader


import os.path
import cv2
import torchvision.transforms as transforms
from PIL import Image
import PIL
import numpy as np
import torch
from torch.autograd import Variable


class DetThread(QThread):

    send_img = pyqtSignal(np.ndarray)
    send_makeup = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)    #  pyqtSignal 自定义型号  有一个 字典类型的参数

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './weights/79999_iter.pth'
        self.hair_model = './weights/weights.005.h5'  # keras
        self.source = ''   # 图片路径
        self.scgan_source = ''
        self.ok = 0
        self.save = './result.jpg'
        # 标志
        self.flag = 0
        # 存放颜色
        self.hair_color = [j for j in range(3)]
        self.hair_ok = 0

        self.Colors = [[j for j in range(3)] for i in range(19)]
        self.Parts  = [i for i in range(21,40)]


        self.table = {
            'face': 1,          # 面部
            'eyebrow_l': 2,     # 眉毛 左
            'eyebrow_r': 3,     # 眉毛 右
            'Eye_l': 4,         # 左 眼睛
            'Eye-r': 5,
            'glasses': 6,
            'ear_r': 7,
            'ear_l': 8,
            'no'   : 9,    # 暂时 不知
            'nose' : 10,
            'teeth': 11,        # 牙
            'upper_lip': 12,    # up 嘴唇
            'lower_lip': 13,
            'neck'     : 14,     # 脖子
            'no2'      : 15,     # 不知
            'clothes'  : 16,     # 上身衣物
            'hair'     : 17,
            'hat'      : 18      # 帽子
        }


    def cv_imread(self,filePath):  # 读取 中文 路径
        if filePath:
            cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8),1)
            return cv_img


    def make_up(self, image, parsing, part=17, color=[250, 250, 250]):

        b, g, r = color

        tar_color = np.zeros_like(image)

        tar_color[:, :, 0] = b
        tar_color[:, :, 1] = g
        tar_color[:, :, 2] = r

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

        if part == 17:
            return image    # 碰到头发直接 return
        else:
            if part == 12 or part == 13:
                image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]    #   h, s, v
            else:
                # image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]    #  image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
                image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

            changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
            changed[parsing != part] = image[parsing != part]

            return changed


    def sharpen(self, img, sigma):
        img = img * 1.0
        gauss_out = gaussian(img, sigma=sigma, multichannel=True)

        alpha = 1.5  # 1.5
        img_out = (img - gauss_out) * alpha + img

        img_out = img_out / 255.0

        mask_1 = img_out < 0
        mask_2 = img_out > 1

        img_out = img_out * (1 - mask_1)
        img_out = img_out * (1 - mask_2) + mask_2
        img_out = np.clip(img_out, 0, 1)
        img_out = img_out * 255
        return np.array(img_out, dtype=np.uint8)


    # hair
    def predicts(self, model, im):
        try:
            h, w, _ = im.shape
            inputs = cv2.resize(im, (512, 512))  # ok
            inputs = inputs.astype('float32')
            inputs.shape = (1,) + inputs.shape
            inputs = inputs / 255
            mask = model.predict(inputs)
            mask.shape = mask.shape[1:]
            mask = cv2.resize(mask, (w, h))
            mask.shape = h, w, 1
            return mask

        except Exception as e:  # 捕获异常
            print(repr(e))


    def recolor(self, im, mask, color=[]):
        try:
            # 工程化
            color = np.array(color, dtype='float', ndmin=3)
            # 染发
            epsilon = 1
            x = np.max(im, axis=2, keepdims=True)  # 获取亮度
            x_target = np.max(color)
            x = x / (255 + epsilon)  # 数学化
            x_target = x_target / (255 + epsilon)
            x = -np.log(1 - x)  # 来到真实世界（尽力了）
            x_target = -np.log(1 - x_target)
            x_mean = np.sum(x * mask) / np.sum(mask)
            x = x_target / x_mean * x  # 调整亮度
            x = 1 - np.exp(-x)  # 回到计算机
            x = x * (255 + epsilon)  # 二进制化
            im = im * (1 - mask) + (x * mask) * (color / np.max(color))
            return im
        except Exception as e:  # 捕获异常
            print(repr(e))


    def hair_main(self, model, im, save, color = [0 , 0, 0]):
        # model模型位置， ifn 原图， ofn 处理结果图， （考虑将color设置为含参数路由）
        if isinstance(model, str):
            model = keras.models.load_model(model, compile=False)

        mask = self.predicts(model, im)
        # print('mask_ok')
        img = self.recolor(im, mask, color)
        cv2.imwrite(save, img)
        result = cv2.imread(save)
        return result

    def mask_evaluate(self, image_path='', cp='', stride=1):
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        # net.cuda()   # 不传入 GPU
        net.load_state_dict(torch.load(cp, map_location=torch.device('cpu')))  # 使用cpu map_location=torch.device('cpu')
        net.eval()
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)  # 512
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # 处理 label
            vis_parsing_anno = parsing.copy().astype(np.uint8)
            vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

            cv2.imwrite('./SCGAN/face_mask/{}.png'.format('mask'), vis_parsing_anno)  # 需要的 图片
                # seg = np.array(Image.open('./SCGAN/face_mask/mask.png'))
                # new = np.zeros_like(seg)
                # new[seg == 0] = 0
                # new[seg == 1] = 4
                # new[seg == 2] = 7
                # new[seg == 3] = 2
                # new[seg == 4] = 6
                # new[seg == 5] = 1
                # new[seg == 6] = 8
                # new[seg == 7] = 9
                # new[seg == 8] = 11
                # new[seg == 9] = 13
                # new[seg == 10] = 12
                # new[seg == 11] = 3
                # new[seg == 12] = 5
                # new[seg == 13] = 10
                # img = Image.fromarray(new)
                # img.save('./SCGAN/face_mask/mask.png')
            # else:
            #     cv2.imwrite('./SCGAN/face_mask/{}.png'.format('scgan'), vis_parsing_anno)
            #     seg = np.array(Image.open('./SCGAN/face_mask/scgan.png'))
            #     new = np.zeros_like(seg)
            #     new[seg == 0] = 0
            #     new[seg == 1] = 4
            #     new[seg == 2] = 7
            #     new[seg == 3] = 2
            #     new[seg == 4] = 6
            #     new[seg == 5] = 1
            #     new[seg == 6] = 8
            #     new[seg == 7] = 9
            #     new[seg == 8] = 11
            #     new[seg == 9] = 13
            #     new[seg == 10] = 12
            #     new[seg == 11] = 3
            #     new[seg == 12] = 5
            #     new[seg == 13] = 10
            #     img = Image.fromarray(new)
            #     img.save('./SCGAN/face_mask/scgan.png')

            return 0



    def run(self):
        try:
            #--------------------------------- 读入 图片
            if self.source == '' :
                self.ok = 0
                return
            else :
                self.ok = 1
                img0 = self.cv_imread(self.source)
                img0 = cv2.resize(img0, (512, 512))

            # 除头发之外 使用这个
            img_res = img0.copy()
            parsing = evaluate(self.source, self.weights)
            parsing = cv2.resize(parsing, img0.shape[0:2], interpolation=cv2.INTER_NEAREST)
            # 头发 model

            if self.hair_ok == 1:
                # 加载 头发 模型
                # print(self.hair_color)
                img_res = self.hair_main(self.hair_model, img0, self.save, self.hair_color)
                # self.hair_ok = 0

            for part, color in zip(self.Parts[:self.flag], self.Colors[:self.flag]):
                img_res = self.make_up(img_res, parsing, part, color)

            self.send_img.emit(img_res)
            self.send_statistic.emit(dict())

        except Exception as e:  # 捕获异常
            print(repr(e))



class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.det_thread = DetThread()  #　　分割　类
        # self.SCGAN_DATA = SCDataLoader()  # 妆容迁移  数据处理 类
        # self.SCGAN_ = SCGAN()             # 　妆容迁移　模型　类
        self.scgan__source = ''             #  妆容图片 路径
        self.img_source = ''                # 原图 路径

        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.result))      # 结果图
        self.det_thread.send_makeup.connect(lambda x: self.show_image(x, self.label_result))   # z妆容图
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.label_raw))   # 原始图
        self.det_thread.send_statistic.connect(self.show_statistic)
        # self.RunProgram.triggered.connect(lambda: self.det_thread.start())
        self.RunProgram.triggered.connect(self.term_or_con)         # 运行模型按钮 term_or_con
        self.SelFile.triggered.connect(self.open_file)             # 打开文件 原始图片

        self.mask_file.triggered.connect(self.mask_img)
        self.SCGAN_img.triggered.connect(self.scgan_img)
        self.SCgan.triggered.connect(self.SCgan_makeup)

        self.cam_switch.triggered.connect(self.camera)

        self.hair_btn.clicked.connect(self.Hair)  # 连接 按钮槽函数
        self.uplip_btn.clicked.connect(self.uplip)
        self.lolip_btn.clicked.connect(self.lolip)
        self.nose_btn.clicked.connect(self.Nose)
        self.neck_btn.clicked.connect(self.Neck)
        self.eyebrow_L_btn.clicked.connect(self.eyebrow_L)
        self.eyebrow_R_btn.clicked.connect(self.eyebrow_R)
        self.face_btn.clicked.connect(self.Face)
        self.eye_l_btn.clicked.connect(self.eye_l)
        self.eye_r_btn.clicked.connect(self.eye_r)

        self.clear_btn.clicked.connect(self.Clear)


    def Hair(self): # 17
        try:
            # print('点击了 HAIR')
            # QColorDialog.setCustomColor(3, QColor(10, 60, 200))  # 设置自定义色块区的第三个色块颜色
            color = QColorDialog.getColor(QColorDialog.customColor(1), self, '选择颜色')
            palette = QPalette()  # 创建一个新调色板
            # palette.setColor(QPalette.Background, color)  # 传递颜色
            # self.setPalette(palette)  # 设置颜色
            # try:
            #     index = self.det_thread.Parts.index(17)  # table 是否在 Parts 中 在则返回位置 ，否则 返回 -1
            # except ValueError:
            #     index = -1
            # print(index)
            # if index == -1:
                # self.det_thread.Colors[self.det_thread.flag] =  list(color.getRgb()[0:3][::-1])
            self.det_thread.hair_color = list(color.getRgb()[0:3][::-1])
            self.det_thread.hair_ok = 1
            # self.det_thread.Parts[self.det_thread.flag] = 17
            # self.det_thread.flag = self.det_thread.flag + 1     # 无 重复 则 标志加一
            # else:        # 有重复
            #     self.det_thread.Colors[index] =  list(color.getRgb()[0:3][::-1])        # 给 重复的 table 赋值
                # self.det_thread.Parts[index] = 17
        except Exception as e:  # 捕获异常
            print(repr(e))

    def uplip(self): # 12
        try:
            color = QColorDialog.getColor(QColorDialog.customColor(3), self, '选择颜色')
            palette = QPalette()  # 创建一个新调色板
            try:
                index = self.det_thread.Parts.index(12)  # table 是否在 Parts 中 在则返回位置 ，否则 返回 -1
            except ValueError:
                index = -1

            # print(index, list(color.getRgb()))

            if index == -1:
                self.det_thread.Colors[self.det_thread.flag] =  list(color.getRgb()[0:3][::-1])
                self.det_thread.Parts[self.det_thread.flag] = 12
                self.det_thread.flag = self.det_thread.flag + 1  # 无 重复 则 标志加一
            else:  # 有重复
                self.det_thread.Colors[index] =  list(color.getRgb()[0:3][::-1])  # 给 重复的 table 赋值
                # self.det_thread.Parts[index] = 12

        except Exception as e:  # 捕获异常
            print(repr(e))

    def lolip(self):
        try:
            color = QColorDialog.getColor(QColorDialog.customColor(3), self, '选择颜色')
            palette = QPalette()  # 创建一个新调色板
            try:
                index = self.det_thread.Parts.index(13)  # table 是否在 Parts 中 在则返回位置 ，否则 返回 -1
            except ValueError:
                index = -1

            if index == -1:
                self.det_thread.Colors[self.det_thread.flag] =  list(color.getRgb()[0:3][::-1])
                self.det_thread.Parts[self.det_thread.flag] = 13
                self.det_thread.flag = self.det_thread.flag + 1  # 无 重复 则 标志加一
            else:  # 有重复
                self.det_thread.Colors[index] =  list(color.getRgb()[0:3][::-1])  # 给 重复的 table 赋值
                # self.det_thread.Parts[index] = 13

        except Exception as e:  # 捕获异常
            print(repr(e))

    def Nose(self):
        try:
            color = QColorDialog.getColor(QColorDialog.customColor(1), self, '选择颜色')
            palette = QPalette()  # 创建一个新调色板
            try:
                index = self.det_thread.Parts.index(10)  # table 是否在 Parts 中 在则返回位置 ，否则 返回 -1
            except ValueError:
                index = -1

            if index == -1:
                self.det_thread.Colors[self.det_thread.flag] =  list(color.getRgb()[0:3][::-1])
                self.det_thread.Parts[self.det_thread.flag] = 10
                self.det_thread.flag = self.det_thread.flag + 1  # 无 重复 则 标志加一
            else:  # 有重复
                self.det_thread.Colors[index] =  list(color.getRgb()[0:3][::-1])  # 给 重复的 table 赋值
                self.det_thread.Parts[index] = 10

        except Exception as e:  # 捕获异常
            print(repr(e))


    def Neck(self):
        try:
            color = QColorDialog.getColor(QColorDialog.customColor(1), self, '选择颜色')
            palette = QPalette()  # 创建一个新调色板
            try:
                index = self.det_thread.Parts.index(14)  # table 是否在 Parts 中 在则返回位置 ，否则 返回 -1
            except ValueError:
                index = -1

            if index == -1:
                self.det_thread.Colors[self.det_thread.flag] =  list(color.getRgb()[0:3][::-1])
                self.det_thread.Parts[self.det_thread.flag] = 14
                self.det_thread.flag = self.det_thread.flag + 1  # 无 重复 则 标志加一
            else:  # 有重复
                self.det_thread.Colors[index] =  list(color.getRgb()[0:3][::-1])  # 给 重复的 table 赋值
                self.det_thread.Parts[index] = 14

        except Exception as e:  # 捕获异常
            print(repr(e))

   # 左边眉毛
    def eyebrow_L(self):
        try:
            color = QColorDialog.getColor(QColorDialog.customColor(1), self, '选择颜色')
            palette = QPalette()  # 创建一个新调色板
            try:
                index = self.det_thread.Parts.index(3)  # table 是否在 Parts 中 在则返回位置 ，否则 返回 -1
            except ValueError:
                index = -1
            if index == -1:
                self.det_thread.Colors[self.det_thread.flag] =  list(color.getRgb()[0:3][::-1])
                self.det_thread.Parts[self.det_thread.flag] = 3
                self.det_thread.flag = self.det_thread.flag + 1  # 无 重复 则 标志加一
            else:  # 有重复
                self.det_thread.Colors[index] =  list(color.getRgb()[0:3][::-1])  # 给 重复的 table 赋值
                self.det_thread.Parts[index] = 3
        except Exception as e:  # 捕获异常
            print(repr(e))


    def eyebrow_R(self):
        try:
            color = QColorDialog.getColor(QColorDialog.customColor(1), self, '选择颜色')
            palette = QPalette()  # 创建一个新调色板
            try:
                index = self.det_thread.Parts.index(2)  # table 是否在 Parts 中 在则返回位置 ，否则 返回 -1
            except ValueError:
                index = -1
            if index == -1:
                self.det_thread.Colors[self.det_thread.flag] =  list(color.getRgb()[0:3][::-1])
                self.det_thread.Parts[self.det_thread.flag] = 2
                self.det_thread.flag = self.det_thread.flag + 1  # 无 重复 则 标志加一
            else:  # 有重复
                self.det_thread.Colors[index] =  list(color.getRgb()[0:3][::-1])  # 给 重复的 table 赋值
                self.det_thread.Parts[index] = 2
        except Exception as e:  # 捕获异常
            print(repr(e))


    def Face(self):
        try:
            color = QColorDialog.getColor(QColorDialog.customColor(1), self, '选择颜色')
            palette = QPalette()  # 创建一个新调色板
            try:
                index = self.det_thread.Parts.index(1)  # table 是否在 Parts 中 在则返回位置 ，否则 返回 -1
            except ValueError:
                index = -1
            if index == -1:
                self.det_thread.Colors[self.det_thread.flag] =  list(color.getRgb()[0:3][::-1])
                self.det_thread.Parts[self.det_thread.flag] = 1
                self.det_thread.flag = self.det_thread.flag + 1  # 无 重复 则 标志加一
            else:  # 有重复
                self.det_thread.Colors[index] =  list(color.getRgb()[0:3][::-1])  # 给 重复的 table 赋值
                self.det_thread.Parts[index] = 1

        except Exception as e:  # 捕获异常
            print(repr(e))


    def eye_l(self):
        try:
            color = QColorDialog.getColor(QColorDialog.customColor(1), self, '选择颜色')
            palette = QPalette()  # 创建一个新调色板
            try:
                index = self.det_thread.Parts.index(4)  # table 是否在 Parts 中 在则返回位置 ，否则 返回 -1
            except ValueError:
                index = -1
            if index == -1:
                self.det_thread.Colors[self.det_thread.flag] =  list(color.getRgb()[0:3][::-1])
                self.det_thread.Parts[self.det_thread.flag] = 4
                self.det_thread.flag = self.det_thread.flag + 1  # 无 重复 则 标志加一
            else:  # 有重复
                self.det_thread.Colors[index] =  list(color.getRgb()[0:3][::-1])  # 给 重复的 table 赋值
                # self.det_thread.Parts[index] = 4
        except Exception as e:  # 捕获异常
            print(repr(e))

    def eye_r(self):
        try:
            color = QColorDialog.getColor(QColorDialog.customColor(1), self, '选择颜色')
            palette = QPalette()  # 创建一个新调色板
            try:
                index = self.det_thread.Parts.index(5)  # table 是否在 Parts 中 在则返回位置 ，否则 返回 -1
            except ValueError:
                index = -1
            if index == -1:
                self.det_thread.Colors[self.det_thread.flag] = list(color.getRgb()[0:3][::-1])
                self.det_thread.Parts[self.det_thread.flag] = 5
                self.det_thread.flag = self.det_thread.flag + 1  # 无 重复 则 标志加一
            else:  # 有重复
                self.det_thread.Colors[index] = list(color.getRgb()[0:3][::-1])  # 给 重复的 table 赋值
                # self.det_thread.Parts[index] = 5

        except Exception as e:  # 捕获异常
            print(repr(e))


    def Clear(self):
        try:
            self.det_thread.flag = 0
            self.Colors = [[j for j in range(3)] for i in range(19)]
            self.Parts = [i for i in range(21, 40)]

            images = self.det_thread.cv_imread(self.img_source[0])  # 中文 OK
            images = cv2.resize(images, (512,512))
            # print('Clear')
            self.show_image(images, self.result)

        except Exception as e:  # 捕获异常
            print(repr(e))


    def SCgan_makeup(self):
        try:
            opt = TestOptions().parse()

            opt.dataroot = self.img_source[0]
            opt.dirmap = self.scgan__source[0]

            # 初始配置  源文件路径， 源文件label路径 ：固定，  妆容文件路径 : 固定
            data_loader = SCDataLoader(opt)
            SCGan = create_model(opt, data_loader)
            SCGan.test()
            # print("Finished!!!")
            images = self.det_thread.cv_imread('./results/fpx.jpg')  # 中文 OK
            # images = cv2.resize(images, (512, 512))
            self.show_image(images, self.result)
            QMessageBox.information(self, "Tips!", "Finished")

        except Exception as e:  # 捕获异常
            print(repr(e))


    def mask_img(self):   # 生成 mask 图片 并 保存
        try:
            self.det_thread.mask_evaluate(self.img_source[0], self.det_thread.weights, 1)
            # self.det_thread.mask_evaluate(self.scgan__source[0], self.det_thread.weights, 1, 1)
            QMessageBox.information(self, "Tips!", "mask已保存")
        except Exception as e:  # 捕获异常
            print(repr(e))


    def scgan_img(self):
        try:
            self.scgan__source = QFileDialog.getOpenFileName(self, '选择妆容图片', "./imgs", ""  "*.jpg *.png" )
            if self.scgan__source[0]:
                # print(self.scgan__source[0])
                self.det_thread.scgan_source = self.scgan__source[0]
                self.statusbar.showMessage('加载文件：{}'.format(os.path.basename(self.scgan__source[0])
                                                            if os.path.basename(self.scgan__source[0]) != '0'
                                                            else '摄像头设备'))
                images = self.det_thread.cv_imread(self.scgan__source[0])  # 中文 OK
                # images = cv2.resize(images, (512, 512))
                self.show_image(images, self.label_result)
        except Exception as e:  # 捕获异常
            print(repr(e))

    def status_bar_init(self):
        self.statusbar.showMessage('界面已准备')

    def open_file(self):
        try:
            self.img_source = QFileDialog.getOpenFileName(self, '选取图片', os.getcwd(), "Pic File( "           # *.mp4 *.mkv *.avi *.flv
                                                                               "*.jpg *.png)")
               # 返回 文件绝对路径 D:/plant_QT/PyQt5-YOLOv5-master/header100people.jpg    和 文件筛选类型
            # print(source[0])
            # 防止路径为空
            if self.img_source[0]:  # source[0] 文件绝对路径  D:/plant_QT/PyQt5-YOLOv5-master/header100people.jpg

                # print(self.img_source[0])
                self.det_thread.source = self.img_source[0]
                self.statusbar.showMessage('加载文件：{}'.format(os.path.basename(self.det_thread.source)
                                                        if os.path.basename(self.det_thread.source) != '0'
                                                        else '摄像头设备'))
                images = self.det_thread.cv_imread(self.img_source[0])   #  中文 OK
                images = cv2.resize(images, (512, 512))
                self.show_image(images, self.label_raw)
            else:
                QMessageBox.information(self, "Tips!", "未选择图片! 请选择图片")

        except Exception as e:    # 捕获异常
            print(repr(e))



    # 执行识别
    def term_or_con(self):

        self.det_thread.start()
        self.statusbar.showMessage('<<<<<<<<<<<<<< 正在上妆 >>>>>>>>>> ')


    def camera(self):
        if self.cam_switch.isChecked():
            self.det_thread.source = '0'
            self.statusbar.showMessage('摄像头已打开')
        else:
            self.det_thread.terminate()
            if hasattr(self.det_thread, 'vid_cap'):
                self.det_thread.vid_cap.release()
            if self.RunProgram.isChecked():
                self.RunProgram.setChecked(False)
            self.statusbar.showMessage('摄像头已关闭')


    # 处理结果
    def show_statistic(self):
        try:
            if self.det_thread.ok == 0:
                if self.det_thread.source == '':
                    QMessageBox.warning(self, "警告", "未选择图片！请选择图片")  # 消息对话框
                self.det_thread.ok = 1
            else :
                self.statusbar.showMessage('<<<<<<<<<<<<上妆完成>>>>>>>>>>>>')
                QMessageBox.information(self, "Tips", "OK! 上妆完成")   # 消息对话框
        except Exception as e:
            print(repr(e))


    @staticmethod
    def show_image(img_src, label):
        # print(img_src)
        try:
            ih, iw, _ = img_src.shape

            frame = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))   # 显示图片

        except Exception as e:
            print(repr(e))



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')  # 取肖 之后 所有警告

    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
# pyinstaller -F -i Tree.ico -w plant_win.py -p vit_model.py -p main_ui.py --hidden-import vit_model --hidden-import main_ui
# pyinstaller -F --onefile plant_win.spec

'''
解决 _C.cp38-win_amd64.pyd warning
for d in a.datas:
	if '_C.cp38-win_amd64.pyd' in d[0]:
		a.datas.remove(d)
		break
'''