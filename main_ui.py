# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :main_ui.py
# @Time     :2021/11/26 22:45
# -*- coding: utf-8 -*-


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton,\
     QComboBox , QFrame
    # 适应最小内容长度  # 下拉列表  QComboBox

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1850, 700)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./icon/灰原哀.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)


        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setStyleSheet("background: #19232D;")    # 主界面背景

        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(11)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)

        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox)

        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(4)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName("horizontalLayout")

        # 输入结果 文字
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setScaledContents(True)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_2.setStyleSheet("background: #19232D;"  
                                    "font-size: 14pt;"
                                    "color:White;"
                                    )
        self.horizontalLayout.addWidget(self.label_2, 0, QtCore.Qt.AlignHCenter)

        # 妆容图片
        self.makeup = QtWidgets.QLabel(self.groupBox)
        self.makeup.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.makeup.setFont(font)
        self.makeup.setScaledContents(True)
        self.makeup.setAlignment(QtCore.Qt.AlignCenter)
        self.makeup.setObjectName("label")
        self.makeup.setStyleSheet(";"  # QComboBox{background:Green};
                                 "font-size: 14pt;"
                                 "color:White;"
                                 )
        self.horizontalLayout.addWidget(self.makeup)
        self.verticalLayout_3.addLayout(self.horizontalLayout)

        # 检测结果
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label.setStyleSheet(";"  # QComboBox{background:Green};
                                    "font-size: 14pt;"
                                    "color:White;"
                                    )
        self.horizontalLayout.addWidget(self.label)
        self.verticalLayout_3.addLayout(self.horizontalLayout)


        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        # 显示 输入图片
        self.label_raw = QtWidgets.QLabel(self.groupBox)
        self.label_raw.setAutoFillBackground(False)
        self.label_raw.setStyleSheet("background-color:  #19232D;")
        self.label_raw.setText("")
        self.label_raw.setScaledContents(False)
        self.label_raw.setAlignment(QtCore.Qt.AlignCenter)
        self.label_raw.setFrameShape(QFrame.Box)
        self.label_raw.setScaledContents(True)
        self.label_raw.setObjectName("label_raw")
        self.horizontalLayout_2.addWidget(self.label_raw)
        # 显示 检测图片
        self.label_result = QtWidgets.QLabel(self.groupBox)
        self.label_result.setFrameShape(QFrame.Box)
        self.label_result.setText("")
        self.label_result.setScaledContents(False)
        self.label_result.setScaledContents(True)
        self.label_result.setObjectName("label_result")
        self.horizontalLayout_2.addWidget(self.label_result)

        self.result = QtWidgets.QLabel(self.groupBox)
        self.result.setFrameShape(QFrame.Box)
        self.result.setText("")
        self.result.setScaledContents(False)
        self.result.setScaledContents(True)
        self.result.setObjectName("result")
        self.horizontalLayout_2.addWidget(self.result)

        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout_3.setStretch(1, 10)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)

        # self.line = QtWidgets.QFrame(self.groupBox)
        # self.line.setFrameShape(QtWidgets.QFrame.VLine)
        # self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.line.setObjectName("line")
        # self.horizontalLayout_3.addWidget(self.line)

        # self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        # self.verticalLayout_2.setObjectName("verticalLayout_2")
        # self.statistic = QtWidgets.QPushButton(self.groupBox)
        # self.statistic.setEnabled(False)
        # font = QtGui.QFont()
        # font.setFamily("Agency FB")
        # font.setPointSize(11)
        # font.setStyleStrategy(QtGui.QFont.PreferDefault)
        # self.statistic.setFont(font)
        # self.statistic.setAcceptDrops(False)
        # self.statistic.setAutoFillBackground(False)
        # self.statistic.setStyleSheet("background:transparent;"
        #                              "font-size: 14pt;"
        #                              "color:White;")
        # self.statistic.setObjectName("statistic")
        # self.verticalLayout_2.addWidget(self.statistic, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        # self.horizontalLayout_3.addLayout(self.verticalLayout_2)


        self.horizontalLayout_3.setStretch(0, 9)
        self.horizontalLayout_3.setStretch(1, 2)   #
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.verticalLayout.addWidget(self.groupBox)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.statusbar.setStyleSheet("background: #19232D;"  
                                    "font-size: 13pt;"
                                    "color:#90EE90;")

        self.toolBar = QtWidgets.QToolBar(MainWindow)    # QToolBar控件是由文本按钮，图标或其他小控件按钮组成的可移动面板
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.toolBar.setStyleSheet("background:#3D59AB;")  # #50C878
        # 妆容
        # self.makeup_btn = QPushButton(MainWindow)
        # self.makeup_btn.setStyleSheet("background: #292421;"  # QComboBox{background:Green};
        #                             "font-size: 14pt;"
        #                             "color:#FFE4E1;"
        #                             )
        # self.makeup_btn.setText("makeup")  # text
        # self.makeup_btn.setObjectName("makeup_btn")
        # self.makeup_btn.move(150, 10)
        # self.horizontalLayout_4.addWidget(self.makeup_btn)


        # hair button
        self.hair_btn =QPushButton(MainWindow)
        self.hair_btn.setStyleSheet("background: #292421;"  # QComboBox{background:Green};
                                    "font-size: 14pt;"
                                    "color:#FFE4E1;"
                                    )
        self.hair_btn.setText("Hair")  # text
        self.hair_btn.setObjectName("hair_btn")
        self.hair_btn.move(150,10)
        self.horizontalLayout_4.addWidget(self.hair_btn)
        # 上嘴唇
        self.uplip_btn =QPushButton(MainWindow)
        self.uplip_btn.setStyleSheet("background: #0B1746;"  # QComboBox{background:Green};
                                    "font-size: 14pt;"
                                    "color:#FFE4E1;"
                                    )
        self.uplip_btn.setText("UpLips")  # text
        self.uplip_btn.setObjectName("upLips_btn")
        self.horizontalLayout_4.addWidget(self.uplip_btn)
        # 下嘴唇
        self.lolip_btn = QPushButton(MainWindow)
        self.lolip_btn.setStyleSheet("background: #0B1746;"  # QComboBox{background:Green};
                                     "font-size: 14pt;"
                                     "color:#FFE4E1;"
                                     )
        self.lolip_btn.setText("LoLips")  # text
        self.lolip_btn.setObjectName("loLips_btn")
        self.horizontalLayout_4.addWidget(self.lolip_btn)
        # 鼻子
        self.nose_btn = QPushButton(MainWindow)
        self.nose_btn.setStyleSheet("background: #5E2612;"  # QComboBox{background:Green};
                                     "font-size: 14pt;"
                                     "color:#FFE4E1;"
                                     )
        self.nose_btn.setText("Nose")  # text
        self.nose_btn.setObjectName("nose_btn")
        self.horizontalLayout_4.addWidget(self.nose_btn)
        # 眉毛 L
        self.eyebrow_L_btn = QPushButton(MainWindow)
        self.eyebrow_L_btn.setStyleSheet("background: #228B22;" 
                                    "font-size: 14pt;"
                                    "color:#FFE4E1;"
                                    )
        self.eyebrow_L_btn.setText("eyebrow_L")  # text
        self.eyebrow_L_btn.setObjectName("eyebrow_L_btn")
        self.horizontalLayout_4.addWidget(self.eyebrow_L_btn)

        self.eyebrow_R_btn = QPushButton(MainWindow)
        self.eyebrow_R_btn.setStyleSheet("background: #228B22;"  
                                    "font-size: 14pt;"
                                    "color:#FFE4E1;"
                                    )
        self.eyebrow_R_btn.setText("eyebrow_R")  # text
        self.eyebrow_R_btn.setObjectName("eyebrow_R_btn")
        self.horizontalLayout_4.addWidget(self.eyebrow_R_btn)

        # 脖子
        self.neck_btn = QPushButton(MainWindow)
        self.neck_btn.setStyleSheet("background: #A0522D;"  # QComboBox{background:Green};
                                         "font-size: 14pt;"
                                         "color:#FFE4E1;"
                                         )
        self.neck_btn.setText("neck")  # text
        self.neck_btn.setObjectName("neck_btn")
        self.horizontalLayout_4.addWidget(self.neck_btn)

        # FACE
        self.face_btn = QPushButton(MainWindow)
        self.face_btn.setStyleSheet("background: #9933FA;"  # QComboBox{background:Green};
                                         "font-size: 14pt;"
                                         "color:#FFE4E1;"
                                         )
        self.face_btn.setText("face")  # text
        self.face_btn.setObjectName("face_btn")
        self.horizontalLayout_4.addWidget(self.face_btn)


        # 美瞳
        self.eye_l_btn = QPushButton(MainWindow)
        self.eye_l_btn.setStyleSheet("background: #2E8B57;"  # QComboBox{background:Green};
                                    "font-size: 14pt;"
                                    "color:#FFE4E1;"
                                    )
        self.eye_l_btn.setText("eye_L")  # text
        self.eye_l_btn.setObjectName("eye_l_btn")
        self.horizontalLayout_4.addWidget(self.eye_l_btn)

        self.eye_r_btn = QPushButton(MainWindow)
        self.eye_r_btn.setStyleSheet("background: #2E8B57;"  # QComboBox{background:Green};
                                    "font-size: 14pt;"
                                    "color:#FFE4E1;"
                                    )
        self.eye_r_btn.setText("eye_R")  # text
        self.eye_r_btn.setObjectName("eye_r_btn")
        self.horizontalLayout_4.addWidget(self.eye_r_btn)

        # Clear
        self.clear_btn = QPushButton(MainWindow)
        self.clear_btn.setStyleSheet("background: #000000;"  # QComboBox{background:Green};
                                         "font-size: 14pt;"
                                         "color:#FFE4E1;"
                                         )
        self.clear_btn.setText("Clear")  # text
        self.clear_btn.setObjectName("clear_btn")
        self.horizontalLayout_4.addWidget(self.clear_btn)



        '''  下拉框
        # self.SelModel.addItem("")   # 0
        # # self.SelModel.addItem("")   # 1
        '''

        # 生成 mask 图片
        self.mask_file = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("./icon/Icons/3D.ico"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.mask_file.setIcon(icon1)
        self.mask_file.setObjectName("mask_file")
        self.RunProgram = QtWidgets.QAction(MainWindow)

        # 妆容迁移
        self.SCgan = QtWidgets.QAction(MainWindow)
        iconsc= QtGui.QIcon()
        iconsc.addPixmap(QtGui.QPixmap("./icon/Icons/Drawing App.ico"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.SCgan.setIcon(iconsc)
        self.SCgan.setObjectName("SCgan")
        self.RunProgram = QtWidgets.QAction(MainWindow)

        #　妆容图片
        self.SCGAN_img = QtWidgets.QAction(MainWindow)
        scgan_icon = QtGui.QIcon()
        scgan_icon.addPixmap(QtGui.QPixmap("./icon/面膜.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.SCGAN_img.setIcon(scgan_icon)
        self.SCGAN_img.setObjectName("SCGAN_img")
        self.RunProgram = QtWidgets.QAction(MainWindow)

        self.SelFile = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("./icon/Icons/Folder - Apps.ico"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.SelFile.setIcon(icon2)
        self.SelFile.setObjectName("SelFile")
        self.RunProgram = QtWidgets.QAction(MainWindow)

        icon3 = QtGui.QIcon()
        # icon3.addPixmap(QtGui.QPixmap("./icon/停止.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)  # 控件或窗口为使能但未激活状态
        icon3.addPixmap(QtGui.QPixmap("./icon/开始.png"),   QtGui.QIcon.Active, QtGui.QIcon.Off)
        # QIcon.On 当小部件处于“关闭”状态时显示 pixmap   QIcon.Off 当小部件处于“打开”状态时显示 pixmap
        # icon3.addPixmap(QtGui.QPixmap("./icon/停止.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        self.RunProgram.setIcon(icon3)
        self.RunProgram.setObjectName("RunProgram")

        self.cam_switch = QtWidgets.QAction(MainWindow)
        self.cam_switch.setEnabled(True)
        self.cam_switch.setCheckable(True)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("./icon/摄像头开.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon4.addPixmap(QtGui.QPixmap("./icon/摄像头关.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.cam_switch.setIcon(icon4)
        self.cam_switch.setObjectName("cam_switch")

        self.Exit = QtWidgets.QAction(MainWindow)
        self.Exit.setObjectName("Exit")
        # addAction() 添加具有文本或图标的工具按钮
        # addWidget() 添加工具栏中按钮以外的控件
        # addToolBar() 使用QMainWindow类的方法添加一个新的工具栏
        # setMovable()  工具变得可移动
        # setOrientation()  工具栏的方向可以设置为Qt.Horizontal或Qt.certical
        # self.toolBar.addWidget(self.SelModel)           # 添加 下拉框
        self.toolBar.addAction(self.SelFile)
        # self.toolBar.addAction(self.cam_switch)
        self.toolBar.addAction(self.SCGAN_img)
        self.toolBar.addAction(self.mask_file)
        self.toolBar.addAction(self.RunProgram)
        self.toolBar.addAction(self.SCgan)

        self.setWindowOpacity(0.98)                           # 设置窗口透明度
        # self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        self.gridLayout.setSpacing(10)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AI-虚拟化妆系统"))
        self.label_2.setText(_translate("MainWindow", "输入图"))
        self.makeup.setText(_translate("MainWindow", "妆容图"))
        self.label.setText(_translate("MainWindow", "效果图"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.SelFile.setText(_translate("MainWindow", "选择文件"))
        self.SelFile.setToolTip(_translate("MainWindow", "选择文件"))
        self.RunProgram.setText(_translate("MainWindow", "执行程序"))
        self.RunProgram.setToolTip(_translate("MainWindow", "执行程序"))
        self.cam_switch.setText(_translate("MainWindow", "摄像头开关"))
        self.cam_switch.setToolTip(_translate("MainWindow", "摄像头开关"))
        self.Exit.setText(_translate("MainWindow", "退出"))
