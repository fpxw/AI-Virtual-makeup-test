# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :setup.py
# @Time     :2021/12/8 18:02


import sys
sys.setrecursionlimit(100000)   #修改　递归深度
import os
from cx_Freeze import setup, Executable

# ADD FILES    'includes':[]
files = ['./icon/Icons/DVD-R.ico']

build_exe_options = {'packages': [], 'excludes': [], 'includes':['torch._VF', 'torch.distributions.constraints','torch.onnx.symbolic_opset7','torch.onnx.symbolic_opset8','torch.onnx.symbolic_opset12','torch.onnx.symbolic_opset13', 'torch.onnx.symbolic_opset14', 'keras.engine.base_layer_v1']}

# TARGET
target = Executable(
    script="makeup_main.py",
   # base="Win32GUI",
    icon="./icon/Icons/DVD-R.ico"
)

# SETUP CX FREEZE
setup(
    name="AI_Makeup",
    version="1.0",
    description="Modern GUI for Python applications",
    author="FuPX",
    options={'build_exe':build_exe_options},
    executables=[target]
)
