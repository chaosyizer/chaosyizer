'''
-*- coding: utf-8 -*-
@Time    : 2023/4/16 20:44
@Author  : cjk
@File    : DIPmain.py
'''
import math
import cmath
import sys

from ultralytics import YOLO

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QInputDialog, QMessageBox
from PyQt5.QtCore import Qt
from DIP import Ui_mainWindow

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 中文标题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Window(QMainWindow, Ui_mainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setupUi(self)  # 渲染页面控件
        self.connect_signals()  # 设置信号槽
        self.cvImagePath = None
        self.show_edit.setText("请先打开图片")


    def connect_signals(self):
        # self.btn_sure.clicked.connect(self.btn_sure_clicked) # 绑定确定按钮事件
        # self.btn_cancel.clicked.connect(self.btn_cancel_clicked) # 绑定取消按钮事件
        self.openImage.triggered.connect(self.openImage_clicked)
        self.actionhuiduzhifang.triggered.connect(self.grayhist)
        self.actionjunzhilvbo_2.triggered.connect(self.blur)
        self.actiontuxiangfanzhuan.triggered.connect(self.reversal)
        self.actionduishubianhuan.triggered.connect(self.logarithm)
        self.actiongamabianhuan.triggered.connect(self.gamma)
        self.actionduibidulashen.triggered.connect(self.stretch)
        self.actionhuidujifenceng.triggered.connect(self.Grayscale_layering)
        self.actionbitepingmianfenceng.triggered.connect(self.Bit_plane_layering)
        self.actionzhifangtujunheng.triggered.connect(self.Histogram_equalization)
        self.actiongaosilvbo.triggered.connect(self.GaussianBlur)
        self.actionlaplacian.triggered.connect(self.Laplacian)
        self.actionsobel.triggered.connect(self.Sobel)
        self.actionScharr.triggered.connect(self.Scharr)
        self.actionPassivation_masking.triggered.connect(self.Passivation_masking)
        self.actionpinyuditonglvbo.triggered.connect(self.LPF)
        self.actionpinyugaotonglvbo.triggered.connect(self.HPF)
        self.actionzhongzhilvbo.triggered.connect(self.Median_filtering)
        self.actionnixiebojunzhilvbo.triggered.connect(self.iHarBlur)
        self.actionzishiyingzhongzhilvbo.triggered.connect(self.Adaptive_median_filtering)
        self.actionyundongmohutuxiangfuyuan.triggered.connect(self.Motion_blur_image_restoration)
        self.actionfuzhi.triggered.connect(self.corrosion)
        self.actionpengzhang.triggered.connect(self.expansion)
        self.actionkai.triggered.connect(self.open)
        self.actionbi.triggered.connect(self.close)
        self.actiontuxiangbianyuanjiance.triggered.connect(self.edge_detection)
        self.actiontuxiangerzhihua.triggered.connect(self.thresh_binary)
        self.actionkjunzhi.triggered.connect(self.partition_kmean)
        self.actionshengzhang.triggered.connect(self.partition_grow)
        self.actionfenli.triggered.connect(self.partition_part)
        self.actionxingrenmubiaojiance.triggered.connect(self.Pedestrian_detection_video)
        self.actionimage.triggered.connect(self.Pedestrian_detection_image)

    def btn_sure_clicked(self):
        self.show_edit.setText('hello world')  # 点击确定按钮显示hello world

    def btn_cancel_clicked(self):
        self.show_edit.clear()  # 点击取消按钮清空显示框

    def openImage_clicked(self):
        # 弹出一个文件选择框，第一个返回值imgName记录选中的文件路径+文件名，第二个返回值imgType记录文件的类型
        imgPath, imgType = QFileDialog.getOpenFileName(self.centralwidget, "打开图片", " ",
                                                       "*.jpg;;*.png;;All Files(*)")
        if imgPath == '':
            pass  # 防止关闭或取消导入关闭所有页面
        else:
            self.show_edit.setText("路径：" + imgPath)
            self.cvImagePath = imgPath
            cvimg = cv2.imread(self.cvImagePath)
            showImage(self, cvimg)

    def cvimg2pixmap(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr -> rgb
        h, w, c = img.shape  # 获取图片形状
        image = QImage(img, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(image)

    def grayhist(self):
        """
        灰度直方图
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0
        # 读图
        img = cv2.imread(self.cvImagePath)
        # 转换成灰度图
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.figure()
        plt.title("灰度直方图")
        # 显示灰度图
        plt.subplot(1, 2, 1)
        plt.imshow(img2, cmap='gray')
        # 显示灰度直方图
        plt.subplot(1, 2, 2)
        # 获取直方图，由于灰度图img2是二维数组，需转换成一维数组
        plt.hist(img2.ravel(), 256)
        # 显示直方图
        plt.show()

    def blur(self):
        """
        均值滤波
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0
        number, ok = QInputDialog.getInt(self, "均值滤波", "卷积层 行数 1-10", min=1, max=10)
        if ok:
            line = number
        else:
            return 0
        number, ok = QInputDialog.getInt(self, "均值滤波", "卷积层 列数 1-10", min=1, max=10)
        if ok:
            column = number
        else:
            return 0
        img = cv2.imread(self.cvImagePath)
        dst = cv2.blur(img, (line, column))  # 卷积层
        plt.figure()
        plt.subplot(121),plt.title("Original"),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.subplot(122),plt.title("Blur"),plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        plt.show()

    def reversal(self):
        """
        图像反转
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0
        img = cv2.imread(self.cvImagePath)  # 读取彩色图像(BGR)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 颜色转换：BGR(OpenCV) -> Gray
        imgInv = 255 - imgGray
        plt.figure()
        plt.subplot(311), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("imgRGB"), plt.axis('off')
        plt.subplot(312), plt.imshow(imgGray, cmap='gray'), plt.title("imgGray"), plt.axis('off')
        plt.subplot(313), plt.imshow(imgInv, cmap='gray'), plt.title("imgInv"), plt.axis('off')
        plt.show()

    def logarithm(self):
        """
        对数变换
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0
        img = cv2.imread(self.cvImagePath, flags=0)  # flags=0 读取为灰度图像

        h, w = img.shape[0], img.shape[1]
        img_log = np.zeros((h, w))
        c = 1.0
        # number, ok = QInputDialog.getDouble(self, "对数变换", "c的值 0.1-5.0", min=0.1, max=5.0)
        # if ok:
        #     c = number
        # else:
        #     return 0
        for i in range(h):
            for j in range(w):
                img_log[i, j] = c * (math.log(1.0 + img[i, j]))

        # 映射到[0,255]
        cv2.normalize(img_log, img_log, 0, 255, cv2.NORM_MINMAX)

        fft = np.fft.fft2(img)  # 傅里叶变换
        fft_shift = np.fft.fftshift(fft)  # 中心化
        normImg = lambda x: 255. * (x - x.min()) / (x.max() - x.min() + 1e-6)  # 归一化

        magnitude_spectrum = np.uint8(normImg(np.abs(fft_shift)))

        fft_log = np.fft.fft2(img_log)  # 傅里叶变换
        fft_shift_log = np.fft.fftshift(fft_log)  # 中心化
        magnitude_spectrum_log = np.uint8(normImg(np.abs(fft_shift_log)))

        plt.figure()
        plt.subplot(221), plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Original_Gray'), plt.axis('off')
        plt.subplot(222), plt.imshow(img_log, cmap='gray', vmin=0, vmax=255), plt.title("After_Log"), plt.axis('off')
        plt.subplot(223), plt.imshow(8 * np.log(np.abs(fft_shift)), cmap='gray', vmin=0, vmax=255), plt.title(
            "Original_Gray_FFT"), plt.axis('off')
        plt.subplot(224), plt.imshow(8 * np.log(np.abs(fft_shift_log)), cmap='gray', vmin=0, vmax=255), plt.title(
            "After_Log_FFT"), plt.axis('off')
        plt.tight_layout()
        plt.show()

    def gamma(self):
        """
        幂律（伽马）变换
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        number, ok = QInputDialog.getDouble(self, "幂律（伽马）变换", "gamma值 0.04-25.0", min=0.04, max=25.0, decimals=2)
        if ok:
            gamma = number
        else:
            return 0
        number, ok = QInputDialog.getDouble(self, "幂律（伽马）变换", "c值 1-10", min=1, max=10)
        if ok:
            c = number
        else:
            return 0
        img = cv2.imread(self.cvImagePath, flags=0)  # flags=0 读取为灰度图像

        normImg = lambda x: 255. * (x - x.min()) / (x.max() - x.min() + 1e-6)  # 归一化为 [0,255]
        imgGamma = c * np.power(img, gamma)
        imgGamma = np.uint8(normImg(imgGamma))

        plt.figure()
        plt.subplot(121), plt.axis('off')
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.subplot(122), plt.axis('off')
        plt.imshow(imgGamma, cmap='gray', vmin=0, vmax=255)
        plt.title(f"$c={c}  \gamma={gamma}$")
        plt.show()

    def stretch(self):
        """
        对比度拉伸
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        imgGray = cv2.imread(self.cvImagePath, flags=0)  # flags=0 读取为灰度图像
        height, width = imgGray.shape[:2]  # 图片的高度和宽度

        number, ok = QInputDialog.getInt(self, "对比度拉伸 分三段", "第一折点横坐标r1 1-254", min=1, max=254)
        if ok:
            r1 = number
        else:
            return 0
        number, ok = QInputDialog.getInt(self, "对比度拉伸 分三段", "第一折点纵坐标s1 1-254", min=1, max=254)
        if ok:
            s1 = number
        else:
            return 0
        number, ok = QInputDialog.getInt(self, "对比度拉伸 分三段", f"第二折点横坐标r2 {r1+1}-255", min=r1+1, max=255)
        if ok:
            r2 = number
        else:
            return 0
        number, ok = QInputDialog.getInt(self, "对比度拉伸 分三段", f"第二折点横坐标s2 {s1+1}-255", min=s1+1, max=255)
        if ok:
            s2 = number
        else:
            return 0

        imgStretch = np.empty((height, width), np.uint8)  # 创建空白数组
        k1 = s1 / r1  # imgGray[h,w] < r1:
        k2 = (s2 - s1) / (r2 - r1)  # r1 <= imgGray[h,w] <= r2
        k3 = (255 - s2) / (255 - r2)  # imgGray[h,w] > r2
        for h in range(height):
            for w in range(width):
                if imgGray[h, w] < r1:
                    imgStretch[h, w] = k1 * imgGray[h, w]
                elif r1 <= imgGray[h, w] <= r2:
                    imgStretch[h, w] = k2 * (imgGray[h, w] - r1) + s1
                elif imgGray[h, w] > r2:
                    imgStretch[h, w] = k3 * (imgGray[h, w] - r2) + s2

        plt.figure()
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.8, wspace=0.1, hspace=0.1)
        plt.subplot(131), plt.title("s=T(r)")
        x = [0, r1, r2, 255]
        y = [0, s1, s2, 255]
        plt.plot(x, y)
        plt.axis([0, 256, 0, 256])
        plt.text(r1, s1, "(r1,s1)", fontsize=10)
        plt.text(r2, s2, "(r2,s2)", fontsize=10)
        plt.xlabel("r, Input value")
        plt.ylabel("s, Output value")
        plt.subplot(132), plt.imshow(imgGray, cmap='gray', vmin=0, vmax=255), plt.title("Original"), plt.axis('off')
        plt.subplot(133), plt.imshow(imgStretch, cmap='gray', vmin=0, vmax=255), plt.title("Stretch"), plt.axis('off')
        plt.show()

    def Grayscale_layering(self):
        """
        灰度级分层
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        imgGray = cv2.imread(self.cvImagePath, flags=0)  # flags=0 读取为灰度图像

        a, b = 155, 245  # 突出 [a, b] 区间的灰度
        number, ok = QInputDialog.getInt(self, "灰度级分层", "突出区域左边界 0-255", min=0, max=255)
        if ok:
            a = number
        else:
            return 0
        number, ok = QInputDialog.getInt(self, "灰度级分层", f"突出区域右边界 {a + 1}-255", min=a + 1, max=255)
        if ok:
            b = number
        else:
            return 0
        # Gray layered strategy 1: binary image
        imgLayer1 = imgGray.copy()
        imgLayer1[(imgLayer1[:, :] < a) | (imgLayer1[:, :] > b)] = 0  # 其它区域：黑色
        imgLayer1[(imgLayer1[:, :] >= a) & (imgLayer1[:, :] <= b)] = 255  # 灰度级窗口：白色

        # Gray layered strategy 2: grayscale image
        imgLayer2 = imgGray.copy()
        imgLayer2[(imgLayer2[:, :] >= a) & (imgLayer2[:, :] <= b)] = 255  # 灰度级窗口：白色，其它区域不变

        plt.figure()
        plt.subplot(231), plt.imshow(imgGray, cmap='gray'), plt.title('Original'), plt.axis('off')
        plt.subplot(232), plt.imshow(imgLayer1, cmap='gray'), plt.title('Binary layered'), plt.axis('off')
        plt.subplot(233), plt.imshow(imgLayer2, cmap='gray'), plt.title('Grayscale layered'), plt.axis('off')

        plt.subplot(234), plt.hist(imgGray.ravel(), 256), plt.title('Original hist')
        plt.subplot(235), plt.hist(imgLayer1.ravel(), 256), plt.title('Binary layered hist')
        plt.subplot(236), plt.hist(imgLayer2.ravel(), 256), plt.title('Grayscale layered hist')
        plt.show()

    def Bit_plane_layering(self):
        """
        比特平面分层
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        img = cv2.imread(self.cvImagePath, flags=0)  # flags=0 读取为灰度图像
        height, width = img.shape[:2]  # 图片的高度和宽度

        plt.figure()
        for l in range(9, 0, -1):
            plt.subplot(3, 3, (9 - l) + 1, xticks=[], yticks=[])
            if l == 9:
                plt.imshow(img, cmap='gray'), plt.title('Original')
            else:
                imgBit = np.empty((height, width), dtype=np.uint8)  # 创建空数组
                for w in range(width):
                    for h in range(height):
                        x = np.binary_repr(img[h, w], width=8)  # 以字符串形式返回输入数字的二进制表示形式
                        x = x[::-1]
                        a = x[l - 1]
                        imgBit[h, w] = int(a)  # 第 i 位二进制的值
                plt.imshow(imgBit, cmap='gray')
                plt.title(f"{l - 1}")
        plt.show()

    def Histogram_equalization(self):
        """
        直方图均衡
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        img = cv2.imread(self.cvImagePath, flags=0)  # flags=0 读取为灰度图像
        imgEqu = cv2.equalizeHist(img)  # 使用 cv2.qualizeHist 完成直方图均衡化变换

        fig = plt.figure(figsize=(7, 7))
        plt.subplot(221), plt.title("Original image "), plt.axis('off')
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)  # 原始图像
        plt.subplot(222), plt.title("Hist-equalized image"), plt.axis('off')
        plt.imshow(imgEqu, cmap='gray', vmin=0, vmax=255)  # 转换图像
        histImg, bins = np.histogram(img.flatten(), 256)  # 计算原始图像直方图
        plt.subplot(223, yticks=[]), plt.bar(bins[:-1], histImg)  # 原始图像直方图
        plt.title("Histogram of original image"), plt.axis([0, 255, 0, np.max(histImg)])
        histEqu, bins = np.histogram(imgEqu.flatten(), 256)  # 计算原始图像直方图
        plt.subplot(224, yticks=[]), plt.bar(bins[:-1], histEqu)  # 转换图像直方图
        plt.title("Histogram of equalized image"), plt.axis([0, 255, 0, np.max(histImg)])
        plt.show()

    def GaussianBlur(self):
        """
        高斯滤波
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0
        img = cv2.imread(self.cvImagePath, flags=0)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        size = img.shape
        mean = 0
        sigma = 0
        number, ok = QInputDialog.getInt(self, "高斯滤波前的加高斯噪声", "均值 0-100", min=0, max=100)
        if ok:
            mean = number
        else:
            return 0
        number, ok = QInputDialog.getInt(self, "高斯滤波前的加高斯噪声", "方差 0-400", min=0, max=400)
        if ok:
            sigma = number
        else:
            return 0

        gauss = np.random.normal(mean, sigma, size)
        img_noise = img + gauss

        while True:
            number, ok = QInputDialog.getInt(self, "高斯滤波卷积核大小", "高度宽度都为奇数 1、3、5、7、9", min=1, max=9)
            if ok:
                if number % 2 == 1:
                    k = number
                    break
                else:
                    QMessageBox.information(self, "提示", "请输入奇数！")
                    continue
            else:
                return 0

        img_gaussianBlur = cv2.GaussianBlur(img_noise, (k, k), 0)

        plt.figure()
        plt.subplot(131)
        plt.title("Original image")
        plt.imshow(img, cmap='gray')
        plt.subplot(132)
        plt.title("Gaussian noised image")
        plt.imshow(img_noise, cmap='gray')
        plt.subplot(133)
        plt.title("Gaussian blured image")
        plt.imshow(img_gaussianBlur, cmap='gray')

        plt.show()

    def Laplacian(self):
        """
        Laplacian图像锐化
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0
        # 图像锐化：拉普拉斯算子 (Laplacian)
        img = cv2.imread(self.cvImagePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 使用函数 filter2D 实现 Laplace 卷积算子
        kernLaplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Laplacian kernel
        imgLaplace1 = cv2.filter2D(img, -1, kernLaplace, borderType=cv2.BORDER_REFLECT)

        # 使用 cv2.Laplacian 实现 Laplace 卷积算子
        imgLaplace2 = cv2.Laplacian(img, -1, ksize=3)
        imgRecovery = cv2.add(img, imgLaplace2)  # 恢复原图像

        plt.figure(figsize=(9, 6))
        plt.subplot(131), plt.axis('off'), plt.title("Original")
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.subplot(132), plt.axis('off'), plt.title("Laplacian_edge")
        plt.imshow(imgLaplace2, cmap='gray', vmin=0, vmax=255)
        plt.subplot(133), plt.axis('off'), plt.title("Laplacian_sharpened")
        plt.imshow(imgRecovery, cmap='gray', vmin=0, vmax=255)
        plt.tight_layout()
        plt.show()

    def Sobel(self):
        """
        Sobel图像锐化
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        img = cv2.imread(self.cvImagePath, flags=0)

        # 使用函数 filter2D 实现 Sobel 算子
        kernSobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # SobelX kernel
        kernSobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # SobelY kernel
        imgSobelX = cv2.filter2D(img, -1, kernSobelX, borderType=cv2.BORDER_REFLECT)
        imgSobelY = cv2.filter2D(img, -1, kernSobelY, borderType=cv2.BORDER_REFLECT)

        # 使用 cv2.Sobel 实现 Sobel 算子
        SobelX = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 计算 x 轴方向
        SobelY = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 计算 y 轴方向
        absX = cv2.convertScaleAbs(SobelX)  # 转回 uint8
        absY = cv2.convertScaleAbs(SobelY)  # 转回 uint8
        SobelXY = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 用绝对值近似平方根
        imgRecovery = cv2.add(img, SobelXY)  # 增强原图像

        plt.figure(figsize=(10, 6))
        plt.subplot(131), plt.axis('off'), plt.title("Original")
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.subplot(132), plt.axis('off'), plt.title("Sobel_edge")
        plt.imshow(SobelXY, cmap='gray', vmin=0, vmax=255)
        plt.subplot(133), plt.axis('off'), plt.title("Sobel_sharpened")
        plt.imshow(imgRecovery, cmap='gray', vmin=0, vmax=255)

        plt.tight_layout()
        plt.show()

    def Scharr(self):
        """
        Scharr图像锐化
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0
        # 图像锐化：Scharr 算子
        img = cv2.imread(self.cvImagePath, flags=0)

        # 使用函数 filter2D 实现 Scharr 算子
        kernScharrX = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])  # ScharrX kernel
        kernScharrY = np.array([[-3, 10, -3], [0, 0, 10], [3, 10, 3]])  # ScharrY kernel

        # 使用 cv2.Scharr 实现 Scharr 算子
        ScharrX = cv2.Scharr(img, cv2.CV_16S, 1, 0)  # 计算 x 轴方向
        ScharrY = cv2.Scharr(img, cv2.CV_16S, 0, 1)  # 计算 y 轴方向
        absX = cv2.convertScaleAbs(ScharrX)  # 转回 uint8
        absY = cv2.convertScaleAbs(ScharrY)  # 转回 uint8
        ScharrXY = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 用绝对值近似平方根
        imgRecovery = cv2.add(img, ScharrXY)  # 增强原图像
        plt.figure(figsize=(10, 6))
        plt.subplot(131), plt.axis('off'), plt.title("Original")
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.subplot(132), plt.axis('off'), plt.title("Scharr_edge")
        plt.imshow(ScharrXY, cmap='gray', vmin=0, vmax=255)
        plt.subplot(133), plt.axis('off'), plt.title("Scharr_sharpened")
        plt.imshow(imgRecovery, cmap='gray', vmin=0, vmax=255)

        plt.tight_layout()
        plt.show()

    def Passivation_masking(self):
        """
        Scharr图像锐化
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0
        # 图像锐化: 钝化掩蔽
        img = cv2.imread(self.cvImagePath, flags=0)

        # 对原始图像进行平滑，GaussianBlur(img, size, sigmaX)
        imgGauss = cv2.GaussianBlur(img, (5, 5), sigmaX=5)
        imgGaussNorm = cv2.normalize(imgGauss, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # 掩蔽模板：从原始图像中减去平滑图像
        imgMask = img - imgGaussNorm

        passivation1 = img + 0.6 * imgMask  # k<1 减弱钝化掩蔽
        imgPas1 = cv2.normalize(passivation1, None, 0, 255, cv2.NORM_MINMAX)
        passivation2 = img + imgMask  # k=1 钝化掩蔽
        imgPas2 = cv2.normalize(passivation2, None, 0, 255, cv2.NORM_MINMAX)
        passivation3 = img + 2 * imgMask  # k>1 高提升滤波
        imgPas3 = cv2.normalize(passivation3, None, 0, 255, cv2.NORM_MINMAX)

        plt.figure()
        # titleList = ["1. Original", "2. GaussSmooth", "3. MaskTemplate",
        #              "4. Passivation(k=0.5)", "5. Passivation(k=1.0)", "6. Passivation(k=2.0)"]
        # imageList = [img, imgGauss, imgMask, imgPas1, imgPas2, imgPas3]
        # for i in range(6):
        #     plt.subplot(2, 4, i + 1), plt.title(titleList[i]), plt.axis('off')
        #     plt.imshow(imageList[i], 'gray', vmin=0, vmax=255)

        plt.subplot(221)
        plt.title("Original")
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.subplot(222)
        plt.title("GaussSmooth")
        plt.imshow(imgGauss, cmap='gray', vmin=0, vmax=255)
        plt.subplot(223)
        plt.title("Mask template")
        plt.imshow(imgMask, cmap='gray', vmin=0, vmax=255)
        plt.subplot(224)
        plt.title("Passivation masked")
        plt.imshow(imgPas2, cmap='gray', vmin=0, vmax=255)

        plt.tight_layout()
        plt.show()

    def LPF(self):
        """
        低通滤波
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0
        gray = cv2.imread(self.cvImagePath, flags=0)

        h, w = gray.shape

        for i in range(3000):  # 添加3000个噪声点
            x = np.random.randint(0, h)
            y = np.random.randint(0, w)
            gray[x, y] = 255

        # 傅里叶变换
        img_dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(img_dft)  # 将频域从左上角移动到中间

        # 低通滤波
        dft_shift = lowPassFiltering(dft_shift, 200)
        res = np.log(np.abs(dft_shift))

        # 傅里叶逆变换
        idft_shift = np.fft.ifftshift(dft_shift)  # 将频域从中间移动到左上角
        ifimg = np.fft.ifft2(idft_shift)  # 傅里叶库函数调用
        ifimg = np.abs(ifimg)
        # cv2.imshow("ifimg", np.int8(ifimg))
        # cv2.imshow("gray", gray)

        # 绘制图片
        plt.subplot(131), plt.imshow(gray, 'gray'), plt.title('原图像')
        plt.axis('off')
        plt.subplot(132), plt.imshow(res, 'gray'), plt.title('低通滤波')
        plt.axis('off')
        plt.subplot(133), plt.imshow(np.int8(ifimg), 'gray'), plt.title('滤波后效果')
        plt.axis('off')
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def HPF(self):
        """
        高通滤波
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0
        img = cv2.imread(self.cvImagePath, flags=0)
        # 傅里叶变换
        img_dft = np.fft.fft2(img)
        dft_shift = np.fft.fftshift(img_dft)  # 将频域从左上角移动到中间

        # 高通滤波
        dft_shift = highPassFiltering(dft_shift, 200)
        res = np.log(np.abs(dft_shift))

        # 傅里叶逆变换
        idft_shift = np.fft.ifftshift(dft_shift)  # 将频域从中间移动到左上角
        ifimg = np.fft.ifft2(idft_shift)  # 傅里叶库函数调用
        ifimg = np.abs(ifimg)

        # 绘制图片
        plt.subplot(131), plt.imshow(img, 'gray'), plt.title('原图像')
        plt.axis('off')
        plt.subplot(132), plt.imshow(res, 'gray'), plt.title('高通滤波')
        plt.axis('off')
        plt.subplot(133), plt.imshow(np.int8(ifimg), 'gray'), plt.title('滤波后效果')
        plt.axis('off')
        plt.show()

    def Median_filtering(self):
        """
        中值滤波
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        img = cv2.imread(self.cvImagePath, flags=1)
        while True:
            number, ok = QInputDialog.getInt(self, "中值滤波", "Size 0-10 奇数", min=0, max=10)
            if ok:
                if number % 2 == 1:
                    size = number
                    break
                else:
                    QMessageBox.information(self, "提示", "请输入奇数")
                    continue
            else:
                return 0
        imgMedianBlur = cv2.medianBlur(img, size)
        plt.figure()
        plt.subplot(121), plt.axis('off'), plt.title("Original")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.subplot(122), plt.axis('off'), plt.title(f"medianBlur(size={size})")
        plt.imshow(cv2.cvtColor(imgMedianBlur, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()

    def iHarBlur(self):
        """
        逆谐波均值滤波
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        img = cv2.imread(self.cvImagePath, 0)

        while True:
            number, ok = QInputDialog.getInt(self, "逆谐波均值变换", "Kernel Size 0-9 奇数", min=0, max=9)
            if ok:
                if number % 2 == 1:
                    kernel_size = number
                    QMessageBox.information(self, "提示", "该功能运行速度较慢，请耐心等待")
                    break
                else:
                    QMessageBox.information(self, "提示", "请输入奇数")
                    continue
            else:
                return 0

        print(1)
        G_mean_img = np.zeros(img.shape)
        k = int((kernel_size - 1) / 2)
        Q = -1.5
        print(2)
        self.show_edit.setText("处理中，请耐心等待，可能会出现“未响应”")
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if i < k or i > (img.shape[0] - k - 1) or j < k or j > (img.shape[1] - k - 1):
                    G_mean_img[i][j] = img[i][j]
                else:
                    result_top = 0
                    result_down = 0
                    for n in range(kernel_size):
                        for m in range(kernel_size):
                            if Q > 0:
                                result_top += pow(np.float(img[i - k + n][j - k + m]), Q + 1)
                                result_down += pow(np.float(img[i - k + n][j - k + m]), Q)
                            else:
                                if img[i - k + n][j - k + m] == 0:
                                    G_mean_img[i][j] = 0
                                    break
                                else:
                                    result_top += pow(np.float64(img[i - k + n][j - k + m]), Q + 1)
                                    result_down += pow(np.float64(img[i - k + n][j - k + m]), Q)
                        else:
                            continue
                        break

                    else:
                        if result_down != 0:
                            G_mean_img[i][j] = result_top / result_down
        self.show_edit.setText("处理完毕")
        print(3)
        G_mean_img = np.uint8(G_mean_img)

        plt.figure()
        plt.subplot(121)
        plt.title("原图")
        plt.imshow(img, cmap='gray')
        plt.subplot(122)
        plt.title("逆谐波均值滤波")
        plt.imshow(G_mean_img, cmap='gray')
        plt.show()

    def Adaptive_median_filtering(self):
        """
        自适应中值滤波
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0
        img = cv2.imread(self.cvImagePath, 0)  # flags=0 读取为灰度图像
        # 法一
        # 应用自适应中值滤波
        # imgAdaMedFilter = adaptive_median_filter(img, 7)
        hImg = img.shape[0]
        wImg = img.shape[1]

        smax = 7  # 允许最大窗口尺寸
        m, n = smax, smax
        imgAriMean = cv2.boxFilter(img, -1, (m, n))  # 算术平均滤波

        # 边缘填充
        hPad = int((m - 1) / 2)
        wPad = int((n - 1) / 2)
        imgPad = np.pad(img.copy(), ((hPad, m - hPad - 1), (wPad, n - wPad - 1)), mode="edge")

        imgMedianFilter = np.zeros(img.shape)  # 中值滤波器
        imgAdaMedFilter = np.zeros(img.shape)  # 自适应中值滤波器
        for i in range(hPad, hPad + hImg):
            for j in range(wPad, wPad + wImg):
                # # 1. 中值滤波器 (Median filter)
                # ksize = 3
                # k = int(ksize / 2)
                # pad = imgPad[i - k:i + k + 1, j - k:j + k + 1]  # 邻域 Sxy, m*n
                # imgMedianFilter[i - hPad, j - wPad] = np.median(pad)

                # 2. 自适应中值滤波器 (Adaptive median filter)
                ksize = 3
                k = int(ksize / 2)
                pad = imgPad[i - k:i + k + 1, j - k:j + k + 1]
                zxy = img[i - hPad][j - wPad]
                zmin = np.min(pad)
                zmed = np.median(pad)
                zmax = np.max(pad)

                if zmin < zmed < zmax:
                    if zmin < zxy < zmax:
                        imgAdaMedFilter[i - hPad, j - wPad] = zxy
                    else:
                        imgAdaMedFilter[i - hPad, j - wPad] = zmed
                else:
                    while True:
                        ksize = ksize + 2
                        if zmin < zmed < zmax or ksize > smax:
                            break
                        k = int(ksize / 2)
                        pad = imgPad[i - k:i + k + 1, j - k:j + k + 1]
                        zmed = np.median(pad)
                        zmin = np.min(pad)
                        zmax = np.max(pad)
                    if zmin < zmed < zmax or ksize > smax:
                        if zmin < zxy < zmax:
                            imgAdaMedFilter[i - hPad, j - wPad] = zxy
                        else:
                            imgAdaMedFilter[i - hPad, j - wPad] = zmed

        plt.figure()
        plt.subplot(121), plt.axis('off'), plt.title("Original")
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        # plt.subplot(132), plt.axis('off'), plt.title("Median filter")
        # plt.imshow(imgMedianFilter, cmap='gray', vmin=0, vmax=255)
        plt.subplot(122), plt.axis('off'), plt.title("Adaptive median filter")
        plt.imshow(imgAdaMedFilter, cmap='gray', vmin=0, vmax=255)
        plt.tight_layout()
        plt.show()

    def Motion_blur_image_restoration(self):
        """
        运动模糊图像复原
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        # 读取原始图像
        img = cv2.imread(self.cvImagePath)

        def getMotionDsf(shape, angle, dist):
            xCenter = (shape[0] - 1) / 2
            yCenter = (shape[1] - 1) / 2
            sinVal = np.sin(angle * np.pi / 180)
            cosVal = np.cos(angle * np.pi / 180)
            PSF = np.zeros(shape)  # 点扩散函数
            for i in range(dist):  # 将对应角度上motion_dis个点置成1
                xOffset = round(sinVal * i)
                yOffset = round(cosVal * i)
                PSF[int(xCenter - xOffset), int(yCenter + yOffset)] = 1
            return PSF / PSF.sum()  # 归一化

        def makeBlurred(image, PSF, eps):  # 对图片进行运动模糊
            fftImg = np.fft.fft2(image)  # 进行二维数组的傅里叶变换
            fftPSF = np.fft.fft2(PSF) + eps
            fftBlur = np.fft.ifft2(fftImg * fftPSF)
            fftBlur = np.abs(np.fft.fftshift(fftBlur))
            return fftBlur

        def inverseFilter(image, PSF, eps):  # 逆滤波
            fftImg = np.fft.fft2(image)
            fftPSF = np.fft.fft2(PSF) + eps  # 噪声功率，这是已知的，考虑epsilon
            imgInvFilter = np.fft.ifft2(fftImg / fftPSF)  # 计算F(u,v)的傅里叶反变换
            imgInvFilter = np.abs(np.fft.fftshift(imgInvFilter))
            return imgInvFilter

        def wienerFilter(input, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
            fftImg = np.fft.fft2(input)
            fftPSF = np.fft.fft2(PSF) + eps
            fftWiener = np.conj(fftPSF) / (np.abs(fftPSF) ** 2 + K)
            imgWienerFilter = np.fft.ifft2(fftImg * fftWiener)
            imgWienerFilter = np.abs(np.fft.fftshift(imgWienerFilter))
            return imgWienerFilter

        # 读取原始图像
        img = cv2.imread(self.cvImagePath, 0)
        hImg, wImg = img.shape[:2]

        # 不含噪声的运动模糊
        PSF = getMotionDsf((hImg, wImg), 45, 100)  # 运动模糊函数
        imgBlurred = np.abs(makeBlurred(img, PSF, 1e-6))  # 生成不含噪声的运动模糊图像
        imgInvFilter = inverseFilter(imgBlurred, PSF, 1e-6)  # 逆滤波
        imgWienerFilter = wienerFilter(imgBlurred, PSF, 1e-6)  # 维纳滤波

        # 带有噪声的运动模糊
        scale = 0.05  # 噪声方差
        noisy = imgBlurred.std() * np.random.normal(loc=0.0, scale=scale, size=imgBlurred.shape)  # 添加高斯噪声
        imgBlurNoisy = imgBlurred + noisy  # 带有噪声的运动模糊
        imgNoisyInv = inverseFilter(imgBlurNoisy, PSF, scale)  # 对添加噪声的模糊图像进行逆滤波
        imgNoisyWiener = wienerFilter(imgBlurNoisy, PSF, scale)  # 对添加噪声的模糊图像进行维纳滤波

        plt.figure(figsize=(9, 7))
        plt.subplot(231), plt.title("blurred image"), plt.axis('off'), plt.imshow(imgBlurred, 'gray')
        plt.subplot(232), plt.title("inverse filter"), plt.axis('off'), plt.imshow(imgInvFilter, 'gray')
        plt.subplot(233), plt.title("Wiener filter"), plt.axis('off'), plt.imshow(imgWienerFilter, 'gray')
        plt.subplot(234), plt.title("blurred image with noisy"), plt.axis('off'), plt.imshow(imgBlurNoisy, 'gray')
        plt.subplot(235), plt.title("inverse filter"), plt.axis('off'), plt.imshow(imgNoisyInv, 'gray')
        plt.subplot(236), plt.title("Wiener filter"), plt.axis('off'), plt.imshow(imgNoisyWiener, 'gray')
        plt.tight_layout()
        plt.show()

    def corrosion(self):
        """
        腐蚀
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        # 读取原始图像
        imgGray = cv2.imread(self.cvImagePath, flags=0)  # flags=0 读取为灰度图像
        ret, imgBin = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 二值化处理

        # 图像腐蚀
        kSize = (3, 3)  # 卷积核的尺寸
        kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
        imgErode1 = cv2.erode(imgBin, kernel=kernel)  # 图像腐蚀

        kSize = (9, 9)
        kernel = np.ones(kSize, dtype=np.uint8)
        imgErode2 = cv2.erode(imgBin, kernel=kernel)

        kSize = (25, 25)
        kernel = np.ones(kSize, dtype=np.uint8)
        imgErode3 = cv2.erode(imgBin, kernel=kernel)

        plt.figure(figsize=(10, 5))
        plt.subplot(141), plt.axis('off'), plt.title("Origin")
        plt.imshow(imgBin, cmap='gray', vmin=0, vmax=255)
        plt.subplot(142), plt.title("eroded kSize=(3,3)"), plt.axis('off')
        plt.imshow(imgErode1, cmap='gray', vmin=0, vmax=255)
        plt.subplot(143), plt.title("eroded kSize=(9,9)"), plt.axis('off')
        plt.imshow(imgErode2, cmap='gray', vmin=0, vmax=255)
        plt.subplot(144), plt.title("eroded kSize=(25,25)"), plt.axis('off')
        plt.imshow(imgErode3, cmap='gray', vmin=0, vmax=255)
        plt.tight_layout()
        plt.show()

    def expansion(self):
        """
        膨胀
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        # 读取原始图像
        imgGray = cv2.imread(self.cvImagePath, flags=0)  # flags=0 读取为灰度图像
        ret, imgBin = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 二值化处理

        # 图像膨胀
        kSize = (3, 3)  # 卷积核的尺寸
        kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
        imgDilate1 = cv2.dilate(imgBin, kernel=kernel)  # 图像膨胀

        kSize = (5, 5)
        kernel = np.ones(kSize, dtype=np.uint8)
        imgDilate2 = cv2.dilate(imgBin, kernel=kernel)  # 图像膨胀

        kSize = (7, 7)
        kernel = np.ones(kSize, dtype=np.uint8)
        imgDilate3 = cv2.dilate(imgBin, kernel=kernel)  # 图像膨胀

        plt.figure(figsize=(10, 5))
        plt.subplot(141), plt.axis('off'), plt.title("Origin")
        plt.imshow(imgBin, cmap='gray', vmin=0, vmax=255)
        plt.subplot(142), plt.title("dilate kSize=(3,3)"), plt.axis('off')
        plt.imshow(imgDilate1, cmap='gray', vmin=0, vmax=255)
        plt.subplot(143), plt.title("dilate kSize=(5,5)"), plt.axis('off')
        plt.imshow(imgDilate2, cmap='gray', vmin=0, vmax=255)
        plt.subplot(144), plt.title("dilate kSize=(7,7)"), plt.axis('off')
        plt.imshow(imgDilate3, cmap='gray', vmin=0, vmax=255)
        plt.tight_layout()
        plt.show()

    def open(self):
        """
        开运算（先腐蚀后膨胀）
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        # 读取原始图像
        imgGray = cv2.imread(self.cvImagePath, flags=0)  # flags=0 读取为灰度图像
        ret, imgBin = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 二值化处理

        # 图像腐蚀
        kSize = (3, 3)  # 卷积核的尺寸
        kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
        imgErode = cv2.erode(imgBin, kernel=kernel)  # 图像腐蚀

        # 图像的开运算
        kSize = (3, 3)  # 卷积核的尺寸
        kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
        imgOpen = cv2.morphologyEx(imgGray, cv2.MORPH_OPEN, kernel)

        plt.figure(figsize=(9, 5))
        plt.subplot(131), plt.axis('off'), plt.title("Origin")
        plt.imshow(imgGray, cmap='gray', vmin=0, vmax=255)
        plt.subplot(132), plt.title("Eroded kSize=(3,3)"), plt.axis('off')
        plt.imshow(imgErode, cmap='gray', vmin=0, vmax=255)
        plt.subplot(133), plt.title("Opening kSize=(3,3)"), plt.axis('off')
        plt.imshow(imgOpen, cmap='gray', vmin=0, vmax=255)
        plt.tight_layout()
        plt.show()

    def close(self):
        """
        闭运算（先膨胀后腐蚀）
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        # 10.4 图像的闭运算 (cv.morphologyEx)
        # 读取原始图像
        imgGray = cv2.imread(self.cvImagePath, flags=0)  # flags=0 读取为灰度图像
        mu, sigma = 0.0, 10.0
        noiseGause = np.random.normal(mu, sigma, imgGray.shape)
        imgNoisy = imgGray + noiseGause
        imgNoisy = np.uint8(cv2.normalize(imgNoisy, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
        ret, imgBin = cv2.threshold(imgNoisy, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 二值化处理

        # 图像的闭运算
        kSize = (2, 2)  # 卷积核的尺寸
        kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
        imgClose1 = cv2.morphologyEx(imgBin, cv2.MORPH_CLOSE, kernel)

        kSize = (3, 3)  # 卷积核的尺寸
        kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
        imgClose2 = cv2.morphologyEx(imgBin, cv2.MORPH_CLOSE, kernel)

        kSize = (5, 5)  # 卷积核的尺寸
        kernel = np.ones(kSize, dtype=np.uint8)  # 生成盒式卷积核
        imgClose3 = cv2.morphologyEx(imgBin, cv2.MORPH_CLOSE, kernel)

        plt.figure(figsize=(10, 5))
        plt.subplot(141), plt.axis('off'), plt.title("Origin")
        plt.imshow(imgNoisy, cmap='gray', vmin=0, vmax=255)
        plt.subplot(142), plt.title("Closed kSize=(2,2)"), plt.axis('off')
        plt.imshow(imgClose1, cmap='gray', vmin=0, vmax=255)
        plt.subplot(143), plt.title("Closed kSize=(3,3)"), plt.axis('off')
        plt.imshow(imgClose2, cmap='gray', vmin=0, vmax=255)
        plt.subplot(144), plt.title("Closed kSize=(5,5)"), plt.axis('off')
        plt.imshow(imgClose3, cmap='gray', vmin=0, vmax=255)
        plt.tight_layout()
        plt.show()

    def edge_detection(self):
        """
        边缘检测
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        img = cv2.imread(self.cvImagePath, flags=0)  # 读取为灰度图像

        # 自定义卷积核
        # Roberts 边缘算子
        kernel_Roberts_x = np.array([[1, 0], [0, -1]])
        kernel_Roberts_y = np.array([[0, -1], [1, 0]])
        # Prewitt 边缘算子
        kernel_Prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_Prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        # Sobel 边缘算子
        kernel_Sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_Sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # Laplacian 边缘算子
        kernel_Laplacian_K1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        kernel_Laplacian_K2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

        # 卷积运算
        imgBlur = cv2.blur(img, (3, 3))  # Blur 平滑后再做 Laplacian 变换
        imgLaplacian_K1 = cv2.filter2D(imgBlur, -1, kernel_Laplacian_K1)
        imgLaplacian_K2 = cv2.filter2D(imgBlur, -1, kernel_Laplacian_K2)
        imgRoberts_x = cv2.filter2D(img, -1, kernel_Roberts_x)
        imgRoberts_y = cv2.filter2D(img, -1, kernel_Roberts_y)
        imgRoberts = np.uint8(cv2.normalize(abs(imgRoberts_x) + abs(imgRoberts_y), None, 0, 255, cv2.NORM_MINMAX))
        imgPrewitt_x = cv2.filter2D(img, -1, kernel_Prewitt_x)
        imgPrewitt_y = cv2.filter2D(img, -1, kernel_Prewitt_y)
        imgPrewitt = np.uint8(cv2.normalize(abs(imgPrewitt_x) + abs(imgPrewitt_y), None, 0, 255, cv2.NORM_MINMAX))
        imgSobel_x = cv2.filter2D(img, -1, kernel_Sobel_x)
        imgSobel_y = cv2.filter2D(img, -1, kernel_Sobel_y)
        imgSobel = np.uint8(cv2.normalize(abs(imgSobel_x) + abs(imgSobel_y), None, 0, 255, cv2.NORM_MINMAX))

        # Canny 边缘检测， kSize 为高斯核大小，t1,t2为阈值大小
        kSize = (5, 5)
        imgGauss = cv2.GaussianBlur(img, kSize, sigmaX=1.0)  # sigma=1.0
        t1, t2 = 50, 150
        imgCanny = cv2.Canny(imgGauss, t1, t2)

        plt.figure(figsize=(12, 8))
        plt.subplot(231), plt.title('Origin'), plt.imshow(img, cmap='gray'), plt.axis('off')
        # plt.subplot(232), plt.title('Laplacian_K1'), plt.imshow(imgLaplacian_K1, cmap='gray'), plt.axis('off')
        plt.subplot(232), plt.title('Laplacian'), plt.imshow(imgLaplacian_K2, cmap='gray'), plt.axis('off')
        plt.subplot(233), plt.title('Roberts'), plt.imshow(imgRoberts, cmap='gray'), plt.axis('off')
        # plt.subplot(346), plt.title('Roberts_X'), plt.imshow(imgRoberts_x, cmap='gray'), plt.axis('off')
        # plt.subplot(3, 4, 10), plt.title('Roberts_Y'), plt.imshow(imgRoberts_y, cmap='gray'), plt.axis('off')
        plt.subplot(234), plt.title('Prewitt'), plt.imshow(imgPrewitt, cmap='gray'), plt.axis('off')
        # plt.subplot(347), plt.title('Prewitt_X'), plt.imshow(imgPrewitt_x, cmap='gray'), plt.axis('off')
        # plt.subplot(3, 4, 11), plt.title('Prewitt_Y'), plt.imshow(imgPrewitt_y, cmap='gray'), plt.axis('off')
        plt.subplot(235), plt.title('Sobel'), plt.imshow(imgSobel, cmap='gray'), plt.axis('off')
        plt.subplot(236), plt.title('Canny'), plt.imshow(imgCanny, cmap='gray'), plt.axis('off')
        # plt.subplot(348), plt.title('Sobel_X'), plt.imshow(imgSobel_x, cmap='gray'), plt.axis('off')
        # plt.subplot(3, 4, 12), plt.title('Sobel_Y'), plt.imshow(imgSobel_y, cmap='gray'), plt.axis('off')
        plt.tight_layout()
        plt.show()

    def thresh_binary(self):
        """
        图像二值化
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        img = cv2.imread(self.cvImagePath, flags=0)  # 读取为灰度图像
        number, ok = QInputDialog.getInt(self, "图像二值化阈值，小者0，大者255", "阈值 0-255", min=0, max=255)
        if ok:
            thresh = number
        else:
            return 0
        ret1, img_binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        ret2, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        plt.figure()
        plt.subplot(221), plt.title("Original"), plt.axis('off'), plt.imshow(img, 'gray')
        plt.subplot(222), plt.title(f"Binary_By_User(thresh={thresh})"), plt.axis('off'), plt.imshow(img_binary, 'gray')
        plt.subplot(223), plt.title("Original_hist"), plt.hist(img.ravel(), 256)
        plt.subplot(224), plt.title(f"OTSU(thresh={ret2})"), plt.axis('off'), plt.imshow(img_otsu, 'gray')
        plt.show()

    def partition_kmean(self):
        """
        图像区域分割kmean
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        img = cv2.imread(self.cvImagePath, flags=1)  # 读取彩色图像(BGR)

        dataPixel = np.float32(img.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)  # 终止条件
        flags = cv2.KMEANS_RANDOM_CENTERS  # 起始的中心选择

        K = 3  # 设置聚类数
        _, labels, center = cv2.kmeans(dataPixel, K, None, criteria, 10, flags)
        centerUint = np.uint8(center)
        classify = centerUint[labels.flatten()]  # 将像素标记为聚类中心颜色
        imgKmean3 = classify.reshape((img.shape))  # 恢复为二维图像

        K = 4  # 设置聚类数
        _, labels, center = cv2.kmeans(dataPixel, K, None, criteria, 10, flags)
        centerUint = np.uint8(center)
        classify = centerUint[labels.flatten()]  # 将像素标记为聚类中心颜色
        imgKmean4 = classify.reshape((img.shape))  # 恢复为二维图像

        K = 5  # 设置聚类数
        _, labels, center = cv2.kmeans(dataPixel, K, None, criteria, 10, flags)
        centerUint = np.uint8(center)
        classify = centerUint[labels.flatten()]  # 将像素标记为聚类中心颜色
        imgKmean5 = classify.reshape((img.shape))  # 恢复为二维图像

        plt.figure(figsize=(9, 7))
        plt.subplot(221), plt.axis('off'), plt.title("Origin")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 显示 img1(RGB)
        plt.subplot(222), plt.axis('off'), plt.title("K-mean (k=3)")
        plt.imshow(cv2.cvtColor(imgKmean3, cv2.COLOR_BGR2RGB))
        plt.subplot(223), plt.axis('off'), plt.title("K-mean (k=4)")
        plt.imshow(cv2.cvtColor(imgKmean4, cv2.COLOR_BGR2RGB))
        plt.subplot(224), plt.axis('off'), plt.title("K-mean (k=5)")
        plt.imshow(cv2.cvtColor(imgKmean5, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()

    def partition_grow(self):
        """
        图像区域分割 生长
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        img = cv2.imread(self.cvImagePath, flags=0)

        # OTSU 全局阈值处理
        ret, imgOtsu = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)  # 阈值分割, thresh=T
        # 自适应局部阈值处理
        binaryMean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
        # 区域生长图像分割
        # seeds = [(10, 10), (82, 150), (20, 300)]  # 直接给定 种子点
        imgBlur = cv2.blur(img, (3, 3))  # cv2.blur 方法
        _, imgTop = cv2.threshold(imgBlur, 250, 255, cv2.THRESH_BINARY)  # 高百分位阈值产生种子区域
        nseeds, labels, stats, centroids = cv2.connectedComponentsWithStats(imgTop)  # 过滤连通域，获得质心点 (x,y)
        seeds = centroids.astype(int)  # 获得质心像素作为种子点
        imgGrowth = regional_growth(img, seeds, 8)

        plt.figure(figsize=(8, 6))
        plt.subplot(221), plt.axis('off'), plt.title("Origin")
        plt.imshow(img, 'gray')
        plt.subplot(222), plt.axis('off'), plt.title("OTSU(T={})".format(ret))
        plt.imshow(imgOtsu, 'gray')
        plt.subplot(223), plt.axis('off'), plt.title("Adaptive threshold")
        plt.imshow(binaryMean, 'gray')
        plt.subplot(224), plt.axis('off'), plt.title("Region grow")
        plt.imshow(255 - imgGrowth, 'gray')
        plt.tight_layout()
        plt.show()

    def partition_part(self):
        """
        图像区域分割 分离
        :return:
        """
        if self.cvImagePath == None:
            QMessageBox.information(self, "提示", "请先打开图片！")
            return 0

        img = cv2.imread(self.cvImagePath, flags=0)
        hImg, wImg = img.shape
        mean = np.mean(img)  # 窗口区域的均值
        var = np.std(img, ddof=1)  # 窗口区域的标准差，无偏样本标准差
        print("h={}, w={}, mean={:.2f}, var={:.2f}".format(hImg, wImg, mean, var))

        maxMean = 80  # 均值上界
        minVar = 10  # 标准差下界
        src = img.copy()
        dst1 = np.zeros_like(img)
        dst2 = np.zeros_like(img)
        dst3 = np.zeros_like(img)
        SplitMerge(src, dst1, hImg, wImg, 0, 0, maxMean, minVar, cell=32)  # 最小分割区域 cell=32
        SplitMerge(src, dst2, hImg, wImg, 0, 0, maxMean, minVar, cell=16)  # 最小分割区域 cell=16
        SplitMerge(src, dst3, hImg, wImg, 0, 0, maxMean, minVar, cell=8)  # 最小分割区域 cell=8

        plt.figure(figsize=(9, 7))
        plt.subplot(221), plt.axis('off'), plt.title("Origin")
        plt.imshow(img, 'gray')
        plt.subplot(222), plt.axis('off'), plt.title("Region split (c=32)")
        plt.imshow(dst1, 'gray')
        plt.subplot(223), plt.axis('off'), plt.title("Region split (c=16)")
        plt.imshow(dst2, 'gray')
        plt.subplot(224), plt.axis('off'), plt.title("Region split (c=8)")
        plt.imshow(dst3, 'gray')
        plt.tight_layout()
        plt.show()

    def Pedestrian_detection_video(self):
        """
        行人目标检测-视频
        :return:
        """
        path, type = QFileDialog.getOpenFileName(self.centralwidget, "打开视频", " ",
                                                 "*.mp4;;*.avi;;All files(*)")

        if path == '':
            pass  # 防止关闭或取消导入关闭所有页面
        else:
            if path == None:
                QMessageBox.information(self, "提示", "请先打开图片/视频！")
                return 0
        self.show_edit.setText("路径：" + path)
        # Load the YOLOv8 model
        model = YOLO('yolov8n.pt')

        if type == "*.mp4" or "*.avi":
            # Open the video file
            cap = cv2.VideoCapture(path)

            # Loop through the video frames
            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()

                if success:
                    # Run YOLOv8 inference on the frame
                    results = model(frame, classes=0)

                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()

                    # Display the annotated frame
                    cv2.imshow("YOLOv8 Inference Break the loop if 'q' is pressed", annotated_frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    # Break the loop if the end of the video is reached
                    break

            # Release the video capture object and close the display window
            cap.release()
            cv2.destroyAllWindows()



    def Pedestrian_detection_image(self):
        """
        行人目标检测-图片
        :return:
        """
        path, type = QFileDialog.getOpenFileName(self.centralwidget, "打开图片", " ",
                                                 "*.jpg;;*.png;;All files(*)")

        if path == '':
            pass  # 防止关闭或取消导入关闭所有页面
        else:
            if path == None:
                QMessageBox.information(self, "提示", "请先打开图片！")
                return 0
        self.show_edit.setText("路径：" + path)
        cvimg = cv2.imread(path)
        showImage(self, cvimg)
        # Load the YOLOv8 model
        model = YOLO('yolov8n.pt')
        if type == "*.jpg" or "*.png":
            # 使用模型对图片进行预测
            results = model.predict(path, show= False, save=False, classes=0)
            img = results[0].plot()
            # 获取图片的原始宽度和高度
            h, w = img.shape[:2]
            # 定义目标大小区间
            max_width = 800
            max_height = 900
            # 计算宽度和高度的最大缩放比例
            scale = min(max_width / w, max_height / h)

            # 使用最大缩放比例来缩放图片
            img = cv2.resize(img, None, fx=scale, fy=scale)
            cv2.imshow("result", img)
            cv2.waitKey(0)

'''
class分界线
'''
def adaptive_median_filter(img, max_size):
    # 获取图像大小
    height, width = img.shape

    # 创建输出图像
    output = np.zeros_like(img)

    # 遍历每个像素
    for i in range(height):
        for j in range(width):
            # 获取当前像素周围的像素值
            values = []
            for k in range(max_size):
                for l in range(max_size):
                    x = i - (max_size // 2) + k
                    y = j - (max_size // 2) + l
                    if x >= 0 and x < height and y >= 0 and y < width:
                        values.append(img[x, y])

            # 计算中值和极差
            median = np.median(values)
            diff = np.max(values) - np.min(values)

            # 如果中值不在最小值和最大值之间，则缩小滤波器的大小
            if median > np.min(values) and median < np.max(values):
                # 进一步计算中值和极差
                median = np.median(values)
                diff = np.max(values) - np.min(values)

                # 如果当前像素的值不在最小值和最大值之间，则使用中值
                if img[i, j] < np.min(values) or img[i, j] > np.max(values):
                    output[i, j] = median
                else:
                    output[i, j] = img[i, j]
            else:
                # 缩小滤波器的大小
                max_size = max_size - 2
                if max_size < 3:
                    output[i, j] = median
                else:
                    return adaptive_median_filter(img, max_size)

    return output
def SplitMerge(src, dst, h, w, h0, w0, maxMean, minVar, cell=4):
    win = src[h0: h0 + h, w0: w0 + w]
    mean = np.mean(win)  # 窗口区域的均值
    var = np.std(win, ddof=1)  # 窗口区域的标准差，无偏样本标准差

    if (mean < maxMean) and (var > minVar) and (h < 2 * cell) and (w < 2 * cell):
        # 该区域满足谓词逻辑条件，判为目标区域，设为白色
        dst[h0:h0 + h, w0:w0 + w] = 255  # 白色
        # print("h0={}, w0={}, h={}, w={}, mean={:.2f}, var={:.2f}".
        #       format(h0, w0, h, w, mean, var))
    else:  # 该区域不满足谓词逻辑条件
        if (h > cell) and (w > cell):  # 区域能否继续分拆？继续拆
            SplitMerge(src, dst, (h + 1) // 2, (w + 1) // 2, h0, w0, maxMean, minVar, cell)
            SplitMerge(src, dst, (h + 1) // 2, (w + 1) // 2, h0, w0 + (w + 1) // 2, maxMean, minVar, cell)
            SplitMerge(src, dst, (h + 1) // 2, (w + 1) // 2, h0 + (h + 1) // 2, w0, maxMean, minVar, cell)
            SplitMerge(src, dst, (h + 1) // 2, (w + 1) // 2, h0 + (h + 1) // 2, w0 + (w + 1) // 2, maxMean, minVar,
                       cell)


def getGrayDiff(image, currentPoint, tmpPoint):  # 求两个像素的距离
    return abs(int(image[currentPoint[0], currentPoint[1]]) - int(image[tmpPoint[0], tmpPoint[1]]))


# 区域生长算法
def regional_growth(img, seeds, thresh=5):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        if (0 < seed[0] < height and 0 < seed[1] < weight): seedList.append(seed)
    label = 1  # 种子位置标记
    connects = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]  # 8 邻接连通
    while (len(seedList) > 0):  # 如果列表里还存在点
        currentPoint = seedList.pop(0)  # 将最前面的那个抛出
        seedMark[currentPoint[0], currentPoint[1]] = label  # 将对应位置的点标记为 1
        for i in range(8):  # 对这个点周围的8个点一次进行相似性判断
            tmpX = currentPoint[0] + connects[i][0]
            tmpY = currentPoint[1] + connects[i][1]
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:  # 是否超出限定阈值
                continue
            grayDiff = getGrayDiff(img, currentPoint, (tmpX, tmpY))  # 计算灰度差
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append((tmpX, tmpY))
    return seedMark



def degradation_function(m, n, a, b, T):
    P = m / 2 + 1
    Q = n / 2 + 1
    Mo = np.zeros((m, n), dtype=complex)
    for u in range(m):
        for v in range(n):
            temp = cmath.pi * ((u - P) * a + (v - Q) * b)
            if temp == 0:
                Mo[u, v] = T
            else:
                Mo[u, v] = T * cmath.sin(temp) / temp * cmath.exp(- 1j * temp)
    return Mo


def image_mapping(image):
    img = image / np.max(image) * 255
    return img


def highPassFiltering(img, size):  # 传递参数为傅里叶变换后的频谱图和滤波尺寸
    h, w = img.shape[0:2]  # 获取图像属性
    h1, w1 = int(h / 2), int(w / 2)  # 找到傅里叶频谱图的中心点
    img[h1 - int(size / 2):h1 + int(size / 2),
    w1 - int(size / 2):w1 + int(size / 2)] = 0  # 中心点加减滤波尺寸的一半，刚好形成一个定义尺寸的滤波大小，然后设置为0
    return img


def gaussLowPassFilter(shape, radius=10):  # 高斯低通滤波器
    # 高斯滤波器：# Gauss = 1/(2*pi*s2) * exp(-(x**2+y**2)/(2*s2))
    u, v = np.mgrid[-1:1:2.0 / shape[0], -1:1:2.0 / shape[1]]
    D = np.sqrt(u ** 2 + v ** 2)
    D0 = radius / shape[0]
    kernel = np.exp(- (D ** 2) / (2 * D0 ** 2))
    return kernel


def lowPassFiltering(img, size):  # 传递参数为傅里叶变换后的频谱图和滤波尺寸
    h, w = img.shape[0:2]  # 获取图像属性
    h1, w1 = int(h / 2), int(w / 2)  # 找到傅里叶频谱图的中心点
    img2 = np.zeros((h, w), np.uint8)  # 定义空白黑色图像，和傅里叶变换传递的图尺寸一致
    img2[h1 - int(size / 2):h1 + int(size / 2),
    w1 - int(size / 2):w1 + int(size / 2)] = 1  # 中心点加减滤波尺寸的一半，刚好形成一个定义尺寸的滤波大小，然后设置为1，保留低频部分
    img3 = img2 * img  # 将定义的低通滤波与传入的傅里叶频谱图一一对应相乘，得到低通滤波
    return img3


def dft2Image(image):  # 最优扩充的快速傅立叶变换
    # 中心化, centralized 2d array f(x,y) * (-1)^(x+y)
    mask = np.ones(image.shape)
    mask[1::2, ::2] = -1
    mask[::2, 1::2] = -1
    fImage = image * mask  # f(x,y) * (-1)^(x+y)

    # 最优 DFT 扩充尺寸
    rows, cols = image.shape[:2]  # 原始图片的高度和宽度
    rPadded = cv2.getOptimalDFTSize(rows)  # 最优 DFT 扩充尺寸
    cPadded = cv2.getOptimalDFTSize(cols)  # 用于快速傅里叶变换

    # 边缘扩充(补0), 快速傅里叶变换
    dftImage = np.zeros((rPadded, cPadded, 2), np.float32)  # 对原始图像进行边缘扩充
    dftImage[:rows, :cols, 0] = fImage  # 边缘扩充，下侧和右侧补0
    cv2.dft(dftImage, dftImage, cv2.DFT_COMPLEX_OUTPUT)  # 快速傅里叶变换
    return dftImage


def showImage(self, img):
    img = Window.cvimg2pixmap(self, img)
    # 根据图像与label的比例，最大化图像在label中的显示
    ratio = max(img.width() / self.label_image.width(), img.height() / self.label_image.height())
    img = img.scaled(int(img.width() / ratio), int(img.height() / ratio))
    # 图像在label中居中显示
    self.label_image.setAlignment(Qt.AlignCenter)
    self.label_image.setPixmap(img)


def main():
    app = QApplication(sys.argv)
    mywindow = Window()
    mywindow.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
