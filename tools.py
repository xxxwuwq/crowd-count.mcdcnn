# -*-coding:utf-8-*-

import numpy as np
import cv2 as cv
import math
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat

np.set_printoptions(threshold=np.inf)


class Tool(object):
    def __init__(self):
        self.crop_size = 256
        self.k = 2
        self.scale = 4
        self.beta = 0.3

    def random_crop(self, img, points):
        """
        训练时随机进行裁剪
        :param img: numpy.ndaaray, 输入图片,(h, w, c) or (h, w)
        :param points:
        :return:
        """
        h, w = img.shape[0], img.shape[1]
        # 如果图片小于裁剪尺寸，则裁剪尺寸宽高变为原来1/2
        crop_size = self.crop_size
        if h < self.crop_size or w < self.crop_size:
            crop_size = self.crop_size // 2
        #  随机选择裁剪点
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)

        x2 = x1 + crop_size
        y2 = y1 + crop_size

        points_bak = points.copy()
        cropped_points = []
        for i in range(len(points)):
            # 筛选裁剪范围内的点
            if x1 <= points_bak[i, 0] <= x2 and y1 <= points_bak[i, 1] <= y2:
                points_bak[i, 0] = (points[i, 0] - x1)
                points_bak[i, 1] = (points[i, 1] - y1)
                cropped_points.append(points_bak[i])

        # 得到裁剪的图片、点标注及人数
        cropped_img = img[y1:y2, x1:x2, ...]
        cropped_points = np.asarray(cropped_points)
        croped_cnt = len(cropped_points)

        return cropped_img, cropped_points, croped_cnt

    def fspecial(self, rows, cols, sigma):
        """
        二维高斯核
        :param krow: float, 高斯核高度
        :param kcol: float, 高斯核宽度
        :param sigma: float, sigma参数
        :return: 二位高斯核
        """
        y, x = np.mgrid[-rows / 2 + 0.5:rows / 2 + 0.5, -cols / 2 + 0.5:cols / 2 + 0.5]
        gaussian_dis = np.exp(-(np.square(x) + np.square(y)) / (2 * np.power(sigma, 2))) / (2 * np.power(sigma, 2))
        norm = gaussian_dis / gaussian_dis.sum()

        return norm

    def knn(self, pointx_x, point_y, points, k):
        """
        k近邻距离
        :param pointx_x: float，人头中心点x坐标
        :param point_y: float, 人头中心点y坐标
        :param points: float, numpy.ndarray, 图片中所有人头点集合，(n, 2)
        :param k: int, k近邻
        :return: k近邻距离
        """
        num_points = len(points)
        if k >= num_points:
            return 1.0
        else:
            distance = np.zeros((num_points, 1), dtype=np.float)
            for i in range(num_points):
                x1 = points[i, 0]
                y1 = points[i, 1]
                # 欧式距离
                distance[i, 0] = math.sqrt(math.pow(pointx_x - x1, 2) + math.pow(point_y - y1, 2))
            distance[:, 0] = np.sort(distance[:, 0])
            sum = 0.0
            for j in range(1, k + 1):
                sum = sum + distance[j, 0]
        return sum / k

    def get_density_map(self, dmp_szie, points, use_knn):
        """
        密度图
        :param dmp_szie: tuple or list, 密度图宽高, [h, w] or (h, w)
        :param points: numpy.ndarray, 人头中心点集合, (n, 2)
        :param use_knn: Ture or False，是否使用几何自适应高斯核
        :return: 高斯密度图
        """
        h, w = dmp_szie[0], dmp_szie[1]
        density_map = np.zeros((h, w))

        num = len(points)
        if num == 0:
            return density_map

        for i in range(num):
            x = min(w, max(0, abs(int(math.floor(points[i, 0])))))
            y = min(h, max(0, abs(int(math.floor(points[i, 1])))))

            sigma = 15
            ksize = 15
            if use_knn:
                avg_dist = self.knn(x, y, points, self.k)
                # limit with in 100 pixels
                avg_dist = max(1.0, min(avg_dist, 25.0))
                sigma = self.beta * avg_dist
                ksize = avg_dist
            radius = ksize / 2
            x1 = x - int(math.floor(radius))
            y1 = y - int(math.floor(radius))
            x2 = x + int(math.ceil(radius))
            y2 = y + int(math.ceil(radius))

            # 边界处理
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            H = self.fspecial(y2 - y1, x2 - x1, sigma)
            density_map[y1:y2, x1:x2] = density_map[y1:y2, x1:x2] + H

        return np.asarray(density_map)

    def read_train_data(self, img_path, gt_path, use_knn=True):
        """
        读取训练数据(For shanghai tech dataset)
        :param img_path:
        :param gt_path:
        :param scale:
        :param use_knn:
        :return:
        """
        # opencv的格式为BGR
        img = cv.imread(img_path)
        data = loadmat(gt_path)
        points = data['image_info'][0][0]['location'][0][0]
        # number = data['image_info'][0][0]['number'][0][0]
        cropped_img, cropped_points, cropped_count = self.random_crop(img, points)

        density_map_points = cropped_points / self.scale
        h1, w1, c1 = cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2]
        h2, w2 = h1 // self.scale, w1 // self.scale
        density_map_size = [h2, w2]
        density_map = self.get_density_map(density_map_size, density_map_points, use_knn=use_knn)

        crowd_img = cropped_img.reshape((1, h1, w1, c1))
        density_map = density_map.reshape((1, h2, w2, 1))
        crowd_count = np.asarray(cropped_count).reshape((1, 1))
        return crowd_img, density_map, crowd_count

    def read_test_data(self, img_path, gt_path, use_knn=True):
        """
        读取测试数据(For shanghai tech dataset)
        :param img_path:
        :param gt_path:
        :param use_knn:
        :return:
        """
        crowd_img = cv.imread(img_path)
        data = loadmat(gt_path)
        points = data['image_info'][0][0]['location'][0][0]
        crowd_count = data['image_info'][0][0]['number'][0][0]
        density_map_points = points / self.scale
        h1, w1, c1 = crowd_img.shape[0], crowd_img.shape[1], crowd_img.shape[2]
        h2, w2 = h1 // self.scale, w1 // self.scale
        density_map_size = [h2, w2]
        density_map = self.get_density_map(density_map_size, density_map_points, use_knn=use_knn)

        crowd_img = crowd_img.reshape((1, h1, w1, c1))
        density_map = density_map.reshape((1, h2, w2, 1))
        crowd_count = np.asarray(crowd_count).reshape((1, 1))

        return crowd_img, density_map, crowd_count

    def show_dmp(self, density_map):
        """
        展示密度图
        :param density_map:
        :return:
        """
        plt.imshow(density_map, cmap='jet')

    def mae_metrix(self, gt, est):
        """
        绝对误差
        :param gt:
        :param est:
        :return:
        """
        return np.abs(np.subtract(gt, est)).mean()

    def mse_metrix(self, gt, est):
        """
        平均误差
        :param gt:
        :param est:
        :return:
        """
        return np.power(np.subtract(gt, est), 2).mean()

    def set_GPU(self, gpu=0):
        """
        设置gpu使用
        :param gpu:
        :return:
        """
        g_id = str(gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = g_id


if __name__ == '__main__':
    img_dir = './data/ShanghaiTech/part_A_final/train_data/images/'
    gt_dir = './data/ShanghaiTech/part_A_final/train_data/ground_truth/'

    img_list = os.listdir(img_dir)
    gt_list = os.listdir(gt_dir)
    tool = Tool()
    for i in range(len(img_list)):
        img_path = img_dir + img_list[i]
        gt_path = gt_dir + gt_list[i]
        # gt_path = gt_dir + 'GT_' + img_list[i].split(r'.')[0]
        img, dmp, cnt = tool.read_test_data(img_path, gt_path)
        dmp = cv.resize(dmp[0, ..., 0], (img.shape[2], img.shape[1]), interpolation=cv.INTER_CUBIC)
        tool.show_dmp(dmp)
        plt.imshow(img[0, ...], alpha=0.5)
        plt.show()
        print(img.shape, dmp.shape, cnt, dmp.sum() / 16)
    #
