import collections

import numpy as np
import cv2
from skimage.morphology import skeletonize
import math

class Skeletonize_Extend():

    def __init__(self, drop=False, if_debug=False):
        self.drop = drop
        self.kernel_size = 3
        self.merge_size = 20
        self.if_debug = if_debug
        self.total_mean_width = 10

    def run(self, source_img, mask_img, mask_th):

        mask_img[mask_img < mask_th] = 0
        mask_img[mask_img != 0] = 255

        skeleton = skeletonize(mask_img, method='lee')
        gray_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
        # canny_img = cv2.Canny(gray_img, 20, 100)
        _, binary_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY_INV)

        if self.if_debug:
            cv2.imwrite('data/debug_results_extend/mask.png', mask_img)
            cv2.imwrite('data/debug_results_extend/gray.png', gray_img)
            # cv2.imwrite('../data/debug_results_extend/canny.png', canny_img)
            cv2.imwrite('data/debug_results_extend/skeleton_lee.png', skeleton)
            cv2.imwrite('data/debug_results_extend/binary.png', binary_img)

        dist_img = cv2.distanceTransform(mask_img, cv2.DIST_L2, 3)

        skeleton[:self.kernel_size, :] = 0
        skeleton[-(self.kernel_size + 1):, :] = 0
        skeleton[:, :self.kernel_size] = 0
        skeleton[:, -(self.kernel_size + 1):] = 0

        ends_list = self.extractEndslist(skeleton)

        ends_dict = self.calEndDirection(skeleton, ends_list)

        skeleton_extend = self.extendSkeleton_Binary(source_img, binary_img, skeleton, ends_dict)

        if self.if_debug:
            cv2.imwrite('data/debug_results_extend/skeleton_extend.png', skeleton_extend)
            self.showPoints(skeleton_extend, ends_list)


    def extendSkeleton2(self, source_img, mask_img, skel, ends_dict):
        hsv_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2HSV)
        IMG_H, IMG_W = skel.shape[0], skel.shape[1]
        for i, end_dict in ends_dict.items():
            end_point = end_dict['point']
            end_hsv = hsv_img[end_point[1]][end_point[0]]
            while True:
                if end_point[0] < self.kernel_size or end_point[0] > IMG_W - self.kernel_size or end_point[
                    1] < self.kernel_size or end_point[1] > IMG_H - self.kernel_size:
                    break
                end_skel_window = skel[(end_point[1] - 1):(end_point[1] + 2),
                                       (end_point[0] - 1):(end_point[0] + 2)]
                end_hsv_window = hsv_img[(end_point[1] - 1):(end_point[1] + 2),
                                         (end_point[0] - 1):(end_point[0] + 2)]
                end_skel_window[1][1] = 0
                end_skel_point_ = np.where(end_skel_window != 0)
                end_skel_window[1][1] = 255
                if len(end_skel_point_[0]) > 1:
                    print(3)
                    break
                else:
                    end_skel_point = (end_skel_point_[1][0], end_skel_point_[0][0])
                    diff_hsv_min = 2
                    diff_hsv_min_point = (0, 0)
                    for col in range(3):
                        for row in range(3):
                            if self.distance2D((col, row), end_skel_point) >= 2:
                                end_hsv_ = end_hsv_window[col][row]
                                diff_hsv = self.costHSV(end_hsv, end_hsv_window[col][row])
                                if diff_hsv < diff_hsv_min:
                                    diff_hsv_min = diff_hsv
                                    diff_hsv_min_point = (col, row)
                    if diff_hsv_min < 1:
                        end_skel_window[diff_hsv_min_point[1]][diff_hsv_min_point[0]] = 255
                        end_point = (end_point[0]-1+diff_hsv_min_point[0], end_point[1]-1+diff_hsv_min_point[1])
                    else:
                        print(2)
                        break
            cv2.imwrite('data/debug_results_extend/skeleton_extend.png', skel)
        return skel


    def extendSkeleton(self, source_img, mask_img, skel, ends_dict):
        hsv_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2HSV)
        IMG_H, IMG_W = skel.shape[0], skel.shape[1]
        for i, end_dict in ends_dict.items():
            end_point = end_dict['point']
            end_dir_c = end_dict['dir_c']
            end_dir = end_dict['dir']
            while True:
                if end_point[0] < 3 * self.kernel_size or end_point[0] > IMG_W - 3 * self.kernel_size or end_point[
                    1] < 3 * self.kernel_size or end_point[1] > IMG_H - 3 * self.kernel_size:
                    break
                end_hsv = hsv_img[end_point[1]][end_point[0]]
                diff_hsv_min = 2
                diff_hsv_min_point = (0, 0)
                if end_dir_c == 'r':
                    for row in range(end_point[1] - 2 * self.kernel_size, end_point[1] + 2 * self.kernel_size + 1):
                        end_hsv_ = source_img[row][end_point[0] - 2 * self.kernel_size]
                        diff_hsv = self.costHSV(end_hsv, end_hsv_)
                        if diff_hsv < diff_hsv_min:
                            diff_hsv_min = diff_hsv
                            diff_hsv_min_point = (end_point[0] - 2 * self.kernel_size, row)
                elif end_dir_c == 'l':
                    for row in range(end_point[1] - 2 * self.kernel_size, end_point[1] + 2 * self.kernel_size + 1):
                        end_hsv_ = source_img[row][end_point[0] + 2 * self.kernel_size + 1]
                        diff_hsv = self.costHSV(end_hsv, end_hsv_)
                        if diff_hsv < diff_hsv_min:
                            diff_hsv_min = diff_hsv
                            diff_hsv_min_point = (end_point[0] + 2 * self.kernel_size + 1, row)
                elif end_dir_c == 'u':
                    for col in range(end_point[0] - 2 * self.kernel_size, end_point[0] + 2 * self.kernel_size + 1):
                        end_hsv_ = source_img[end_point[1] + 2 * self.kernel_size + 1][col]
                        diff_hsv = self.costHSV(end_hsv, end_hsv_)
                        if diff_hsv < diff_hsv_min:
                            diff_hsv_min = diff_hsv
                            diff_hsv_min_point = (col, end_point[1] + 2 * self.kernel_size + 1)
                else:
                    for col in range(end_point[0] - 2 * self.kernel_size, end_point[0] + 2 * self.kernel_size + 1):
                        end_hsv_ = source_img[end_point[1] - 2 * self.kernel_size][col]
                        diff_hsv = self.costHSV(end_hsv, end_hsv_)
                        if diff_hsv < diff_hsv_min:
                            diff_hsv_min = diff_hsv
                            diff_hsv_min_point = (col, end_point[1] - 2 * self.kernel_size)
                if diff_hsv_min < 0.5:
                    cv2.line(skel, end_point, diff_hsv_min_point, 255, 1)
                    end_dir = (end_point[0] - diff_hsv_min_point[0], end_point[1] - diff_hsv_min_point[1])
                    end_dir_c = self.DirClass(end_dir)
                    end_point = diff_hsv_min_point
                else:
                    break
        return skel

    def extendSkeleton_Gray(self, source_img, gray_img, skel, ends_dict):
        IMG_H, IMG_W = skel.shape[0], skel.shape[1]
        for i, end_dict in ends_dict.items():
            end_point = end_dict['point']
            end_dir_c = end_dict['dir_c']
            end_dir = end_dict['dir']
            while True:
                if end_point[0] < 3 * self.kernel_size or end_point[0] > IMG_W - 3 * self.kernel_size or end_point[
                    1] < 3 * self.kernel_size or end_point[1] > IMG_H - 3 * self.kernel_size:
                    break
                end_gray = gray_img[end_point[1]][end_point[0]]
                diff_gray_min = 255
                diff_gray_min_point = (0, 0)
                if end_dir_c == 'r':
                    for row in range(end_point[1] - 2 * self.kernel_size, end_point[1] + 2 * self.kernel_size + 1):
                        end_gray_ = gray_img[row][end_point[0] - 2 * self.kernel_size]
                        diff_gray = abs(end_gray_ - end_gray)
                        if diff_gray < diff_gray_min:
                            diff_gray_min = diff_gray
                            diff_gray_min_point = (end_point[0] - 2 * self.kernel_size, row)
                elif end_dir_c == 'l':
                    for row in range(end_point[1] - 2 * self.kernel_size, end_point[1] + 2 * self.kernel_size + 1):
                        end_gray_ = gray_img[row][end_point[0] + 2 * self.kernel_size + 1]
                        diff_gray = abs(end_gray_ - end_gray)
                        if diff_gray < diff_gray_min:
                            diff_gray_min = diff_gray
                            diff_gray_min_point = (end_point[0] + 2 * self.kernel_size + 1, row)
                elif end_dir_c == 'u':
                    for col in range(end_point[0] - 2 * self.kernel_size, end_point[0] + 2 * self.kernel_size + 1):
                        end_gray_ = gray_img[end_point[1] + 2 * self.kernel_size + 1][col]
                        diff_gray = abs(end_gray_ - end_gray)
                        if diff_gray < diff_gray_min:
                            diff_gray_min = diff_gray
                            diff_gray_min_point = (col, end_point[1] + 2 * self.kernel_size + 1)
                else:
                    for col in range(end_point[0] - 2 * self.kernel_size, end_point[0] + 2 * self.kernel_size + 1):
                        end_gray_ = gray_img[end_point[1] - 2 * self.kernel_size][col]
                        diff_gray = abs(end_gray_ - end_gray)
                        if diff_gray < diff_gray_min:
                            diff_gray_min = diff_gray
                            diff_gray_min_point = (col, end_point[1] - 2 * self.kernel_size)
                if diff_gray_min < 40:
                    cv2.line(skel, end_point, diff_gray_min_point, 255, 1)
                    end_dir = (end_point[0] - diff_gray_min_point[0], end_point[1] - diff_gray_min_point[1])
                    end_dir_c = self.DirClass(end_dir)
                    end_point = diff_gray_min_point
                else:
                    break
        return skel

    def extendSkeleton_Binary(self, source_img, binary_img, skel, ends_dict):
        IMG_H, IMG_W = skel.shape[0], skel.shape[1]
        for i, end_dict in ends_dict.items():
            end_point = end_dict['point']
            end_dir_c = end_dict['dir_c']
            end_dir = end_dict['dir']
            while True:
                if end_point[0] < 3 * self.kernel_size or end_point[0] > IMG_W - 3 * self.kernel_size or end_point[
                    1] < 3 * self.kernel_size or end_point[1] > IMG_H - 3 * self.kernel_size:
                    break
                if end_dir_c == 'r':
                    binary_row_list = []
                    for row in range(end_point[1] - 2 * self.kernel_size, end_point[1] + 2 * self.kernel_size + 1):
                        end_binary_ = binary_img[row][end_point[0] - 2 * self.kernel_size]
                        if end_binary_ == 255:
                            binary_row_list.append(row)
                    if len(binary_row_list) == 0:
                        break
                    next_point = (end_point[0] - 2 * self.kernel_size, (binary_row_list[0] + binary_row_list[-1]) // 2)
                elif end_dir_c == 'l':
                    binary_row_list = []
                    for row in range(end_point[1] - 2 * self.kernel_size, end_point[1] + 2 * self.kernel_size + 1):
                        end_binary_ = binary_img[row][end_point[0] + 2 * self.kernel_size + 1]
                        if end_binary_ == 255:
                            binary_row_list.append(row)
                    if len(binary_row_list) == 0:
                        break
                    next_point = (end_point[0] + 2 * self.kernel_size + 1, (binary_row_list[0] + binary_row_list[-1]) // 2)
                elif end_dir_c == 'u':
                    binary_col_list = []
                    for col in range(end_point[0] - 2 * self.kernel_size, end_point[0] + 2 * self.kernel_size + 1):
                        end_binary_ = binary_img[end_point[1] + 2 * self.kernel_size + 1][col]
                        if end_binary_ == 255:
                            binary_col_list.append(col)
                    if len(binary_col_list) == 0:
                        break
                    next_point = ((binary_col_list[0] + binary_col_list[-1])//2, end_point[1] + 2 * self.kernel_size + 1)
                else:
                    binary_col_list = []
                    for col in range(end_point[0] - 2 * self.kernel_size, end_point[0] + 2 * self.kernel_size + 1):
                        end_binary_ = binary_img[end_point[1] - 2 * self.kernel_size][col]
                        if end_binary_ == 255:
                            binary_col_list.append(col)
                    if len(binary_col_list) == 0:
                        break
                    next_point = ((binary_col_list[0] + binary_col_list[-1]) // 2, end_point[1] - 2 * self.kernel_size)
                cv2.line(skel, end_point, next_point, 255, 1)
                end_dir = (end_point[0] - next_point[0], end_point[1] - next_point[1])
                end_dir_c = self.DirClass(end_dir)
                end_point = next_point
        return skel

    def costHSV(self, hsv1, hsv2):
        hue_1, hue_2 = int(hsv1[0]), int(hsv2[0])
        if min(hue_1, hue_2) < 10 and max(hue_1, hue_2) > 156:
            if hue_2 > hue_1:
                hue_2 -= 180
            else:
                hue_1 -= 180
        hue_1, hue_2 = hue_1 / 180 * math.pi, hue_2 / 180 * math.pi
        sat_1, sat_2 = int(hsv1[1]) / 255, int(hsv2[1]) / 255
        val_1, val_2 = int(hsv1[2]) / 255, int(hsv2[2]) / 255
        hsv_sp1 = [val_1, sat_1 * math.cos(hue_1), sat_1 * math.sin(hue_1)]
        hsv_sp2 = [val_2, sat_2 * math.cos(hue_2), sat_2 * math.sin(hue_2)]
        dis_hsv = self.distance3D(hsv_sp1, hsv_sp2)
        return dis_hsv

    def distance3D(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5

    def distance2D(self, point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    def calEndDirection(self, skel, ends_list):
        IMG_W, IMG_H = skel.shape[1], skel.shape[0]
        ends_dict = {}
        for i, end in enumerate(ends_list):
            end_dict = {}
            end_dict['point'] = end
            if end[0] < (4 * self.kernel_size) or end[0] > (IMG_W - 4 * self.kernel_size) or end[1] < (4 * self.kernel_size) or end[1] > (IMG_H - 4 * self.kernel_size):
                end_dict['border'] = True
                continue
            else:
                end_dict['border'] = False
            end_window = skel[(end[1] - 2 * self.kernel_size):(end[1] + 2 * self.kernel_size),
                              (end[0] - 2 * self.kernel_size):(end[0] + 2 * self.kernel_size)]
            EWRP = np.where(end_window == 255)
            EWRP -= np.ones_like(EWRP) * 6
            y_sum = np.sum(EWRP[0])
            x_sum = np.sum(EWRP[1])
            xy_multi = 6 / max(abs(x_sum), abs(y_sum))
            end_dict['dir'] = (x_sum * xy_multi, y_sum * xy_multi)
            dir_n = self.DirClass(end_dict['dir'])
            end_dict['dir_c'] = dir_n
            ends_dict[i] = end_dict
        return ends_dict

    def DirClass(self, dir):
        x_sum, y_sum = dir[0], dir[1]
        if x_sum == 0:
            if y_sum > 0:
                dir_n = 'u'
            elif y_sum < 0:
                dir_n = 'b'
        elif x_sum > 0:
            if y_sum >= x_sum:
                dir_n = 'u'
            elif y_sum < x_sum and y_sum >= -x_sum:
                dir_n = 'r'
            else:
                dir_n = 'b'
        else:
            if y_sum >= -x_sum:
                dir_n = 'u'
            elif y_sum < -x_sum and y_sum > x_sum:
                dir_n = 'l'
            else:
                dir_n = 'b'
        return dir_n

    def extractEndslist(self, skel):
        ends = self.extractEnds(skel)
        for e in ends:
            if e.shape[0] == 0:
                return []

        return list(zip(ends[1], ends[0]))

    def extractEnds(self, skel):

        skel = skel.copy()
        skel[skel != 0] = 1
        skel = np.uint8(skel)

        kernel = np.uint8([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])

        src_depth = -1
        filtered = cv2.filter2D(skel, src_depth, kernel)

        p_ends = np.where(filtered == 11)

        return np.array([p_ends[0], p_ends[1]])

    def showPoints(self, skel, ends_list):
        back = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        for end in ends_list:
            cv2.circle(back, end, 3, (0, 0, 255))
        cv2.imwrite('data/debug_results_extend/show_points.jpg', back)