import numpy as np
import cv2
from skimage.morphology import skeletonize

class Skeletonize():

    def __init__(self, drop=False, if_debug=False):
        self.drop = drop
        self.kernel_size = 3
        self.merge_size = 20
        self.cmap = self.voc_cmap(N=256, normalized=False)
        self.if_debug = if_debug
        self.total_mean_width = 10

    def voc_cmap(self, N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        return cmap

    def run(self, mask, density=7):

        if self.if_debug:
            cv2.imwrite('data/debug_results/mask.png', mask)

        skeleton = skeletonize(mask, method='lee')

        if self.if_debug:
            cv2.imwrite('data/debug_results/skeleton_lee.png', skeleton)

        dist_img = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

        # black out borders
        # skeleton = cv2.copyMakeBorder(skeleton[self.kernel_size:-self.kernel_size, self.kernel_size:-self.kernel_size], self.kernel_size, self.kernel_size, self.kernel_size, self.kernel_size, cv2.BORDER_CONSTANT, value=0)
        skeleton[:self.kernel_size, :] = 0
        skeleton[-(self.kernel_size + 1):, :] = 0
        skeleton[:, :self.kernel_size] = 0
        skeleton[:, -(self.kernel_size + 1):] = 0

        # 提取端点,交叉点
        ends = self.extractEnds(skeleton)
        ints = self.extractInts(skeleton)
        ints_rf = self.mergeInts(ints, thre=3*self.kernel_size)

        if self.if_debug:
            self.showPoints(skeleton, ends, ints)

        # 交叉点置零,将图像分为若干段
        skeleton_f = skeleton.copy()
        skeleton_f[tuple([ints[0], ints[1]])] = 0

        if self.if_debug:
            cv2.imwrite('data/debug_results/skeleton_f.png', skeleton_f)

        # 提取路线段
        num_labels, labels = cv2.connectedComponents(skeleton_f)
        routes = self.extractRoutes(num_labels, labels, skeleton_f)

        if self.if_debug:
           self.showRoutes(routes, skeleton_f, prune=False)

        # 计算DLO宽度(平均宽度:该段DLO像素宽度的众数;最大宽度:该段DLO像素宽度的最大值),以及整张图像上所有DLO像素的宽度
        routes = self.estimateRoutewidthFromSegment(routes, dist_img, min_px=3)
        # 删去部分毛刺(平均宽度远小于整张图像上所有DLO像素宽度的DLO,DLO长度小于1.3倍该段DLO像素宽度最大值的DLO)
        routes = self.prune_short_routes(routes)

        if self.if_debug:
           self.showRoutes(routes, skeleton_f, prune=True)

        # DLO重新转化为图像
        skeleton_rf, routes_im = self.RoutesToSkeleton(routes, skeleton_f)

        # 重新提取端点
        ends_rf = self.extractEnds(skeleton_rf)

        # 端点信息绑定(孤立端点,交叉端点,端点隶属DLO编号),端点绑定到交叉点
        ends_dict_iso, ends_dict_int, ints_dict_ends = self.constructEndsDict(routes, routes_im, ends_rf, ints_rf)

        if self.if_debug:
            cv2.imwrite('data/debug_results/skeleton_rf.png', skeleton_rf)

        # 计算交叉端点方向
        ends_dict_int, _ = self.calcRouteDirection(routes_im, ends_dict_int)
        # 处理交叉端点
        end_pairs = self.handleIntersections(skeleton_rf, ends_dict_int, ints_dict_ends)

        # 计算孤立端点方向
        ends_dict_iso, dir_ends_dict = self.calcRouteDirection(routes_im, ends_dict_iso)
        # 孤立端点匹配
        end_pairs += self.checkForContinuity(skeleton_rf, ends_dict_iso, dir_ends_dict)

        # 清除可能出现的重复端点对
        end_pairs = self.cleanEndPairs(end_pairs)

        if self.if_debug:
            # 重新连接
            skeleton_m = self.mergeEnds(skeleton_rf, end_pairs)
            cv2.imwrite('data/debug_results/skeleton_m.png', skeleton_m)

        return skeleton_rf, routes, end_pairs, ends_dict_iso, ends_dict_int



    def showPoints(self, skel, ends=None, ints=None):
        back = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        back = ~back
        if ends is not None:
            ends_list = list(zip(ends[1], ends[0]))
            for end in ends_list:
                cv2.circle(back, end, 3, (0, 0, 255))
        if ints is not None:
            ints_list = list(zip(ints[1], ints[0]))
            for int in ints_list:
                cv2.circle(back, int, 3, (0, 255, 0))
        cv2.imwrite('data/debug_results/show_points.jpg', back)

    def showRoutes(self, routes, skel, prune=False):
        back = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        back_white = np.ones_like(back) * 255
        for i in routes.keys():
            for point in routes[i]['route']:
                back_white[point[0]][point[1]] = self.cmap[i]
        if prune:
            cv2.imwrite('data/debug_results/show_routes_prune.jpg', back_white)
        else:
            cv2.imwrite('data/debug_results/show_routes.jpg', back_white)

    def showEndsAndRadius(self, skel, ends_dict_rf, routes):
        back = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        back = ~back
        for i, end_dict in ends_dict_rf.items():
            end_point = end_dict['point']
            end_route_label = end_dict['route_label'][0]
            radii = int(0.1 * len(routes[end_route_label]))
            cv2.circle(back, (end_point[1], end_point[0]), radii, (0, 255, 0))
        cv2.imwrite('data/debug_results/show_ends_and_radius.jpg', back)

    def mergeInts(self, ints, thre):
        ints_list = list(zip(ints[0], ints[1]))
        ints_rf = []
        for p in ints_list:
            already_in = False
            for v in ints_rf:
                if self.distance2D(p, v) <= thre:
                    already_in = True
            if not already_in:
                ints_rf.append(p)
        return np.array(ints_rf).T

    def RoutesToSkeleton(self, routes, skel):
        back1 = np.zeros_like(skel)
        back2 = np.zeros_like(skel)
        for i in routes.keys():
            for point in routes[i]['route']:
                back1[point[0]][point[1]] = 255
                back2[point[0]][point[1]] = i
        return back1, back2

    def extractRoutes(self, num_labels, labels, skel_img):
        skel = skel_img.copy()
        ends_all = self.extractEndslist(skel)
        routes = {}
        for n in range(1, num_labels):
            ends_f = [e for e in ends_all if labels[tuple([e[1], e[0]])] == n]
            if len(ends_f) == 2:
                route = self.walkFaster(skel, ends_f[0])
                if len(route) > 0:
                    routes[n] = {'route': route, 'ends': []}
        return routes

    def walkFaster(self, skel, start):

        route = [(int(start[1]), int(start[0]))]
        end = False
        while not end:
            end = True
            act = route[-1]
            skel[act[0], act[1]] = 0.
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                if skel[act[0] + dx, act[1] + dy]:
                    aim_x = act[0] + dx
                    aim_y = act[1] + dy
                    route.append((aim_x, aim_y))
                    end = False
                    break

        route = np.array(route)
        route -= 1

        return route

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

    def extractInts(self, skel):

        skel = skel.copy()
        skel[skel != 0] = 1
        skel = np.uint8(skel)

        kernel = np.uint8([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])

        src_depth = -1
        filtered = cv2.filter2D(skel, src_depth, kernel)

        p_ints = np.where(filtered > 12)

        return np.array([p_ints[0], p_ints[1]])

    def distance2D(self, point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    def estimateRoutewidthFromSegment(self, routes, dist_img, min_px = 3):
        for i in routes.keys():
            widths = [dist_img[tuple(p)] for p in routes[i]['route']]
            widths_int = [np.round(dist_img[tuple(p)]) for p in routes[i]['route']]
            average_width = np.argmax(np.bincount(widths_int))
            max_width = np.max(widths) if widths else min_px
            routes[i]['width'] = (average_width, max_width)
        self.total_mean_width = np.mean([routes[i]['width'][0] for i in routes.keys()])
        return routes

    def prune_short_routes(self, routes):
        del_list = []
        for i in routes.keys():
            if len(routes[i]['route']) < routes[i]['width'][1] * 1.3 or routes[i]['width'][0] < self.total_mean_width * 0.5:
                del_list.append(i)
        for del_index in del_list:
            del routes[del_index]
        return routes


    def constructEndsDict(self, routes, routes_im, ends, ints):
        ends_dict_iso = {}
        ends_dict_int = {}
        ints_dict_end = {}
        ends_list = list(zip(ends[0], ends[1]))
        if ints.shape[0] > 0:
            ints_list = list(zip(ints[0], ints[1]))
        else:
            ints_list = []
        for i, end in enumerate(ends_list):
            window_size = self.kernel_size * 2
            label_cover = routes_im[(end[0] - window_size):(end[0] + window_size),
                                    (end[1] - window_size):(end[1] + window_size)]
            routes_label = [v for v in np.unique(label_cover) if v != 0]
            if len(routes_label) <= 1:
                ends_dict_iso[i] = {"point": end, "route_label": routes_im[end[0]][end[1]], "point_label": i, "point_type": 'iso'}
            else:
                ends_dict_int[i] = {"point": end, "route_label": routes_im[end[0]][end[1]], "point_label": i, "point_type": 'int'}
                for j, int in enumerate(ints_list):
                    if self.distance2D(end, int) < self.kernel_size * 4:
                        if j not in ints_dict_end:
                            ints_dict_end[j] = [ends_dict_int[i]]
                        else:
                            ints_dict_end[j].append(ends_dict_int[i])
            routes[routes_im[end[0]][end[1]]]['ends'].append(i)
        return ends_dict_iso, ends_dict_int, ints_dict_end

    # 对于任意非交叉点的端点:
    # [1].是否处于图像边缘,是,则停止检测;
    # [2].计算该顶点的方向,归属到米字型的八个方向;
    # [3].在该方向对应的90度范围内寻找对应的
    def checkForContinuity(self, skel, ends_dict_rf, dir_ends_dict):
        dir_coor_dict = {'l': ['ru', 'r', 'rb'],
                         'lu': ['r', 'rb', 'b'],
                         'u': ['rb', 'b', 'lb'],
                         'ru': ['b', 'lb', 'l'],
                         'r': ['lb', 'l', 'lu'],
                         'rb': ['l', 'lu', 'u'],
                         'b': ['lu', 'u', 'ru'],
                         'lb': ['u', 'ru', 'r']}
        have_paired = []
        end_pairs = []
        for i, end_dict in ends_dict_rf.items():
            if end_dict in have_paired:
                continue
            end_border = end_dict['border']
            if end_border:
                continue
            end_dir = end_dict['dir_c']
            coor_dir_list = dir_coor_dict[end_dir]
            wait_list = []
            for coor_dir in coor_dir_list:
                wait_list += dir_ends_dict[coor_dir]
            for end_dict_ in wait_list:
                if end_dict_ in have_paired:
                    continue
                if end_dict_['border'] or end_dict['route_label'] == end_dict_['route_label']:
                    continue
                CM = self.calcEndSimilarity(end_dict, end_dict_, skel)
                if CM < 0.8:
                    end_dict['point_type'] = 'iso_p'
                    end_dict_['point_type'] = 'iso_p'
                    end_pairs.append((end_dict, end_dict_))
                    have_paired.append(end_dict)
                    have_paired.append(end_dict_)
        return end_pairs

    def mergeEnds(self, skel, end_pairs):
        skel_ = skel.copy()
        for end_pair in end_pairs:
            end_dict_1 = end_pair[0]
            end_dict_2 = end_pair[1]
            end_p1 = end_dict_1['point']
            end_p2 = end_dict_2['point']
            cv2.line(skel_, (end_p1[1], end_p1[0]), (end_p2[1], end_p2[0]), 255, thickness=1)
        return skel_



    def calcRouteDirection(self, routes_im, ends_dict_rf):
        IMG_W, IMG_H = routes_im.shape[1], routes_im.shape[0]
        Dir_L = {'l': 'r', 'lu': 'rb', 'u': 'b', 'ru': 'lb', 'r': 'l', 'rb': 'lu', 'b': 'u', 'lb': 'ru'}
        dir_ends_dict = {'l': [], 'lu': [], 'u': [], 'ru': [], 'r': [], 'rb': [], 'b': [], 'lb': []}
        for i, end_dict in ends_dict_rf.items():
            end_point = end_dict['point']
            end_route_label = end_dict['route_label']
            if end_point[1] < (4 * self.kernel_size) or end_point[1] > (IMG_W - 4 * self.kernel_size) or end_point[0] < (4 * self.kernel_size) or end_point[0] > (IMG_H - 4 * self.kernel_size):
                end_dict['border'] = True
                continue
            else:
                end_dict['border'] = False
            end_window = routes_im[(end_point[0] - 2 * self.kernel_size):(end_point[0] + 2 * self.kernel_size),
                                   (end_point[1] - 2 * self.kernel_size):(end_point[1] + 2 * self.kernel_size)]
            EWRP = np.where(end_window == end_route_label)
            EWRP -= np.ones_like(EWRP) * 6
            y_sum = np.sum(EWRP[0])
            x_sum = np.sum(EWRP[1])
            if x_sum == 0:
                if y_sum > 0:
                    dir_n = 'u'
                elif y_sum < 0:
                    dir_n = 'b'
            elif x_sum > 0:
                if y_sum > 2.4 * abs(x_sum):
                    dir_n = 'u'
                elif y_sum > 0.4 * abs(x_sum) and y_sum <= 2.4 * abs(x_sum):
                    dir_n = 'ru'
                elif y_sum > -0.4 * abs(x_sum) and y_sum <= 0.4 * abs(x_sum):
                    dir_n = 'r'
                elif y_sum > -2.4 * abs(x_sum) and y_sum <= -0.4 * abs(x_sum):
                    dir_n = 'rb'
                else:
                    dir_n = 'b'
            else:
                if y_sum > 2.4 * abs(x_sum):
                    dir_n = 'u'
                elif y_sum > 0.4 * abs(x_sum) and y_sum <= 2.4 * abs(x_sum):
                    dir_n = 'lu'
                elif y_sum > -0.4 * abs(x_sum) and y_sum <= 0.4 * abs(x_sum):
                    dir_n = 'l'
                elif y_sum > -2.4 * abs(x_sum) and y_sum <= -0.4 * abs(x_sum):
                    dir_n = 'lb'
                else:
                    dir_n = 'b'

            end_dict['dir_c'] = Dir_L[dir_n]
            end_dict['dir'] = (x_sum, y_sum)
            dir_ends_dict[Dir_L[dir_n]].append(end_dict)

        return ends_dict_rf, dir_ends_dict

    def calcEndSimilarity(self, end_dict_1, end_dict_2, skel, flag='end_pair'):
        point1 = end_dict_1['point']
        point2 = end_dict_2['point']
        dir_p1 = end_dict_1['dir']
        dir_p2 = end_dict_2['dir']
        CE = self.costEuclidean(point1, point2)
        if CE > 100:
            return 1
        else:
            print("unknown type")
        CD = self.costDirection(dir_p1, dir_p2)
        CC = self.costCurvature(point1, point2, dir_p1, dir_p2)
        CM = 0.01 * CE + CD + CC
        if self.if_debug:
            back = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
            back = ~back
            cv2.circle(back, (point1[1], point1[0]), 3, (0, 255, 0))
            cv2.circle(back, (point2[1], point2[0]), 3, (0, 255, 0))
            # if flag == 'int_pair':
            #     back = back[(point1[0]-4*self.kernel_size):(point1[0]+4*self.kernel_size),
            #                 (point1[1]-4*self.kernel_size):(point1[1]+4*self.kernel_size)]
            #     back = cv2.resize(back, (288, 288))
            cv2.imwrite('data/debug_results/similarity_test/calcSimilarity_{}.jpg'.format(CM), back)
        return CM


    def costEuclidean(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def costDirection(self, dir1, dir2):
        vec1 = np.array(dir1)
        vec2 = np.array(dir2)
        return 1 - np.dot(-vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def costCurvature(self, point1, point2, dir1, dir2):
        vec0 = np.array([(point2[1]-point1[1]), (point2[0]-point1[0])])
        vec1 = np.array(dir1)
        vec2 = np.array(dir2)
        CC1 = np.dot(vec0, -vec1)/(np.linalg.norm(vec0) * np.linalg.norm(vec1))
        CC2 = np.dot(-vec0, -vec2)/(np.linalg.norm(vec0) * np.linalg.norm(vec2))
        return 1 - min([CC1, CC2])

    def handleIntersections(self, skel, ends_dict_int, ints_dict_ends):
        end_pairs = []
        for i, int_dict_ends in ints_dict_ends.items():
            int_end_num = len(int_dict_ends)
            if int_end_num == 2:
                end_pairs.append((int_dict_ends[0], int_dict_ends[1]))
            elif int_end_num == 3:
                end_pairs += self.handleFork(skel, ends_dict_int, int_dict_ends)
            elif int_end_num == 4:
                end_pairs += self.handleCross(skel, ends_dict_int, int_dict_ends)
            else:
                print('{} fork road is not in the scope of treatment.'.format(int_end_num))
        return end_pairs

    def handleFork(self, skel, ends_dict_int, int_dict_ends):
        print('handleFork')
        end_pairs = []
        CM_01 = self.calcEndSimilarity(int_dict_ends[0], int_dict_ends[1], skel, 'int_pair')
        CM_02 = self.calcEndSimilarity(int_dict_ends[0], int_dict_ends[2], skel, 'int_pair')
        CM_12 = self.calcEndSimilarity(int_dict_ends[1], int_dict_ends[2], skel, 'int_pair')
        index_l = ['01', '02', '12']
        CM_l = [CM_01, CM_02, CM_12]
        CM_max_index = index_l[CM_l.index(max(CM_l))]
        index_l.remove(CM_max_index)
        for index_remain in index_l:
            ind1 = int(index_remain[0])
            ind2 = int(index_remain[1])
            end_pairs.append((int_dict_ends[ind1], int_dict_ends[ind2]))
        return end_pairs

    def handleCross(self, skel, ends_dict_int, int_dict_ends):
        print('handleCross')
        end_pairs = []
        CM_01 = self.calcEndSimilarity(int_dict_ends[0], int_dict_ends[1], skel, 'int_pair')
        CM_02 = self.calcEndSimilarity(int_dict_ends[0], int_dict_ends[2], skel, 'int_pair')
        CM_03 = self.calcEndSimilarity(int_dict_ends[0], int_dict_ends[3], skel, 'int_pair')
        CM_12 = self.calcEndSimilarity(int_dict_ends[1], int_dict_ends[2], skel, 'int_pair')
        CM_13 = self.calcEndSimilarity(int_dict_ends[1], int_dict_ends[3], skel, 'int_pair')
        CM_23 = self.calcEndSimilarity(int_dict_ends[2], int_dict_ends[3], skel, 'int_pair')
        index_l = ['01', '02', '03', '12', '13', '23']
        CM_l = [CM_01, CM_02, CM_03, CM_12, CM_13, CM_23]
        CM_min_index = index_l[CM_l.index(min(CM_l))]
        ind1 = int(CM_min_index[0])
        ind2 = int(CM_min_index[1])
        end_pairs.append((int_dict_ends[ind1], int_dict_ends[ind2]))
        CM_the_other_index = '0123'.replace(CM_min_index[0], '').replace(CM_min_index[1], '')
        ind3 = int(CM_the_other_index[0])
        ind4 = int(CM_the_other_index[1])
        end_pairs.append((int_dict_ends[ind3], int_dict_ends[ind4]))
        return end_pairs

    def cleanEndPairs(self, end_pairs):
        end_pairs_clean = []
        for end_pair in end_pairs:
            if end_pair not in end_pairs_clean:
                end_pairs_clean.append(end_pair)
        return end_pairs_clean