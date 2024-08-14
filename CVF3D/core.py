import itertools

import cv2
import numpy as np
import arrow
import math

from .segment.predict import SegNet
from .process.skeletonize_EWD_color import Skeletonize as Skeletonize_EWD
from .process.skeletonize_LAB import Skeletonize as Skeletonize_LAB
from .draw_drac import calcEllipseFromEnds, draw_drac

class Pipeline():

    def __init__(self, checkpoint_seg=None, img_w=1280, img_h=960, if_debug=True, scene='LAB'):
        if checkpoint_seg is not None:
            self.network_seg = SegNet(model_name="deeplabv3plus_resnet101", checkpoint_path=checkpoint_seg, img_w=img_w, img_h=img_h)
        else:
            self.network_seg = None
        self.if_debug = if_debug
        self.cmap = self.voc_cmap(N=256, normalized=False)
        self.scene = scene

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

    def run(self, source_img, mask_img=None, mask_th=127):
        t0 = arrow.utcnow()

        if self.network_seg is not None and mask_img is None:
            mask_img = self.network_seg.predict_img(source_img)
        elif self.network_seg is None and mask_img is None:
            print('Error: mask_img is not found!')
        else:
            print('Get mask_img from DIR')

        # 图像平滑
        mask_img_G = cv2.GaussianBlur(mask_img, (7, 7), 0)

        # mask_img_G = mask_img.copy()
        # mask_img_G = cv2.erode(mask_img, (5, 5))

        if self.if_debug:
            cv2.imwrite('data/debug_results/mask_guass.png', mask_img_G)

        mask_img_G[mask_img_G < mask_th] = 0
        mask_img_G[mask_img_G != 0] = 255

        print(mask_img_G.sum()/255)

        seg_time = (arrow.utcnow() - t0).total_seconds() * 1000

        img_out, skeleton_or_routelist, times, routes, end_pairs, ends_dict = self.process(source_img=source_img, mask_img=mask_img_G, mask_th=mask_th, mask_img_origin=mask_img)

        times['seg_time'] = seg_time

        tot_time = (arrow.utcnow() - t0).total_seconds() * 1000
        times['tot_time'] = tot_time - seg_time

        return img_out, skeleton_or_routelist, times, routes, end_pairs, ends_dict

    def process(self, source_img, mask_img, mask_th, mask_img_origin):

        times = {}

        t0 = arrow.utcnow()

        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2HSV)
        dist_img = cv2.distanceTransform(mask_img_origin, cv2.DIST_L2, 3)

        if self.scene == 'EWD':
            SKEL = Skeletonize_EWD(if_debug=self.if_debug)
        elif self.scene == 'LAB':
            SKEL = Skeletonize_LAB(if_debug=self.if_debug)

        skeleton_rf, routes, end_pairs, ends_dict_rf, int_dicts_m = SKEL.run(mask_img, source_img=source_img)

        skel_time = (arrow.utcnow() - t0).total_seconds() * 1000
        times['skel_time'] = skel_time

        # end_pairs_dict = self.calcEndPairsContinuity(ends_dict_rf, end_pairs, source_img)

        skeleton_m = self.mergeEnds(skeleton_rf, ends_dict_rf, end_pairs)
        ends_dict_rf = self.combineEndsDict(routes, ends_dict_rf, end_pairs)

        if self.scene == 'EWD':
            combEnds_list = self.combineEndsRoutes_NoCircle(routes, ends_dict_rf)
            combEnds_list = self.deleteCloseRoutes(routes, ends_dict_rf, combEnds_list)
            combEnds_list = self.validFromHSV(routes, ends_dict_rf, combEnds_list, source_img)

            comb_time = (arrow.utcnow() - t0).total_seconds() * 1000
            times['comb_time'] = comb_time
            route_seg = self.showCombRoutes(skeleton_m, routes, ends_dict_rf, combEnds_list)
            # route_seg = self.showCombRoutes_WithCrossOrder(skeleton_m, routes, ends_dict_rf, combEnds_list, int_dicts_m, source_img, dist_img, mask_img_origin)

            return route_seg, skeleton_m, times, routes, end_pairs, ends_dict_rf

        elif self.scene == 'LAB':
            combEnds_list = self.combineEndsRoutes_NoCircle(routes, ends_dict_rf)
            combEnds_list = self.deleteCloseRoutes(routes, ends_dict_rf, combEnds_list)
            comb_time = (arrow.utcnow() - t0).total_seconds() * 1000 - skel_time
            times['comb_time'] = comb_time
            route_seg = self.showCombRoutes(skeleton_m, routes, ends_dict_rf, combEnds_list)
            route_seg_list = self.showCombRoutes_Single(skeleton_m, routes, ends_dict_rf, combEnds_list)

            return route_seg, route_seg_list, times, routes, end_pairs, ends_dict_rf



    def calcEndPairsContinuity(self, ends_dict, end_pairs, hsv):
        end_pairs_dict = {}
        for end_pair in end_pairs:
            end_pair_continuity = self.calcCrossContinuity(end_pair, ends_dict, hsv)
            end_pair_ = [end_pair[0], end_pair[1]]
            end_pair = tuple(sorted(end_pair_))
            end_pairs_dict[end_pair] = end_pair_continuity
        return end_pairs_dict

    def mergeEnds(self, skel, ends_dict_rf, end_pairs):
        skel_ = skel.copy()
        for end_pair in end_pairs:
            end_dict_1 = ends_dict_rf[end_pair[0]]
            end_dict_2 = ends_dict_rf[end_pair[1]]
            end_p1 = end_dict_1['point']
            end_p2 = end_dict_2['point']
            cv2.line(skel_, (end_p1[1], end_p1[0]), (end_p2[1], end_p2[0]), 255, thickness=1)
        return skel_

    def combineEndsDict(self, routes, ends_dict_rf, end_pairs):
        for i, end_dict in ends_dict_rf.items():
            for end_pair in end_pairs:
                if end_dict['point_label'] == end_pair[0]:
                    end_dict['pair_ends'].append(int(end_pair[1]))
                elif end_dict['point_label'] == end_pair[1]:
                    end_dict['pair_ends'].append(int(end_pair[0]))
            route_label = end_dict['route_label']
            ends = routes[route_label]['ends'].copy()
            ends.remove(i)
            end_dict['route_end'] = ends[0]
        return ends_dict_rf


    def combineEndsRoutes(self, routes, ends_dict):
        combEnds_list = []
        for i, end_dict in ends_dict.items():
            # 从没有配对端点的孤立端点出发
            if end_dict['point_type'] == 'iso':
                first_label = end_dict['point_label']
                have_traversed = [first_label]

                def addMultiItemsList(baseList, addList):
                    large_baseList = []
                    for baselist in baseList:
                        for addlist in addList:
                            large_baseList.append(baselist + addlist)
                    return large_baseList

                def getRouteEnd(end_label):
                    next_label = ends_dict[end_label]['route_end']
                    if ends_dict[next_label]['point_type'] == 'iso':
                        have_traversed.append(next_label)
                        return [[next_label]]
                    else:
                        have_traversed.append(next_label)
                        return addMultiItemsList([[next_label]], getPairEnd(next_label))

                def getPairEnd(end_label):
                    next_labels = ends_dict[end_label]['pair_ends']
                    re_baseList = []
                    for next_label in next_labels:
                        if next_label not in have_traversed:
                            have_traversed.append(next_label)
                            re_baseList += addMultiItemsList([[next_label]], getRouteEnd(next_label))
                        else:
                            have_traversed.append(next_label)
                            re_baseList = [[next_label]]
                    return re_baseList

                combEnds_list += addMultiItemsList([[first_label]], getRouteEnd(first_label))
        return combEnds_list


    def combineEndsRoutes_NoCircle(self, routes, ends_dict):
        combEnds_list = []
        for i, end_dict in ends_dict.items():
            # 从没有配对端点的孤立端点出发
            if end_dict['point_type'] == 'iso':
                first_label = end_dict['point_label']

                def addMultiItemsList(baseList, addList):
                    large_baseList = []
                    for baselist in baseList:
                        for addlist in addList:
                            large_baseList.append(baselist + addlist)
                    return large_baseList

                def getRouteEnd(end_label):
                    next_label = ends_dict[end_label]['route_end']
                    if ends_dict[next_label]['point_type'] == 'iso':
                        return [[next_label]]
                    else:
                        return addMultiItemsList([[next_label]], getPairEnd(next_label))

                def getPairEnd(end_label):
                    next_labels = ends_dict[end_label]['pair_ends']
                    re_baseList = []
                    for next_label in next_labels:
                        re_baseList += addMultiItemsList([[next_label]], getRouteEnd(next_label))
                    return re_baseList

                combEnds_list += addMultiItemsList([[first_label]], getRouteEnd(first_label))
        return combEnds_list

    def deleteCloseRoutes(self, routes, ends_dict, combEnds):
        del_list = []
        have_existed = []
        for i, singleRoute in enumerate(combEnds):
            k = 0
            total_len_route = 0
            while k < len(singleRoute):
                total_len_route += len(routes[ends_dict[singleRoute[k]]['route_label']]['route'])
                k += 2
            if total_len_route < 50:
                del_list.append(i)
                continue
            if singleRoute[0] > singleRoute[-1]:
                singleRoute.reverse()
            if ends_dict[singleRoute[-1]]['point_type'] != 'iso' or ends_dict[singleRoute[0]]['point_type'] != 'iso':
                del_list.append(i)
            else:
                if singleRoute in have_existed:
                    del_list.append(i)
                else:
                    have_existed.append(singleRoute)
        del_list.reverse()
        for del_index in del_list:
            del combEnds[del_index]
        return combEnds


    def validFromHSV(self, routes, ends_dict, combEnds, source_img):
        del_list = []
        start_end = {}
        for i, singleRoute in enumerate(combEnds):
            diff_ends = self.costHSV(ends_dict[singleRoute[0]]['end_hsv'], ends_dict[singleRoute[-1]]['end_hsv'])
            diff_route = self.calcRouteScore(routes, singleRoute, ends_dict, source_img)
            if singleRoute[0] not in start_end:
                start_end[singleRoute[0]] = [(i, diff_ends, diff_route)]
            else:
                start_end[singleRoute[0]].append((i, diff_ends, diff_route))
            if singleRoute[-1] not in start_end:
                start_end[singleRoute[-1]] = [(i, diff_ends, diff_route)]
            else:
                start_end[singleRoute[-1]].append((i, diff_ends, diff_route))
        for j, pot_routes in start_end.items():
            if len(pot_routes) == 1:
                continue
            thre = min([float(100 * v[1] + v[2]) for v in pot_routes])
            for pot_route in pot_routes:
                if 100 * pot_route[1] + pot_route[2] > thre:
                    if pot_route[0] not in del_list:
                        del_list.append(pot_route[0])
        del_list.sort()
        del_list.reverse()
        for del_index in del_list:
            del combEnds[del_index]
        return combEnds

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

    def calcRouteScore(self, routes, singleRoute, ends_dict, rgb):
        if len(singleRoute) <= 2:
            return 100
        route_score = 0
        k = 0
        hsv_mid_list = []
        while k < len(singleRoute) - 1:
            route = routes[ends_dict[singleRoute[k]]['route_label']]['route']
            len_route = len(route)
            p_mid = route[len_route//2]
            hsv_p_mid = rgb[p_mid[0]][p_mid[1]]
            # p1 = ends_dict[singleRoute[k]]['point']
            # p2 = ends_dict[singleRoute[k+1]]['point']
            # hsv_p1 = ends_dict[singleRoute[k]]['end_hsv']
            # hsv_p2 = ends_dict[singleRoute[k+1]]['end_hsv']
            # hsv_p12 = tuple([hsv_p1[0]/2 + hsv_p2[0]/2, hsv_p1[1]/2 + hsv_p2[1]/2, hsv_p1[2]/2 + hsv_p2[2]/2])
            hsv_mid_list.append(hsv_p_mid)
            k += 2
        for j in range(1, len(hsv_mid_list)):
            route_score += self.costHSV(hsv_mid_list[0], hsv_mid_list[j])
        return route_score / (len(hsv_mid_list) - 1)



    def showCombRoutes(self, skel, routes, ends_dict, combEnds_list):
        back = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        back_w = np.zeros_like(back)
        color_index = 10
        for j, singleRoute in enumerate(combEnds_list):
            # back_w = np.zeros_like(back)
            turn = True
            for k in range(len(singleRoute)-1):
                point_label = singleRoute[k]
                # p1 = ends_dict[point_label]['point']
                # cv2.circle(back_w, (p1[1], p1[0]), 3, (0, 255, 0))
                line_color = (int(self.cmap[color_index][0]),
                              int(self.cmap[color_index][1]),
                              int(self.cmap[color_index][2]))
                if turn:
                    turn = False
                    route_label = ends_dict[point_label]['route_label']
                    radius = routes[route_label]['width'][0]
                    for point in routes[route_label]['route']:
                        cv2.circle(back_w, (point[1], point[0]), int(radius) + 1, line_color, -1)
                        back_w[point[0]][point[1]] = self.cmap[color_index]
                else:
                    turn = True
                    next_label = singleRoute[k+1]
                    route_label = ends_dict[next_label]['route_label']
                    end_p1 = ends_dict[point_label]['point']
                    end_p2 = ends_dict[next_label]['point']
                    line_thickness = max(int(routes[route_label]['width'][0] * 2 - 2), 3)
                    line_color = (int(self.cmap[color_index][0]), int(self.cmap[color_index][1]), int(self.cmap[color_index][2]))
                    cv2.line(back_w, (end_p1[1], end_p1[0]), (end_p2[1], end_p2[0]), line_color, thickness=line_thickness)
            # cv2.imwrite('data/debug_results/route_test/route_comb_visib_{}.jpg'.format(color_index), back_w)
            color_index += 1
        # cv2.imwrite('data/debug_results/route_test/route_comb_visib.jpg', back_w)
        return back_w

    def showCombRoutes_WithCrossOrder(self, skel, routes, ends_dict, combEnds_list, ints_dict, rgb, dist_img, mask):
        route_pairs_have_drawn = {}
        cross_pairs_have_drawn = {}
        back = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        back_w = np.zeros_like(back)
        color_index = 10
        for j, singleRoute in enumerate(combEnds_list):
            turn = True
            wait_for_redraw = False
            for k in range(len(singleRoute)-1):
                line_color = (int(self.cmap[color_index][0]),
                              int(self.cmap[color_index][1]),
                              int(self.cmap[color_index][2]))
                if turn:
                    turn = False
                    route_pair_ = [singleRoute[k], singleRoute[k+1]]
                    route_pair = tuple(sorted(route_pair_))
                    route_continuity = self.calcRouteContinuity(routes, singleRoute, k, ends_dict, rgb)
                    self.drawRoute(route_pair, ends_dict, routes, line_color, back_w, dist_img)
                    connected_cross_pair = []
                    if k >= 2 and k < len(singleRoute) - 2:
                        cross_pair_l_ = [singleRoute[k - 1], singleRoute[k]]
                        cross_pair_l = tuple(sorted(cross_pair_l_))
                        connected_cross_pair.append(cross_pair_l)
                        cross_pair_r_ = [singleRoute[k + 1], singleRoute[k + 2]]
                        cross_pair_r = tuple(sorted(cross_pair_r_))
                        connected_cross_pair.append(cross_pair_r)
                    elif k < 2 and len(singleRoute) >= 4:
                        cross_pair_r_ = [singleRoute[k + 1], singleRoute[k + 2]]
                        cross_pair_r = tuple(sorted(cross_pair_r_))
                        connected_cross_pair.append(cross_pair_r)
                    elif k >= len(singleRoute) - 2 and len(singleRoute) >= 4:
                        cross_pair_l_ = [singleRoute[k - 1], singleRoute[k]]
                        cross_pair_l = tuple(sorted(cross_pair_l_))
                        connected_cross_pair.append(cross_pair_l)
                    if route_pair not in route_pairs_have_drawn.keys():
                        route_pairs_have_drawn[route_pair] = {'score': route_continuity,
                                                              'line_color': line_color,
                                                              'connected_pairs': connected_cross_pair}
                    else:
                        if route_pairs_have_drawn[route_pair]['score'] > route_continuity:  # 原有路线不如新路线,绘制并覆盖原有路线
                            self.drawRoute(route_pair, ends_dict, routes, line_color, back_w, dist_img)
                            route_pairs_have_drawn[route_pair] = {'score': route_continuity,
                                                                  'line_color': line_color,
                                                                  'connected_pairs': connected_cross_pair}
                        else:
                            wait_for_redraw = True


                else:
                    turn = True
                    cross_pair_ = [singleRoute[k], singleRoute[k + 1]]
                    cross_pair = tuple(sorted(cross_pair_))
                    # 先画一条
                    self.drawCross(cross_pair, ends_dict, routes, line_color, back_w, ints_dict)
                    # 判断
                    cross_continuity = self.calcCrossContinuity(cross_pair, ends_dict, rgb)
                    intersect_pairs = self.CrossExist(cross_pair, cross_pairs_have_drawn, ends_dict)
                    cross_pairs_have_drawn[cross_pair] = {'score': cross_continuity,
                                                          'line_color': line_color}
                    if len(intersect_pairs) > 0:
                        for intersect_pair in intersect_pairs:
                            if cross_pairs_have_drawn[intersect_pair]['score'] < cross_continuity: # 原有交叉比新交叉连续性更好,重新绘制一次原有交叉
                                self.drawCross(intersect_pair, ends_dict, routes, cross_pairs_have_drawn[intersect_pair]['line_color'], back_w, ints_dict)

                    if wait_for_redraw:
                        wait_for_redraw = False
                        self.drawRoute(route_pair, ends_dict, routes, route_pairs_have_drawn[route_pair]['line_color'], back_w, dist_img)
                        for connected_pair in route_pairs_have_drawn[route_pair]['connected_pairs']:
                            self.drawCross(connected_pair, ends_dict, routes, cross_pairs_have_drawn[connected_pair]['line_color'], back_w, ints_dict)

            cv2.imwrite('data/debug_results/route_test/route_comb_visib_{}.jpg'.format(color_index), back_w)
            color_index += 1
        # back_w[mask < 1] = (0, 0, 0)
        return back_w

    def showCombRoutes_Single(self, skel, routes, ends_dict, combEnds_list):
        route_img_list = []
        for j, singleRoute in enumerate(combEnds_list):
            back = np.zeros_like(skel)
            turn = True
            for k in range(len(singleRoute) - 1):
                point_label = singleRoute[k]
                if turn:
                    turn = False
                    route_label = ends_dict[point_label]['route_label']
                    for point in routes[route_label]['route']:
                        back[point[0]][point[1]] = 255
                else:
                    turn = True
                    next_label = singleRoute[k + 1]
                    end_p1 = ends_dict[point_label]['point']
                    end_p2 = ends_dict[next_label]['point']
                    cv2.line(back, (end_p1[1], end_p1[0]), (end_p2[1], end_p2[0]), color=255, thickness=1)
            route_img_list.append(back)
        return route_img_list

    def calcRouteContinuity(self, routes, singleRoute, k, ends_dict, rgb):
        part_route_score = 0
        if len(singleRoute) >= 4:
            k_left = max(0, k - 2)
            k_right = min(k+3, len(singleRoute)-1)
            partRoute = singleRoute[k_left:k_right+1]
            part_route_score = self.calcRouteScore(routes, partRoute, ends_dict, rgb)
        return part_route_score

    def calcCrossContinuity(self, cross_pair, ends_dict, rgb):
        diff_hsv = 0
        p1 = ends_dict[cross_pair[0]]['point']
        p2 = ends_dict[cross_pair[1]]['point']
        p1_2 = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        if self.if_debug:
            top = max(0, min(p1[0]-30, p2[0]-30))
            bottom = min(359, max(p1[0]+30, p2[0]+30))
            left = max(0, min(p1[1]-30, p2[1]-30))
            right = min(639, max(p1[1]+30, p2[1]+30))
            temp_rgb = cv2.cvtColor(rgb, cv2.COLOR_HSV2BGR)
            cv2.circle(temp_rgb, (p1[1], p1[0]), 3, (255, 0, 0), thickness=-1)
            cv2.circle(temp_rgb, (p2[1], p2[0]), 3, (0, 255, 0), thickness=-1)
            cv2.circle(temp_rgb, (p1_2[1], p1_2[0]), 3, (0, 0, 255), thickness=-1)
            cv2.imwrite('data/debug_results/route_test/cross_window.jpg', temp_rgb[top:bottom, left:right])
        hsv_p1 = ends_dict[cross_pair[0]]['end_hsv']
        hsv_p1_ = rgb[p1[0]][p1[1]]
        hsv_p2 = ends_dict[cross_pair[1]]['end_hsv']
        hsv_p1_2 = rgb[p1_2[0]][p1_2[1]]
        diff_hsv += self.costHSV(hsv_p1, hsv_p1_2)
        diff_hsv += self.costHSV(hsv_p1_2, hsv_p2)
        return diff_hsv

    def CrossExist(self, cross_pair, cross_pairs_have_drawn, ends_dict):
        intersect_pairs = []
        p1 = ends_dict[cross_pair[0]]['point']
        p2 = ends_dict[cross_pair[1]]['point']
        for cross_pair_drawn in cross_pairs_have_drawn.keys():
            p3 = ends_dict[cross_pair_drawn[0]]['point']
            p4 = ends_dict[cross_pair_drawn[1]]['point']
            if self.EverCross([p1[0], p1[1], p2[0], p2[1]], [p3[0], p3[1], p4[0], p4[1]]):
                intersect_pairs.append(cross_pair_drawn)
        return intersect_pairs


    def EverCross(self, l1, l2):
        v1 = (l1[0] - l2[0], l1[1] - l2[1])
        v2 = (l1[0] - l2[2], l1[1] - l2[3])
        v0 = (l1[0] - l1[2], l1[1] - l1[3])
        a = v0[0] * v1[1] - v0[1] * v1[0]
        b = v0[0] * v2[1] - v0[1] * v2[0]

        temp = l1
        l1 = l2
        l2 = temp
        v1 = (l1[0] - l2[0], l1[1] - l2[1])
        v2 = (l1[0] - l2[2], l1[1] - l2[3])
        v0 = (l1[0] - l1[2], l1[1] - l1[3])
        c = v0[0] * v1[1] - v0[1] * v1[0]
        d = v0[0] * v2[1] - v0[1] * v2[0]

        if a * b < 0 and c * d < 0:
            return True
        else:
            return False


    def drawRoute(self, end_pair, ends_dict, routes, line_color, back_w, dist_img):
        route_label = ends_dict[end_pair[0]]['route_label']
        radius = routes[route_label]['width'][0]
        for point in routes[route_label]['route']:
            # 非标准绘制模式
            cv2.circle(back_w, (point[1], point[0]), max(round(radius), 1), line_color, -1)
            # 标准绘制模式
            # cv2.circle(back_w, (point[1], point[0]), round(dist_img[point[0]][point[1]]), line_color, -1)
        if self.if_debug:
            cv2.imwrite('data/debug_results/route_test/temp_route.jpg', back_w)
            print(1)


    def drawCross(self, end_pair, ends_dict, routes, line_color, back_w, ints_dict):
        route_label = ends_dict[end_pair[1]]['route_label']
        end_p1 = ends_dict[end_pair[0]]['point']
        end_p2 = ends_dict[end_pair[1]]['point']
        line_thickness = max(int(routes[route_label]['width'][0] * 2 - 1), 3)
        # 标准绘制模式
        cv2.line(back_w, (end_p1[1], end_p1[0]), (end_p2[1], end_p2[0]), line_color, thickness=line_thickness)

        # 非标准绘制模式
        # for int_dict in ints_dict.values():
        #     int_ends = int_dict['int_ends']
        #     if end_pair[0] in int_ends:
        #         int_point = int_dict['point']
        #         break
        # cv2.line(back_w, (end_p1[1], end_p1[0]), (int_point[1], int_point[0]), line_color, thickness=line_thickness)
        # cv2.line(back_w, (int_point[1], int_point[0]), (end_p2[1], end_p2[0]), line_color, thickness=line_thickness)
        if self.if_debug:
            cv2.imwrite('data/debug_results/route_test/temp_route.jpg', back_w)
            print(1)