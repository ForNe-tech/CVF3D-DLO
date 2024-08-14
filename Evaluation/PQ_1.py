import os
from tqdm import tqdm
import cv2
import numpy as np


def get_instance_nums(img):
    IMG_H, IMG_W = img.shape[0], img.shape[1]
    instance_colors = []

    for row in range(IMG_H):
        for col in range(IMG_W):
            p_color = tuple([img[row][col][0], img[row][col][1], img[row][col][2]])
            if p_color not in instance_colors and p_color != (0, 0, 0):
                instance_colors.append(p_color)

    black = np.zeros((IMG_H, IMG_W))
    instance_imgs = {}
    for i, instance_color in enumerate(instance_colors):
        instance_imgs[instance_color] = black.copy()

    for row in range(IMG_H):
        for col in range(IMG_W):
            p_color = tuple([img[row][col][0], img[row][col][1], img[row][col][2]])
            if p_color != (0, 0, 0):
                instance_imgs[p_color][row][col] = 1

    return len(instance_colors), instance_imgs

def PQ(pred, target, epsilon=1e-6):
    intersection_ = pred * target
    intersection = intersection_.sum()
    union_ = pred + target
    union_[union_ > 0] = 1
    union = union_.sum()
    IoU_score = (intersection + epsilon) / (union + epsilon)
    merge = intersection_ * 127 + union_ * 127
    if intersection > 100 and union > 100:
        cv2.imwrite('intersection_union_{}_{}_{}_{}.png'.format(pred.sum(), target.sum(), intersection, union), merge)
    return intersection * IoU_score

def getUnion(pred, target):
    pred_g = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    target_g = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    pred_g[pred_g > 0] = 1
    target_g[target_g > 0] = 1
    union_g = pred_g + target_g
    union_g[union_g > 0] = 1
    return union_g.sum(), pred_g.sum()


if __name__ == "__main__":

    img_list = os.listdir('../data/test_labels')

    img_name = 'c1_29.png'

    pred = cv2.imread(os.path.join('../data/test_predicts_mask', img_name), cv2.IMREAD_UNCHANGED)
    target = cv2.imread(os.path.join('../data/test_labels', img_name), cv2.IMREAD_UNCHANGED)
    pred_h, pred_w = pred.shape[0], pred.shape[1]
    target_h, target_w = target.shape[0], target.shape[1]
    size_h = min(pred_h, target_h)
    size_w = min(pred_w, target_w)
    pred = pred[0:size_h, 0:size_w]
    target = target[0:size_h, 0:size_w]
    
    pred_show = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    # pred_show = pred.copy()
    pred_show[pred_show < 10] = 0
    pred_show[pred_show != 0] = 192
    target_show = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # target_show = target.copy()
    target_show[target_show < 10] = 0
    target_show[target_show != 0] = 63
    cv2.imwrite('pred_target_show.png', pred_show + target_show)

    pred_instance_num, pred_instances = get_instance_nums(pred)
    target_instance_num, target_instances = get_instance_nums(target)
    # for pred in pred_instances.values():
    #     cv2.imwrite('pred.png', pred * 255)

    pixel_N, pixel_pred = getUnion(pred, target)

    PQ_sum = []

    for pred in pred_instances.values():
        PQ_score_possible = []
        for target in target_instances.values():
            PQ_score_possible.append(PQ(pred, target))
        PQ_score = max(PQ_score_possible)
        PQ_sum.append(PQ_score)

    print(PQ_sum, pixel_N, pixel_pred)
    print(img_name + ' PQ_score:', sum(PQ_sum) / pixel_N)