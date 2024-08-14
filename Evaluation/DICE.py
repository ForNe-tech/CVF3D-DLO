import os
import cv2
import numpy as np

def dice_loss(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    # union_ = pred + target
    # union_[union_ > 0] = 1
    # union = union_.sum()
    union = pred.sum() + target.sum()
    dice_score = (2 * intersection + epsilon) / (union + epsilon)
    # dice_loss = 1 - dice_score
    return dice_score

if __name__ == "__main__":

    img_list = os.listdir('../data/test_labels')
    dice_dict = {}
    for img_name in img_list:
        pred = cv2.imread(os.path.join('../data/test_predicts_mask', img_name), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(os.path.join('../data/test_labels', img_name), cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, (640, 360))
        pred[pred < 10] = 0
        pred[pred != 0] = 1
        target[target < 10] = 0
        target[target != 0] = 1
        dice = dice_loss(pred, target)
        dice_dict[img_name] = dice
        print(img_name + ':' + str(dice))

    avg_dice = np.mean([v for v in dice_dict.values()])
    avg_dice_C1 = np.mean([v for key, v in dice_dict.items() if key[1] == '1'])
    avg_dice_C2 = np.mean([v for key, v in dice_dict.items() if key[1] == '2'])
    avg_dice_C3 = np.mean([v for key, v in dice_dict.items() if key[1] == '3'])
    print("avg_dice:", avg_dice)
    print("avg_dice_C1:", avg_dice_C1)
    print("avg_dice_C2:", avg_dice_C2)
    print("avg_dice_C3:", avg_dice_C3)