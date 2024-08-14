import os, cv2
from CVF3D.core import Pipeline

if __name__ == "__main__":

    img_name = 'c1_32.png'

    IMG_PATH = "data/test_imgs/{}".format(img_name)
    MASK_PATH = "data/test_labels/{}".format(img_name)

    IMG_W = 640
    IMG_H = 360

    ckpt_seg_name = "checkpoints/CP_segmentation.pth"

    p = Pipeline(img_w=IMG_W, img_h=IMG_H, if_debug=True, scene='EWD')

    source_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    source_img = cv2.resize(source_img, (IMG_W, IMG_H))

    mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.resize(mask_img, (IMG_W, IMG_H))

    img_out, _, times, _, _, _ = p.run(source_img=source_img, mask_img=mask_img, mask_th=31)

    cv2.imwrite('data/test_predicts_label/{}'.format(img_name), img_out)
    print(times)