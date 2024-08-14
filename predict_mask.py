from CVF3D.segment.predict import SegNet
import os
import cv2
from tqdm import tqdm


ckpt_seg_name = "checkpoints/best_deeplabv3plus_resnet101_dlos_os16_motion.pth"

network_seg = SegNet(model_name="deeplabv3plus_resnet101", checkpoint_path=ckpt_seg_name, img_w=896, img_h=504)

img_list = os.listdir("data/LAB_imgs_0724_DLO/pic")

for img_name in tqdm(img_list):

    source_img = cv2.imread(os.path.join("data/LAB_imgs_0724_DLO/pic", img_name), cv2.IMREAD_UNCHANGED)
    mask_img = network_seg.predict_img(source_img)

    cv2.imwrite(os.path.join("data/LAB_imgs_0724_DLO/mask_predict", img_name), mask_img)