import os, cv2
from CVF3D.core import Pipeline
import numpy as np

if __name__ == "__main__":

    img_list = os.listdir("data/test_imgs")

    total_time = {}
    seg_time = {}
    proc_time = {}

    IMG_W = 640
    IMG_H = 360

    ckpt_seg_name = "checkpoints/CP_segmentation.pth"

    p = Pipeline(checkpoint_seg=ckpt_seg_name, img_w=IMG_W, img_h=IMG_H, if_debug=False, scene='EWD')

    for img_name in img_list:

        print(img_name)

        IMG_PATH = "data/test_imgs/{}".format(img_name)
        MASK_PATH = "data/test_labels/{}".format(img_name)

        source_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        source_img = cv2.resize(source_img, (IMG_W, IMG_H))

        mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img, (IMG_W, IMG_H))

        img_out, _, times, _, _, _ = p.run(source_img=source_img, mask_img=None, mask_th=31)
        print(times)
        total_time[img_name] = times['tot_time'] + times['seg_time']
        seg_time[img_name] = times['seg_time']
        proc_time[img_name] = times['tot_time']

        # cv2.imwrite('data/test_predicts_label/{}'.format(img_name), img_out)

    del total_time['c1_0.png']
    avg_time = np.mean([v for v in total_time.values()])
    avg_time_C1 = np.mean([v for key, v in total_time.items() if key[1] == '1'])
    avg_time_C2 = np.mean([v for key, v in total_time.items() if key[1] == '2'])
    avg_time_C3 = np.mean([v for key, v in total_time.items() if key[1] == '3'])
    print("avg_time:", avg_time, "FPS:", 1000 / avg_time)
    print("avg_time_C1:", avg_time_C1, "FPS:", 1000 / avg_time_C1)
    print("avg_time_C2:", avg_time_C2, "FPS:", 1000 / avg_time_C2)
    print("avg_time_C3:", avg_time_C3, "FPS:", 1000 / avg_time_C3)

    del seg_time['c1_0.png']
    avg_time = np.mean([v for v in seg_time.values()])
    avg_time_C1 = np.mean([v for key, v in seg_time.items() if key[1] == '1'])
    avg_time_C2 = np.mean([v for key, v in seg_time.items() if key[1] == '2'])
    avg_time_C3 = np.mean([v for key, v in seg_time.items() if key[1] == '3'])
    print("avg_time:", avg_time, "FPS:", 1000 / avg_time)
    print("avg_time_C1:", avg_time_C1, "FPS:", 1000 / avg_time_C1)
    print("avg_time_C2:", avg_time_C2, "FPS:", 1000 / avg_time_C2)
    print("avg_time_C3:", avg_time_C3, "FPS:", 1000 / avg_time_C3)

    del proc_time['c1_0.png']
    avg_time = np.mean([v for v in proc_time.values()])
    avg_time_C1 = np.mean([v for key, v in proc_time.items() if key[1] == '1'])
    avg_time_C2 = np.mean([v for key, v in proc_time.items() if key[1] == '2'])
    avg_time_C3 = np.mean([v for key, v in proc_time.items() if key[1] == '3'])
    print("avg_time:", avg_time, "FPS:", 1000 / avg_time)
    print("avg_time_C1:", avg_time_C1, "FPS:", 1000 / avg_time_C1)
    print("avg_time_C2:", avg_time_C2, "FPS:", 1000 / avg_time_C2)
    print("avg_time_C3:", avg_time_C3, "FPS:", 1000 / avg_time_C3)