import cv2
import time
import matplotlib.pyplot as plt
import numpy as np


def draw_features(depth, width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    #for i in range(width * height):
    for i in range(depth):
        plt.subplot(16, depth/16, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        pre_max = 75.0
        pre_min = -25.0
        if pmax > pre_max:
            raise ValueError("predefine max not enough for current: ", pmax)
        if pmin < pre_min:
            raise ValueError("predefine min not enough for current: ", pmin)

        img = ((img - pre_min) / (pre_max - pre_min)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i, depth))
    fig.savefig(savename, dpi=300)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))
