#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project ：Masterarbeit
@File    ：pyq.py
@Author  ：Yu Cao
@Date    ：2021/12/31 16:06 
"""
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def image_add_text(img_path, text, left, top, text_color=(255, 0, 0), text_size=13):
    img = Image.open(img_path)
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式 这里的SimHei.ttf需要有这个字体
    fontStyle = ImageFont.truetype("SimHei.ttf", text_size, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, text_color, font=fontStyle)
    return img


def funk01():
    panoptic = Image.open(r"C:\Users\cyzho\Desktop\pyq\seg\000045_10.png")
    panoptic = panoptic.convert("RGB")
    img = Image.open(r"C:\Users\cyzho\Desktop\pyq\image_2\000045_10.png")
    depth = Image.open(r"C:\Users\cyzho\Desktop\pyq\test_results_kitti2015\000045_10_tmp.png")

    panoptic_np = np.array(panoptic)
    img_np = np.array(img)
    depth_np = np.array(depth)

    h, w,c = img_np.shape

    output = np.zeros_like(img_np)

    output[:,0:w//3,:] = panoptic_np[:,0:w//3,:]
    output[:,w//3:w//3*2,:] = img_np[:,w//3:w//3*2,:]
    output[:,w//3*2:w,:] = depth_np[:,w//3*2:w,:]

    result = Image.fromarray(output)

    text= 'Panoptic'
    left = w//10*1
    top = h//10*8
    text_color = (256, 0, 0)
    draw = ImageDraw.Draw(result)
    # 字体的格式 这里的SimHei.ttf需要有这个字体
    fontStyle = ImageFont.truetype(r"C:\Windows\Fonts\SimHei.ttf", 30, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, text_color, font=fontStyle)

    text= 'Original'
    left = w//7 *3
    draw.text((left, top), text, text_color, font=fontStyle)

    text= 'Depth'
    left = w//10 *8
    draw.text((left, top), text, text_color, font=fontStyle)
    result.show()



    result.save(r"C:\Users\cyzho\Desktop\000045_10.png")

def funk02():
    depth1 = Image.open(r"C:\Users\cyzho\Desktop\pyq\test_results_kitti2015\000045_10.png")
    depth2 = Image.open(r"C:\Users\cyzho\Desktop\pyq\test_results_kitti2015\tmp.png")

    depth1_np = np.array(depth1)
    depth2_np = np.array(depth2)


    h, w, c = depth1_np.shape

    output = np.zeros_like(depth1_np)

    output[:, :, :] = depth1_np[:,:, :]
    #output[0:h//9*5, :, :] = depth2_np[0:h//9*5,:, :]
    output[0:h // 10 * 4, :, :] = 0

    Image.fromarray(output).save(r"C:\Users\cyzho\Desktop\000045_10_tmp.png")


def funk03():
    depth1 = Image.open(r"C:\Users\cyzho\Desktop\000003.jpg")
    depth2 = Image.open(r"C:\Users\cyzho\Desktop\work.png")
    depth2 = depth2.convert("RGB")

    depth1_np = np.array(depth1)
    depth2_np = np.array(depth2)


    h, w, c = depth1_np.shape
    h2, w2, c2 = depth2_np.shape

    output = np.zeros_like(depth1_np)
    # output_new = np.zeros([h, w+w2, 3])

    output[:, w//4:, :] = depth1_np[:, w//4:, :]
    #output[0:h//9*5, :, :] = depth2_np[0:h//9*5,:, :]
    # output[0:h // 10 * 4, :, :] = 0
    # output_new[:, :(w-w//4), :] = output[:, w//4:, :]
    output_new = output[:, w//4:, :]

    depth2_np_new = np.zeros([h, w2, 3])
    depth2_np_new = depth2_np[71:(h2-71),70:(750-90),:]

    r = np.concatenate((output_new, depth2_np_new), axis=1)

    result = Image.fromarray(r)
    result.show()
    result.save(r"C:\Users\cyzho\Desktop\tmp_tmp.png")


if __name__ == "__main__":
    funk03()