import random

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def CupImg(root, out, num = 8):
    imgs = os.listdir(root)
    for img in imgs:
        if '.png' in img:
            jpg = os.path.join(root, img.replace('png', 'jpg'))
            png = os.path.join(root, img)
            Mat_label = cv2.imread(png, 0)
            Mat_img = cv2.imread(jpg, 1)
            h, w = Mat_label.shape
            s = [360, 420, 512]
            random.shuffle(s)
            side = s[0]
            num_h = h // side
            num_w = w // side
            for h in range(0, num_h):
                for w in range(0, num_w):
                    crop_label = Mat_label[h * side:(h + 1) * side, w * side:(w + 1) * side]
                    if cv2.countNonZero(crop_label) != 0:
                        n = len(os.listdir(out))
                        crop_img = Mat_img[h * side:(h + 1) * side, w * side:(w + 1) * side]
                        cv2.imwrite(os.path.join(out, str(n+1) + '_ex.jpg'), crop_img)
                        cv2.imwrite(os.path.join(out, str(n+1) + '_ex.png'), crop_label)


def Cupimg2(root, out):
    imgs = os.listdir(root)
    for img in imgs:
        if '.png' in img:
            jpg = os.path.join(root, img.replace('png', 'jpg'))
            png = os.path.join(root, img)
            Mat_label = cv2.imread(png, 0)
            Mat_img = cv2.imread(jpg, 1)
            H, W = Mat_label.shape
            contours, hierarchy = cv2.findContours(Mat_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                s = [380, 320, 420]
                random.shuffle(s)
                side = s[0]
                x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
                px, py = int(x + w/2), int(y + h/2)
                xx, yy = px - side//2, py - side//2
                hh, ww = side, side
                if xx < 0:
                    xx = 0
                if yy < 0:
                    yy = 0
                if xx + side//2 > W:
                    ww = W - xx
                if yy + side//2 > H:
                    hh = H - yy
                crop_label = Mat_label[yy:(yy+hh), xx:(xx+ww)]
                crop_img = Mat_img[yy:(yy+hh), xx:(xx+ww)]
                n = len(os.listdir(out))
                cv2.imwrite(os.path.join(out, str(n + 1) + '_ex.jpg'), crop_img)
                cv2.imwrite(os.path.join(out, str(n + 1) + '_ex.png'), crop_label)


if __name__ == '__main__':
    root = r'G:\2_YH\Diseas\EX2'
    out = r'F:\3_Data\ImageSegmentation\ex\crop'
    # CupImg(root, out)
    Cupimg2(root, out)