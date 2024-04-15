import cv2
import os
import numpy as np

def ImageDraw(org, ex, blod):
    orgimg = cv2.imdecode(np.fromfile(org, dtype=np.uint8), -1)
    eximg = cv2.imdecode(np.fromfile(ex, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    blodimg = cv2.imdecode(np.fromfile(blod, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    ret1, ret_ex = cv2.threshold(eximg, 60, 255, cv2.THRESH_BINARY)
    ret2, ret_bl = cv2.threshold(blodimg, 60, 255, cv2.THRESH_BINARY)


    con1, _ = cv2.findContours(ret_bl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drw_bl = cv2.drawContours(orgimg, con1, -1, (0, 0, 255), 2)  # img为三通道才能显示轮廓

    con2, _ = cv2.findContours(ret_ex, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drw_ex = cv2.drawContours(drw_bl, con2, -1, (0, 255, 0), 2)  # img为三通道才能显示轮廓

    cv2.imencode('.jpg', drw_ex)[1].tofile(org.replace('.jpg', '_drw.jpg'))


if __name__ == '__main__':
    org = r'F:\项目测试\14_/org.jpg'
    ex = r'F:\项目测试\14_/exud.jpg'
    blod = r'F:\项目测试\14_/blood.jpg'
    ImageDraw(org, ex, blod)

