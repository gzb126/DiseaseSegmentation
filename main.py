import cv2
import os
import scipy.io as scio
from tqdm import tqdm
import numpy as np
from BloodSegmentation import BloodSegmentation
from ExudationSegmentation import ExudationSegmentation

Blood = BloodSegmentation('models_v1.0.0/u2net_bce_itr_9600.pth', 0)
Exudation = ExudationSegmentation(r'models_v1.0.0/VGGUnet_EX.tar', 512)


def MakeVadFile(jpgfil, vadfile, checklable='Hemorrhage'):

    """
    Parameters
    ----------
    jpgfil  输入的jpg原图
    vadfile 输入的.vad文件
    checklable Lesion_Data_Exudate ||  Lesion_Data_Hemorrhage

    Returns  保存vad文件，更新
    -------
    """
    if checklable == 'Exudate':
        detect = Exudation
    else:
        detect = Blood

    mask = detect.Forward(jpgfil)

    con1, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drw_bl = cv2.drawContours(cv2.imdecode(np.fromfile(jpgfil,dtype=np.uint8),-1), con1, -1, (0, 255, 0), 1)

    cv2.imencode('.jpg', drw_bl )[1].tofile(jpgfil.replace('.jpg', '_draw.jpg'))

    vda_data = scio.loadmat(vadfile)
    print(vda_data.keys())

    mask[mask == 255] = 1

    num = len(vda_data['lesion_data'][0][0])
    vda_data['lesion_data'][0][0][num - 1] = mask

    scio.savemat(vadfile, vda_data, appendmat=True, format='5', long_field_names=False,
                 do_compression=True, oned_as='row')


def main1():

    # fils = r'\\10.10.93.215\公共空间\2023_HXMM\ZAL\test'
    fils = r''
    choose = 'Hemorrhage'   #  Exudate ||  Hemorrhage
    for fil in tqdm(os.listdir(fils)):
        f = os.path.join(fils, fil)
        imgpath = [os.path.join(f, i) for i in os.listdir(f) if '.jpg' in i][0]
        vadpath = [os.path.join(f, i) for i in os.listdir(f) if '.vad' in i][0]
        MakeVadFile(imgpath, vadpath, checklable=choose)


def main2(vadfile, maskimg):
    mask = cv2.imdecode(np.fromfile(maskimg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    ret, trash = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    vda_data = scio.loadmat(vadfile)
    print(vda_data.keys())

    trash[trash == 255] = 1

    num = len(vda_data['lesion_data'][0][0])
    vda_data['lesion_data'][0][0][num - 1] = trash

    scio.savemat(vadfile, vda_data, appendmat=True, format='5', long_field_names=False,
                 do_compression=True, oned_as='row')

def CutImage(imgMat, path):
    H, W = imgMat.shape[:2]
    img_length = 1280
    overlap_W = int(abs(W - 4*img_length)//3)
    overlap_H = int(abs(H - 4*img_length)//3)
    factor = imgMat.copy()
    dist = []
    for ii in range(4):
        for i in range(4):
            mat = factor[(ii*img_length - ii * overlap_H):((ii + 1) * (img_length) - ii * overlap_H),
                  (i * img_length - i * overlap_W):((i + 1) * img_length - i * overlap_W)]
            dist.append((ii, i))
            cv2.imencode('.jpg', mat)[1].tofile(os.path.join(path, '{}_{}_cut.jpg'.format(str(ii), str(i))))
    return (img_length, overlap_W, overlap_H, dist)

def MergeImage(Mats, imgMat, itm):
    (img_length, overlap_W, overlap_H, dist) = itm
    MatMask = np.zeros(imgMat.shape[:2], np.uint8)
    for n, mat in enumerate(Mats):
        MatMask2 = np.zeros(imgMat.shape[:2], np.uint8)
        ii = int(dist[n][0])
        i = int(dist[n][1])
        MatMask2[(ii * img_length - ii * overlap_H):((ii + 1) * (img_length) - ii * overlap_H),
        (i * img_length - i * overlap_W):((i + 1) * img_length - i * overlap_W)] = mat
        MatMask = cv2.add(MatMask, MatMask2)
    MatMask = MatMask.astype(np.uint8)
    return MatMask


def mainCIRCLE():
    root = r'F:\项目测试\circle2/'
    for img in os.listdir(root):
        filename = root + img
        org =cv2.imdecode(np.fromfile(filename,dtype=np.uint8),cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))
        max_id = areas.index(max(areas))
        x, y, w, h = cv2.boundingRect(contours[max_id])
        imgMat = org[y:(y+h), x:(x+w)]
        path = img.replace('.jpg', '')
        if not os.path.exists(os.path.join(root, path)):
            os.makedirs(os.path.join(root, path))
        itm = CutImage(imgMat, os.path.join(root, path))
        blfil, exfil = [], []
        for im in os.listdir(os.path.join(root, path)):
            bl = Blood.Forward(os.path.join(root, path, im))
            ex = Exudation.Forward(os.path.join(root, path, im))
            blfil.append(bl)
            exfil.append(ex)
            cv2.imencode('.jpg', bl)[1].tofile(os.path.join(root, path, im.replace('cut', 'bl')))
            cv2.imencode('.jpg', ex)[1].tofile(os.path.join(root, path, im.replace('cut', 'ex')))
        bltrash = MergeImage(blfil, imgMat, itm)
        extrash = MergeImage(exfil, imgMat, itm)
        cv2.imencode('.jpg', bltrash)[1].tofile(os.path.join(root, path, 'merge_bl.jpg'))
        cv2.imencode('.jpg', extrash)[1].tofile(os.path.join(root, path, 'merge_ex.jpg'))
        bl, ex = np.zeros(org.shape[:2], np.uint8), np.zeros(org.shape[:2], np.uint8)
        bl[y:(y+h), x:(x+w)] = bltrash
        ex[y:(y+h), x:(x+w)] = extrash
        cv2.imencode('.jpg', bl)[1].tofile(os.path.join(root, path, 'merge_bl_big.jpg'))
        cv2.imencode('.jpg', ex)[1].tofile(os.path.join(root, path, 'merge_ex_big.jpg'))
        con1, _ = cv2.findContours(bl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        con2, _ = cv2.findContours(ex, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        drw_bl = cv2.drawContours(org, con1, -1, (0, 0, 255), 1)  # img为三通道才能显示轮廓
        drw_ex = cv2.drawContours(drw_bl, con2, -1, (0, 255, 0), 1)  # img为三通道才能显示轮廓
        cv2.imencode('.jpg', drw_ex)[1].tofile(os.path.join(root, img.replace('.jpg', '_drw.jpg')))

def mainCompare():
    pp = r'F:\项目测试\circle/'
    for img in os.listdir(pp):
        filename = pp + img
        org =cv2.imdecode(np.fromfile(filename,dtype=np.uint8),-1)
        bl = Blood.Forward(filename)
        ex = Exudation.Forward(filename)
        con1, _ = cv2.findContours(bl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        con2, _ = cv2.findContours(ex, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        drw_bl = cv2.drawContours(org, con1, -1, (0, 0, 255), 1)  # img为三通道才能显示轮廓
        drw_ex = cv2.drawContours(drw_bl, con2, -1, (0, 255, 0), 1)  # img为三通道才能显示轮廓

        if '.jpg' in img:
            cv2.imencode('.jpg', drw_ex)[1].tofile(os.path.join(pp, img.replace('.jpg', '_drw2.jpg')))
            cv2.imencode('.jpg', bl)[1].tofile(os.path.join(pp, img.replace('.jpg', '_blood.jpg')))
            cv2.imencode('.jpg', ex)[1].tofile(os.path.join(pp, img.replace('.jpg', '_exuda.jpg')))
        if '.jpeg' in img:
            cv2.imencode('.jpg', drw_ex)[1].tofile(os.path.join(pp, img.replace('.jpeg', '_drw2.jpeg')))
            cv2.imencode('.jpg', bl)[1].tofile(os.path.join(pp, img.replace('.jpeg', '_blood.jpeg')))
            cv2.imencode('.jpg', ex)[1].tofile(os.path.join(pp, img.replace('.jpeg', '_exuda.jpeg')))


if __name__ == '__main__':
    # mainCIRCLE()
    mainCompare()

    # vadfile = r'\\Desktop-7nns3lm\g\ZAL\14_\出血\org_2023-12-08_163324_0251/Lesion_Data_Hemorrhage.vad'
    # maskimg = r'\\Desktop-7nns3lm\g\ZAL\14_\出血\org_2023-12-08_163324_0251/blood.jpg'
    # main2(vadfile, maskimg)
    #
