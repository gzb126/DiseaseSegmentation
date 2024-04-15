import argparse
import json
import os
import os.path as osp
import warnings

import PIL.Image
import yaml

from labelme import utils
import base64

import numpy as np
import os
from PIL import Image

np.set_printoptions(threshold=np.inf)

def main():
    warnings.warn("This script is aimed to demonstrate how to convert the\n"
                  "JSON file to a single image dataset, and not to handle\n"
                  "multiple JSON files to generate a real-use dataset.")

    json_file = r'G:\2_YH\Diseas\EX\json'

    count = os.listdir(json_file)
    for i in range(0, len(count)):
        path = os.path.join(json_file, count[i])
        if os.path.isfile(path):
            data = json.load(open(path))

            ##############################
            # save_diretory
            out_dir1 = osp.basename(path).replace('.', '_')
            save_file_name = out_dir1
            out_dir1 = osp.join(osp.dirname(path), out_dir1)

            if not osp.exists(json_file + '\\' + 'labelme_json'):
                os.mkdir(json_file + '\\' + 'labelme_json')
            labelme_json = json_file + '\\' + 'labelme_json'

            out_dir2 = labelme_json + '\\' + save_file_name
            if not osp.exists(out_dir2):
                os.mkdir(out_dir2)

            #########################

            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))

            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            captions = ['{}: {}'.format(lv, ln)
                        for ln, lv in label_name_to_value.items()]
            lbl_viz = utils.draw_label(lbl, img, captions)

            PIL.Image.fromarray(img).save(out_dir2 + '\\' + save_file_name + '_img.png')
            # PIL.Image.fromarray(lbl).save(osp.join(out_dir2, 'label.png'))
            utils.lblsave(osp.join(out_dir2, save_file_name + '_label.png'), lbl)
            PIL.Image.fromarray(lbl_viz).save(out_dir2 + '\\' + save_file_name +
                                              '_label_viz.png')

            with open(osp.join(out_dir2, 'label_names.txt'), 'w') as f:
                for lbl_name in label_names:
                    f.write(lbl_name + '\n')

            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=label_names)
            with open(osp.join(out_dir2, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)

            # save png to another directory
            if not osp.exists(json_file + '\\' + 'mask_png'):
                os.mkdir(json_file + '\\' + 'mask_png')
            mask_save2png_path = json_file + '\\' + 'mask_png'

            utils.lblsave(osp.join(mask_save2png_path, save_file_name + '_label.png'), lbl)

            print('Saved to: %s' % out_dir2)


np.set_printoptions(threshold=np.inf)
# 这个函数用于将红色标签图转为白的的标签（其实红色的标签表示灰度值为1(也是只有一个通道）），但不知道为何会显示出红色
def RedToWhite(img_dir, new_img_dir):
    folders = os.listdir(img_dir)  # 得img_dir中所有文件的名字

    for floder in folders:
        image_path = os.path.join(img_dir, floder)
        img = Image.open(image_path)  # 打开图片
        newImg = np.array(img) * 255  # 红色的标签表示灰度值为1,乘以255后都变为255
        newImg = newImg.astype(np.uint8)
        newImg = Image.fromarray(newImg)
        newImg_path = os.path.join(new_img_dir, floder.replace('_json_label', ''))
        # newImg_path = os.path.join(new_img_dir, floder)
        newImg.save(newImg_path)


def TwoImgAdd():
    import cv2
    soft = r'C:\Users\GIGABYTE\Downloads\IDRiD\Groundtruths\Soft'
    hard = r'C:\Users\GIGABYTE\Downloads\IDRiD\Groundtruths\Hard'
    out = r'G:\2_YH\Diseas\EX2'
    sList = os.listdir(soft)
    hList = os.listdir(hard)
    for himg in hList:
        ex = himg.replace('EX', 'SE')
        if ex in sList:
            img = cv2.imread(os.path.join(hard, himg), 0) + cv2.imread(os.path.join(soft, ex), 0)
            img[img>0] == 255
            img[img<=0] == 0
            cv2.imwrite(os.path.join(out, himg.replace('tif', 'png')), img)
        else:
            img = cv2.imread(os.path.join(hard, himg), 0)
            cv2.imwrite(os.path.join(out, himg.replace('tif', 'png')), img)


if __name__ == '__main__':
    # main()  # 1

    img_path = r'G:\2_YH\Diseas\EX\json\mask_png'
    newImg_path = r'G:\2_YH\Diseas\EX\img'
    RedToWhite(img_path, newImg_path)

    # TwoImgAdd()