# -*- coding : utf-8 -*-
# coding: utf-8
import cv2
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from u2net.u2net import U2NET
from PIL import Image
from skimage import io, transform, color

class U2NetPredict():

    def __init__(self):
        self.net = U2NET(3, 1)
        self.trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_model(self, model_path, gpu_id):
        self.device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.net.to(self.device)
        else:
            self.net.load_state_dict(torch.load(model_path, map_location='cpu'))
        with torch.no_grad():
            self.net.eval()

    def preprocess(self, imgmat):
        if imgmat.ndim == 2:
            imgmat = cv2.cvtColor(imgmat, cv2.COLOR_GRAY2RGB)
        if imgmat.shape[0] != 320:
            imgmat = transform.resize(imgmat, (320, 320), mode='constant')

        test_salobj_dataset = self.trans(imgmat / np.max(imgmat))

        test_salobj_dataloader = DataLoader([test_salobj_dataset])
        for i_test, data_test in enumerate(test_salobj_dataloader):
            inputs_test = data_test.type(torch.FloatTensor)
            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.to(self.device))
            else:
                inputs_test = Variable(inputs_test)
        return inputs_test

    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn

    def Forward(self, imgmat):
        testdata = self.preprocess(imgmat)
        with torch.no_grad():
            d1, d2, d3, d4, d5, d6, d7 = self.net(testdata)
        pred = d1[:, 0, :, :]
        pred = self.normPRED(pred)
        return pred

    def result(self, orgimg, preimg):
        predict = preimg.squeeze()
        predict_np = predict.cpu().data.numpy()
        im = Image.fromarray(predict_np * 255).convert('RGB')
        imo = im.resize((orgimg.shape[1], orgimg.shape[0]), resample=Image.BILINEAR)
        image = cv2.cvtColor(np.asarray(imo), cv2.COLOR_RGB2GRAY)
        ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return image



def total_predict(ori_image, net, side):
    h_step = ori_image.shape[0] // side
    w_step = ori_image.shape[1] // side

    h_rest = -(ori_image.shape[0] - side * h_step)
    w_rest = -(ori_image.shape[1] - side * w_step)

    image_list = []
    predict_list = []
    # 循环切图
    for h in range(h_step):
        for w in range(w_step):
            # 划窗采样
            image_sample = ori_image[(h * side):(h * side + side), (w * side):(w * side + side), :]
            image_list.append(image_sample)
        image_list.append(ori_image[(h * side):(h * side + side), -side:, :])
    for w in range(w_step - 1):
        image_list.append(ori_image[-side:, (w * side):(w * side + side), :])
    image_list.append(ori_image[-side:, -side:, :])

    # 对每个图像块预测
    # predict
    for i, image in enumerate(image_list):
        pred = net.Forward(image)
        predict = net.result(image, pred)

        # cv2.imwrite(r'C:\Users\GIGABYTE\Desktop\exout/' + '{}_org.jpg'.format(str(i)), image)
        # cv2.imwrite(r'C:\Users\GIGABYTE\Desktop\exout/' + 'mak.png', predict)

        predict_list.append(predict)

    # 将预测后的图像块再拼接起来
    count_temp = 0
    tmp = np.ones([ori_image.shape[0], ori_image.shape[1]])
    for h in range(h_step):
        for w in range(w_step):
            tmp[h * side:(h + 1) * side, w * side:(w + 1) * side] = predict_list[count_temp]
            count_temp += 1
        tmp[h * side:(h + 1) * side, w_rest:] = predict_list[count_temp][:, w_rest:]
        count_temp += 1
    for w in range(w_step - 1):
        tmp[h_rest:, (w * side):(w * side + side)] = predict_list[count_temp][h_rest:, :]
        count_temp += 1
    tmp[-(side+1):-1, -(side+1):-1] = predict_list[count_temp][:, :]
    tmp = np.array(tmp, np.uint8)
    return tmp


if __name__ == '__main__':
    net = U2NetPredict()
    net.get_model(r'F:\1_PycharmProjects\Pytorch_U2net\saved_models\u2net_2\u2net_bce_itr_265000_train_0.235691_tar_0.009173.pth', 0)
    imgroot = r'C:\Users\GIGABYTE\Desktop\extest'
    for img in os.listdir(imgroot):
        path = os.path.join(imgroot, img)
        mat = cv2.imread(path)
        out = total_predict(mat, net, 360)
        contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        drw = cv2.drawContours(mat, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(r'C:\Users\GIGABYTE\Desktop\exout/' + img.replace('jpg', 'png'), drw)