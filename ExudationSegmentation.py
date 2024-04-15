import os
import torch
import torch.nn as nn
from PIL import Image
import torch
from automatic_module.predict import VGGUnet_predict
import cv2
import numpy as np
from automatic_module.automatic_segmentation_model import vgg16bn_unet


class ExudationSegmentation:

    def __init__(self, MODEL_WEIGHT_PATH, imgsize):
        self.image_size = imgsize
        self.modelpath = MODEL_WEIGHT_PATH
        self.softmax = nn.Softmax(dim=1)
        model = vgg16bn_unet(output_dim=2, pretrained=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if os.path.isfile(self.modelpath):
            checkpoint = torch.load(self.modelpath, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
        self.model = model.to(self.device)
        self.model.eval()


    def Forward(self, img_path):
        # org = cv2.imread(img_path)
        org = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
        with torch.set_grad_enabled(False):
            image = Image.open(img_path)
            image = image.resize((1024, 1024))
            image =  image.convert('RGB')
            image = np.array(image)
            if (image.shape[2] == 3):
                image = np.transpose(image, (2, 0, 1))
                image = image / 255
            image = torch.from_numpy(image)
            image = torch.unsqueeze(image, 0)
            image = image.to(device=self.device, dtype=torch.float)
            bs, _, h, w = image.shape
            h_size = (h - 1) // self.image_size + 1
            w_size = (w - 1) // self.image_size + 1
            masks_pred = torch.zeros((1, 2, 1024, 1024)).to(dtype=torch.float)
            for i in range(h_size):
                for j in range(w_size):
                    h_max = min(h, (i + 1) * self.image_size)
                    w_max = min(w, (j + 1) * self.image_size)
                    inputs_part = image[:, :, i * self.image_size:h_max, j * self.image_size:w_max]
                    masks_pred_single = self.model(inputs_part)
                    masks_pred[:, :, i * self.image_size:h_max, j * self.image_size:w_max] = masks_pred_single
            mask_pred_softmax_batch = self.softmax(masks_pred).cpu().numpy()
            mask_soft_batch = mask_pred_softmax_batch[:, 1:, :, :]
            mask_soft_batch = np.squeeze(mask_soft_batch)
            mask_soft_batch = mask_soft_batch * 255
            new_mask = np.zeros((1024, 1024, 3))
            new_mask[:, :, 0] = mask_soft_batch
            new_mask[:, :, 1] = mask_soft_batch
            new_mask[:, :, 2] = mask_soft_batch
        mask = new_mask[:, :, 0]
        mask = cv2.resize(mask, (org.shape[1], org.shape[0]))
        mask[mask > 126] = 255
        mask[mask < 126] = 0
        mask = mask.astype(np.uint8)
        ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        return mask


if __name__ == '__main__':
    pp = 'C:/Users/GIGABYTE/Desktop/my/blood/'
    # main(pp)
    net = ExudationSegmentation(r'models_v1.0.0/VGGUnet_EX.tar', 512)
    for img in os.listdir(pp):
        filename = pp + img
        org = cv2.imread(filename)
        mask = net.Forward(filename)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        drw = cv2.drawContours(org, contours, -1, (0, 255, 0), 1)  # img为三通道才能显示轮廓
        cv2.imwrite(os.path.join(pp, img.replace('.jpg', '_mask.jpg')), mask)
        cv2.imwrite(os.path.join(pp, img.replace('.jpg', '_drw.jpg')), drw)
