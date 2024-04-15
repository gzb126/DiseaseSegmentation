import scipy.io as scio
import numpy as np #导入矩阵处理库
import cv2


dat = r'\\10.10.93.215\公共空间\2023_HXMM\ZAL\Exudate\2023-11-21_11-43-56\raw_data/Lesion_Data_Exudate_re.vad'
data=scio.loadmat(dat)

key = data.keys()
print(key)

print(data['__header__'])
print(data['__version__'])
print(data['__globals__'])

a = data['lesion_data']

num = len(data['lesion_data'][0][0])
python_y = data['lesion_data'][0][0][num - 1]*255
cv2.imwrite(r'C:\Users\GIGABYTE\Desktop\my/e.jpg', python_y)


