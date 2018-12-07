import os
from PIL import Image

img1_path='/Users/wywy/Desktop/英文识别测试/img1'
img2_path='/Users/wywy/Desktop/英文识别测试/test1_img'
def resize_picture(img_path):
    for i in os.listdir(img_path):
        if i=='.DS_Store':
            os.remove(img_path+'/'+i)
        else:
            im=Image.open(img_path+'/'+i)
            out=im.resize((280,80),Image.ANTIALIAS)
            out.save(img_path+'/'+i)
resize_picture(img1_path)


