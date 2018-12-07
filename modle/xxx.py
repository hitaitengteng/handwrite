#生成英文单词，并标记label

from captcha.image import ImageCaptcha
from PIL import Image
import os
import random
import numpy as np

#
lable_path='./label.txt'
save_path='/Users/wywy/Desktop/英文识别测试/img'

#
f=open(lable_path,'r')
data=f.readlines()
for name in data:
    name1=name.rstrip()
    if len(name1)==0:
        continue
    for i in range(6):
        lable=name1
        img = ImageCaptcha()
        image = img.generate_image(lable)
        out = image.resize((180, 60), Image.ANTIALIAS)
        out.save(save_path+'/'+str(i)+'_'+lable+'.jpg')


# lable_path='./test2.txt'
# letter='abcdefghijklmnopqrstuvwxyz'
#
# fp=open(lable_path,'w+')
# for number in range(3000):
#     range_number=random.randint(3,8)
#     print(range_number)
#     str_letter=''
#     for i in range(range_number):
#         index=random.randint(0,len(list(letter))-1)
#         str_letter+=list(letter)[index]
#     fp.write(str_letter+'\n')





