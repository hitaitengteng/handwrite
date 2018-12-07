import os
import numpy as np
from PIL import Image
import random
import cv2

#生成对应汉字的编号
# img_path='/Users/wywy/Desktop/chineses_data'
def dict_infor(img_path):
    all_name=[]
    all_label=[]
    for file in os.listdir(img_path):
        if file == '.DS_Store':
            os.remove(img_path + '/' + file)
        else:
            name=file.split('.')[0].split('_')[1:]
            for c in name:
                all_name.append(c)
    set_name=list(set(all_name))
    for label_index in range(len(set_name)):
        all_label.append(label_index)
    dict_info=dict(zip(set_name,all_label))

    return dict_info

# dict_info=dict_infor(img_path)

#生成2-4个名
# save_path='/Users/wywy/Desktop/test_resize1'
def resize_img(img_path,save_path):
    all_img_path=[]
    all_img_size=[]
    all_img_name=[]
    for file in os.listdir(img_path):
        if file == '.DS_Store':
            os.remove(img_path + '/' + file)
        else:
            name = file.split('.')[0].split('_')[-1]
            all_img_name.append(name)
            img_path1=img_path+'/'+file
            all_img_path.append(img_path1)
            img=cv2.imread(img_path1)
            img_size=img.shape[0],img.shape[1]
            all_img_size.append(img_size)

    for img_number in range(80000):
        range_number=random.randint(2,4)
        bg = Image.new('RGB', (224 * 2, 168), 'white')
        start_x, start_y = 0, 0
        name1=''
        for xx in range(range_number):
            random_index = random.randint(0, len(all_img_path)-1)

            random_img_path=all_img_path[random_index]
            random_img=Image.open(random_img_path)
            bg.paste(random_img,(start_x,start_y))
            w, h=all_img_size[random_index][1],all_img_size[random_index][0]
            start_x+=(w+2)
            name11=all_img_name[random_index]

            name1+=str(name11)

            if start_x>224*2:
                pass
        # print(save_path+'/'+str(img_number)+'_'+name1+'.jpg')
        bg.save(save_path+'/'+str(img_number)+'_'+name1+'.jpg')

# resize_img(img_path,save_path)


# img_path='/Users/wywy/Desktop/test_resize'
save_path='/Users/wywy/Desktop/test_resize1'

def resize_picture(img_path,save_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file)
            out=img.resize((180,80),Image.ANTIALIAS)
            out.save(save_path+'/'+file)
resize_picture(save_path,save_path)




















