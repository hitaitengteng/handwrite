import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import  Dataset


def lable_dict(cls_nnumber):
    letter=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0']
    data_onehot = []
    for i in range(len(letter)):
        data_onehot1 = []
        for j in range(len(letter)):
            data_onehot1.append(0)
        data_onehot1[i]=1
        data_onehot.append(data_onehot1)
    dic_info = dict(zip(letter, np.array(data_onehot)))

    return dic_info
dict=lable_dict(27)

img_path='/Users/wywy/Desktop/英文识别测试/img'
test_path='/Users/wywy/Desktop/英文识别测试/test_img'
def get_info(dict_info,img_path):
    all_number_label=[]
    all_img=[]
    for file in os.listdir(img_path):
        lable=file.split('.')[0].split('_')[-1]
        lable_list=list(lable)

        img=cv2.imread(img_path+'/'+file)/255-0.5
        # img=img.transpose((1,0,2))

        # plt.imshow(img)
        # plt.show()

        all_img.append(img)
        num_lable=[]

        for letter in lable_list:
            num=dict_info[letter]
            num_lable.append(num)
        if len(num_lable)<20:
            for j in range(20-len(num_lable)):
                num_lable.append(dict_info['0'])

        all_number_label.append(np.array(num_lable))

    return all_img,np.array(all_number_label)

all_img,label=get_info(dict,img_path)
test_img,test_label=get_info(dict,test_path)

class InforData(Dataset):
    def __init__(self,all_img,label):
        self.all_img=all_img
        self.all_label=label

    def __len__(self):
        return len(self.all_img)
    def __getitem__(self, batch_size):
        self.batch_img=[]
        self.batch_label=[]
        for i in range(batch_size):
            index=np.random.randint(len(self.all_img))
            self.batch_img.append(self.all_img[index])
            self.batch_label.append(self.all_label[index])

        return np.array(self.batch_img),np.array(self.batch_label)

infordata=InforData(all_img,label)
testdata=InforData(test_img,test_label)



if __name__=='__main__':
    for i in range(10):
        img,labels=testdata.__getitem__(10)
        print(img[0].shape)
