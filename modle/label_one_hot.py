import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import  Dataset


#稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def lable_dict(cls_nnumber):
    letter=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    number=[]
    for i in range(len(letter)):
        number.append(i)

    dic_info = dict(zip(letter, number))

    return dic_info
dict=lable_dict(27)


img_path='/Users/wywy/Desktop/英文识别测试/img1'
test_path='/Users/wywy/Desktop/英文识别测试/test1_img'
def get_info(img_path,dict_info):
    all_img=[]
    all_label=[]

    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            name=list(file.split('.')[0].split('_')[-1])
            img=cv2.imread(img_path+'/'+file,0)
            all_img.append(img)
            label=[]
            for i in name:
                label.append(dict_info.get(i))
            all_label.append(label)
    return all_img,all_label

all_img,all_label=get_info(img_path,dict)

test_img,test_label=get_info(test_path,dict)

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

        return np.array(self.batch_img),self.batch_label

infordata=InforData(all_img,all_label)
testdata=InforData(test_img,test_label)



if __name__=='__main__':
    img,labels=infordata.__getitem__(10)
    label_len=[]
    for i in labels:
        label_len.append(len(i))
    print(label_len)

    indices, values, shape=sparse_tuple_from(labels)





