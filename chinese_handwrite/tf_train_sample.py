import numpy as np
import os
import cv2
from torch.utils.data import  Dataset
from PIL import Image
import  tensorflow as tf

img_path='/Users/wywy/Desktop/test_resize'

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



#获取所有图片和标签
def get_all_data(img_path):
    """
        Get all pictures and labels
        Args:
            img_path: Path for image
        Returns:
            all_img：A numpy array
            all_label：A list
        """
    all_label=[]
    all_img=[]
    for file in os.listdir(img_path):
        if file == '.DS_Store':
            os.remove(img_path + '/' + file)
        else:
            name=file.split('.')[0].split('_')[1:]
            img=cv2.imread(img_path+'/'+file,0)
            all_label.append(name)
            all_img.append(img)
    return np.array(all_img),all_label

all_img,all_label=get_all_data(img_path)



class InforData(Dataset):
    def __init__(self,all_img,all_label):
        self.all_img=all_img
        self.all_label=all_label

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


if __name__=='__main__':
    img, labels = infordata.__getitem__(10)
    print(img.shape)
    print(labels)












