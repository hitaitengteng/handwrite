import numpy as np
import os
import cv2
from PIL import Image
import  tensorflow as tf



def lable_dict(cls_nnumber):
    letter=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    all_label=[]
    for i in range(1,len(letter)+1):
        all_label.append(i)

    dic_info = dict(zip(letter, all_label))

    return dic_info
dict=lable_dict(26)

def get_numlabel(str_label,dict,max_len):
    number_label=[]
    for i in list(str_label):
        number_label.append(dict.get(i))
    if len(number_label)<max_len:
        for j in range(max_len-len(list(str_label))):
            number_label.append(0)
    return number_label
# a=get_numlabel('abcefz',dict,20)





test_filename = './test.tfrecords'
test_path='/Users/wywy/Desktop/英文识别测试/test_img'

def saver_lables(img_path,train_filename,dict):
    writer = tf.python_io.TFRecordWriter(train_filename)


    for name in os.listdir(img_path):
        if name=='.DS_Store':
            os.remove(img_path+'/'+name)
        else:

            label=name.split('.')[0].split('_')[-1]
            number_label=get_numlabel(label,dict,20)

            img=Image.open(img_path+'/'+name)
            if img.size==(280,80):
                pass
            else:
                print('xx')
            image = img.tobytes()
            # for i in list(label):
            #     num = dict[i]
            #     ones_label.append(num)
            example = tf.train.Example(features=tf.train.Features(feature={
                # 'class_lables': tf.train.Feature(float_list=tf.train.FloatList(value=[class_label])),
                'lables': tf.train.Feature(float_list=tf.train.FloatList(value=number_label)),
                'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
            }))

            writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_data_for_file(file, capacity,image_size):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([file], num_epochs=None, shuffle=False, capacity=capacity)

    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)

    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            # 'class_lables': tf.FixedLenFeature([], tf.float32),
            'lables': tf.FixedLenFeature([20], tf.float32),
            'images': tf.FixedLenFeature([], tf.string)

        }
    )
    img = tf.decode_raw(features['images'], tf.uint8)
    img = tf.reshape(img, image_size)
    img = tf.cast(img, tf.float32)
    # class_lable = tf.cast(features['class_lables'], tf.float32)
    lables = features['lables']


    return img, lables


def test_shuffle_batch(train_file_path,image_size, batch_size, capacity=1000, num_threads=3):
    images, lables = read_data_for_file(train_file_path, 1000,image_size)

    images_,  lables_ = tf.train.shuffle_batch([images,lables], batch_size=batch_size, capacity=capacity,
                                               min_after_dequeue=100,
                                               num_threads=num_threads)
    return images_, lables_

if __name__=='__main__':


    init = tf.global_variables_initializer()
    saver_lables(test_path,test_filename,dict)
    read_data_for_file(test_filename,100,[80,280,1])
    a=test_shuffle_batch(test_filename,[80,280,1],100)


    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)

        sess.run(init)


        aa,bb=sess.run(a)
        print(aa,
              bb[0])



