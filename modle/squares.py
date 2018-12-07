import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = cv.GaussianBlur(img, (5, 5), 0)    #对图片进行高斯模糊
    # plt.imshow(img)
    # plt.show()
    squares = []
    center=[]
    # center1=set(center)
    for gray in cv.split(img):   #把图片转换成三个通道。因为cv.Canny（）边缘检测的时候必须要单通道灰度图
        for thrs in xrange(0, 255, 2):   #
            if thrs == 0:     #如果说像素点等于0的话
                bin = cv.Canny(gray, 0, 100, apertureSize=5)    #边缘检测   返回一张二值图
                # plt.imshow(img)
                # plt.show()

                bin = cv.dilate(bin, None)    #边缘膨胀，使得边缘检测线条更明显。

            else:
                # _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)   #简单滤波   原始图像gray必须是灰度图（返回返回值说明和处理过后的图片）
                bin = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 7)
                # plt.imshow(im_at_mean)
                plt.show()
            bin, contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)   #接受的参数是黑白图片不是灰度图。
            # print(contours[0].shape)
            # plt.imshow(bin)
            # plt.show()
            # print(_hierarchy)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)   #计算轮廓的长度
                cnt = cv.approxPolyDP(cnt, 0.0002*cnt_len, True)
                # print(cnt.shape)

                if len(cnt) == 4 and cv.contourArea(cnt) > 5  and cv.contourArea(cnt) < 20000 and cv.isContourConvex(cnt):   #4个点，图形面积大于1000，曲线是否是凸性的
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
                        x, y, w, h = cv.boundingRect(cnt)
                        center_=(int(x+w/2),int(y+h/2))
                        center.append(center_)

    return squares,set(center)
# img=cv.imread('/Users/wywy/Desktop/pic1.png')
# print(img.shape)
if __name__ == '__main__':
    from glob import glob
    for fn in glob('/Users/wywy/Desktop/test1.jpg'):

        img = cv.imread(fn)
        squares,center = find_squares(img)
        cv.drawContours( img, squares, -1, (0, 255, 255), 3 )
        cv.imshow('squares', img)
        ch = cv.waitKey()
        if ch == 27:
            break