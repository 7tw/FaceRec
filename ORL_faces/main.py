from PIL import Image
import sys
import os
import os.path
import numpy as np
import PIL.Image
from pylab import *

def change_photo_size28(path,spath):
    '''
    将人脸图片转化为28*28的灰度图片
    '''

    filenames = os.listdir(path)

    for filename in filenames:
        f = os.path.join(path,filename)
        iimag = PIL.Image.open(f).convert('L').resize((28,28))
        savepath = os.path.join(spath,filename)
        #savepath = spath + '/' + filename
        iimag.save(savepath)


def read_photo_for_train(k,photo_path):
    '''
    读取训练图片
    '''
    for i in range(k):
        j = i
        j = str(j)
        st = '.bmp'
        j = j+st
        j = os.path.join(photo_path,j)
        im1 = array(Image.open(j).convert('L'))
        #（28，28）-->(28*28,1)
        im1 = im1.reshape((784,1))
        #把所有的图片灰度值放到一个矩阵中
        #一列代表一张图片的信息
        if i == 0:
            im = im1
        else:
            im = np.hstack((im,im1))
    return im


def layerout(w,b,x):

    '''
    sigmoid函数实现
    '''

    y = np.dot(w,x) + b
    t = -1.0*y
    # n = len(y)
    # for i in range(n):
        # y[i]=1.0/(1+exp(-y[i]))
    y = 1.0/(1+exp(t))
    return y


def mytrain(x_train,y_train):
    '''
    训练样本：ORL_faces.女性标签为0，男性标签为1.
    训练方法：简单的梯度下降法
    '''

    '''
    设置一个隐藏层，784-->隐藏层神经元个数-->1
    '''

    step=int(input('mytrain迭代步数：'))
    a=double(input('学习因子：'))
    inn = 784  #输入神经元个数
    hid = int(input('隐藏层神经元个数：'))#隐藏层神经元个数
    out = 1  #输出层神经元个数

    w = np.random.randn(out,hid)
    w = np.mat(w)
    b = np.mat(np.random.randn(out,1))
    w_h = np.random.randn(hid,inn)
    w_h = np.mat(w_h)
    b_h = np.mat(np.random.randn(hid,1))

    for i in range(step):
        #打乱训练样本
        r=np.random.permutation(60)
        x_train = x_train[:,r]
        y_train = y_train[:,r]
        #mini_batch
        for j in range(50):
            x = np.mat(x_train[:,j])
            x = x.reshape((784,1))
            y = np.mat(y_train[:,j])
            y = y.reshape((1,1))
            hid_put = layerout(w_h,b_h,x)
            out_put = layerout(w,b,hid_put)

            #更新公式的实现
            o_update = np.multiply(np.multiply((y-out_put),out_put),(1-out_put))
            h_update = np.multiply(np.multiply(np.dot((w.T),np.mat(o_update)),hid_put),(1-hid_put))

            outw_update = a*np.dot(o_update,(hid_put.T))
            outb_update = a*o_update
            hidw_update = a*np.dot(h_update,(x.T))
            hidb_update = a*h_update

            w = w + outw_update
            b = b+ outb_update
            w_h = w_h +hidw_update
            b_h =b_h +hidb_update

    return w,b,w_h,b_h


def mytest(x_test,w,b,w_h,b_h):
    '''
    预测结果pre大于0.5，为男；预测结果小于或等于0.5为女
    '''
    hid = layerout(w_h,b_h,x_test);
    pre = layerout(w,b,hid);
    print(pre)
    if pre > 0.5:
        print("这是男性")
    else:
        print("这是女性")


#将人脸图片转化为28*28的灰度图片
path = 'faces'
spath = 'faces'
change_photo_size28(path,spath)


#获取图片信息
im = read_photo_for_train(60,spath)

#归一化
immin = im.min()
immax = im.max()
im = (im-immin)/(immax-immin)

x_train = im

#制作标签，前30张是女性，为0。后30张是男性，为1
y1 = np.zeros((1,30))
y2 = np.ones((1,30))
y_train = np.hstack((y1,y2))

#开始训练
print("----------------------开始训练-----------------------------------------")
w,b,w_h,b_h = mytrain(x_train,y_train)
print("-----------------------训练结束------------------------------------------")


#测试
print("--------------------测试女生-----------------------------------------")

#将人脸图片转化为28*28的灰度图片
path = 'girltests'
spath = 'girltests'
change_photo_size28(path,spath)


#获取图片信息
im = read_photo_for_train(10,spath)

#归一化
immin = im.min()
immax = im.max()
im = (im-immin)/(immax-immin)

x_test = im
#print(x_test.shape)
for i in range(10):
    xx = x_test[:,i]
    xx = xx.reshape((784,1))
    mytest(xx,w,b,w_h,b_h)
print("---------------------测试男生-----------------------------")

#将人脸图片转化为28*28的灰度图片
path = 'boytests'
spath = 'boytests'
change_photo_size28(path,spath)


#获取图片信息
im = read_photo_for_train(10,spath)

#归一化
immin = im.min()
immax = im.max()
im = (im-immin)/(immax-immin)

x_test = im
for i in range(10):
    xx = x_test[:,i]
    xx = xx.reshape((784,1))
    mytest(xx,w,b,w_h,b_h)
