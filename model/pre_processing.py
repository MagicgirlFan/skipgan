# 此文件主要作用是进行数据预处理，将数据进行整合便于模型加载数据
import os
import cv2
import numpy as np
base_path = './mvtec'
read_file_name = ['ground_truth','test']
file_name = os.listdir(base_path)
img_size = [256,256]
for name in file_name:
    names = os.listdir(base_path+'/'+name+'/test')
    for n in names:
        if n == 'good':
            continue
            img_names = os.listdir(base_path+'/'+name+'/test/'+n)
            for im_n in img_names:
                img = cv2.imread(base_path + '/' + name + '/test/' + n + '/' + im_n)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_bgr = cv2.resize(img_bgr,(256,256))
                black_ = np.zeros(img_bgr.shape)
                save_img = np.concatenate((black_,img_bgr),axis=1)

                cv2.imwrite('./datasets/'+name+n+im_n+'.png',save_img)
        else:
            img_names = os.listdir(base_path + '/' + name + '/test/' + n)
            for im_n in img_names:
                img = cv2.imread(base_path + '/' + name + '/test/' + n + '/' + im_n)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_bgr = cv2.resize(img_bgr,(256,256))
                img_mask = cv2.imread(base_path + '/' + name + '/ground_truth/' + n + '/' + im_n.split('.')[0]+'_mask.png')
                img_bgr_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2BGR)
                img_bgr_mask = cv2.resize(img_bgr_mask,(256,256))
                save_img = np.concatenate((img_bgr_mask, img_bgr), axis=1)

                cv2.imwrite('./datasets/'+name+n+im_n+'.png',save_img)

