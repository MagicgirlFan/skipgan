from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
import cv2

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def display_images1(generator, root, picture):
    image = Image.open(root + picture)
    image = np.asarray(image, dtype=np.float)
    w = image.shape[0]
    fake_img = image[:, 0:w, :]
    real_img = image[:, w:, :]

    fake_img_train = (fake_img / 255.0)
    fake_img_train = fake_img_train.transpose((2, 0, 1))
    fake_img_train = np.expand_dims(fake_img_train, 0)
    fake_img_train = torch.from_numpy(fake_img_train)
    fake_img_train = Variable(fake_img_train.type(torch.FloatTensor).cuda())
    gen = generator(fake_img_train)
    gen = gen.cpu().detach().numpy()
    gen = np.array(gen[0])
    gen = (gen ) * 255.0
    gen = gen.transpose((1, 2, 0))
    plt.figure(figsize=(15, 15))
    display_list = [fake_img, real_img, gen]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] / 255.0)
        plt.axis('off')
    plt.show()

def display_images(input_img,target_img,genn,model = False):

    if model == True:
        genn_img = genn(input_img)
    else:
        genn_img = genn
    input_img = ((input_img[0])*255.0).cpu().detach().squeeze().numpy().transpose((1, 2, 0))
    target_img = ((target_img[0])*255.0).cpu().detach().squeeze().numpy().transpose((1, 2, 0))

    genn_img = ((genn_img[0])*255.0).cpu().detach().numpy().squeeze().transpose((1, 2, 0))

    display_list = [input_img, target_img, genn_img]

    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] / 255.0)
        plt.axis('off')
    plt.show()


