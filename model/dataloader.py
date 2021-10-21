from PIL import Image
import glob
import torch
import torch
import numpy as np
import os
from torch.utils.data import DataLoader


class load_ImageData():
    '''
    Read pairs of data, mainly used to read analog noise images
    '''

    def __init__(self, root):
        self.files = sorted(glob.glob(root))

    def devide_real_fake_image(self, image):
        '''
        Decompose the input image from the real image
        :param image: input image
        :return: real image and fake image
        '''
        image = np.asarray(image)

        w = image.shape[0]

        real_img = image[:, 0:w, :]
        fake_img = image[:, w:, :]

        return real_img, fake_img

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        img = np.asarray(img, dtype=np.float32)
        real_img, fake_img = self.devide_real_fake_image(img)

        fake_img = (fake_img / 255.0) # [0,256]->[0,1]
        real_img = (real_img / 255.0)

        fake_img = fake_img.transpose((2, 0, 1)) # w,h,c -> c.w,h
        real_img = real_img.transpose((2, 0, 1))
        fake_img = torch.from_numpy(fake_img)
        real_img = torch.from_numpy(real_img)

        return {'fake_img': fake_img, 'real_img': real_img}

    def __len__(self):
        return len(self.files)


def get_train_data(data_root, batch_size):
    train_dataloader = DataLoader(load_ImageData(root=data_root),
                                  batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return train_dataloader

if __name__ == "__main__":
    print(glob.glob('datasets/*.*'))