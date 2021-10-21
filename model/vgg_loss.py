import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.models as py_models
from torchvision import models
from torchsummary import summary
vgg6_net = py_models.vgg16(pretrained=True).features.to('cuda')
def preprocess_image(cv2im, resize_im=True):
    """
        Preprocess the image and modify the length of the image as224*224

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    #print('miao',cv2im.shape,type(cv2im.cpu().detach().numpy()))
    cv2im = cv2im.cpu().detach().numpy().transpose((0,2, 3, 1))
    # Resize image
    img = []
    if resize_im:
        for i in range(cv2im.shape[0]):
            img.append(cv2.resize(cv2im[i], (224, 224)))
    cv2im = np.asarray(img)
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(0,3, 1, 2)  # Convert array to D,W,H
    # Normalize the channels
    '''for channel, _ in enumerate(im_as_arr[1]):
        print(im_as_arr.shape)
        ch = channel + 1
        im_as_arr[ch] /= 255
        im_as_arr[ch] -= mean[ch]
        im_as_arr[ch] /= std[ch]'''

    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float().to('cuda')


    return im_as_ten


class FeatureVisualization():
    def __init__(self,img,selected_layer,model):
        self.img=img
        self.selected_layer=selected_layer
        self.pretrained_model = model

    def process_image(self):

        img=preprocess_image(self.img)

        return img

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input=self.process_image()
        x=input
        for index,layer in enumerate(self.pretrained_model):
            x=layer(x)
            if (index == self.selected_layer):

                return x

    def get_single_feature(self):
        features=self.get_feature()

        feature=features[:,0,:,:]


        feature=feature.view(feature.shape[1],feature.shape[2])


        return feature

    def save_feature_to_img(self):
        #to numpy
        feature=self.get_single_feature()
        feature=feature.data.numpy()

        #use sigmod to [0,1]
        feature= 1.0/(1+np.exp(-1*feature))

        # to [0,255]
        feature=np.round(feature*255)
        print(feature[0])

        cv2.imwrite('./img.jpg',feature)


def get_style_loss(target,prediction):
    real_img = FeatureVisualization(target, 9, vgg6_net) #读取卷积conv2_2
    fake_img = FeatureVisualization(prediction, 9, vgg6_net)

    real_feature = real_img.get_feature()
    fake_feature = fake_img.get_feature()
    feature_count = real_feature.shape[1]*real_feature.shape[0]
    style_loss = torch.sum(torch.square(real_feature - fake_feature))

    return style_loss / float(feature_count)


