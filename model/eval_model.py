import torch
from model import Net
from dataloader import get_train_data
import time
import torchvision.models as py_models
from torch.autograd import Variable
from Image_display import display_images

vgg6_net = py_models.vgg16(pretrained=True).features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets = get_train_data('datasets/*.*',batch_size=8)

net = Net(3).to(device)
net = torch.load('G.pth').cuda()
epoch = 200
def eval_model(dataset,net):

    t = time.time()
    net.eval()
    for iter,batch in enumerate(dataset):
        fake_img = Variable(batch['fake_img'].type(torch.FloatTensor).cuda())
        real_img = Variable(batch['real_img'].type(torch.FloatTensor).cuda())
        G_img = net(fake_img)


        if iter % 5 == 0:
            display_images(fake_img, real_img, G_img)


eval_model(datasets,net)