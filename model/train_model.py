import torch
from model import Net,Dis
from dataloader import get_train_data
import time
import torchvision.models as py_models
from torch.autograd import Variable
from vgg_loss import get_style_loss
from Image_display import display_images
import torch.nn as nn
vgg6_net = py_models.vgg16(pretrained=True).features # loss
# use equipment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Retrieving Result Set Metadata
datasets = get_train_data('datasets/*.*',batch_size=8)
# loading model
net = Net(3).to(device)

# optimization function
optimizer = torch.optim.Adam(net.parameters(),lr=0.0002,betas=(0.5,0.999))
# loss func
criterion = torch.nn.MSELoss(reduction='mean').to(device)
criterionCAE = torch.nn.L1Loss(reduction='mean').to(device)
# Variation of training parameters
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.8)

epoch = 200
l_adv = nn.BCELoss()
def train(dataset,net,net_D,criterion,epoch):
    step_n = 0

    for i in range(epoch):
        loss_mse_mean = 0.0
        loss_cae_mean = 0.0
        loss_vgg_mean = 0.0

        # train G
        for iter,batch in enumerate(dataset):
            fake_img = Variable(batch['fake_img'].type(torch.FloatTensor).cuda())
            real_img = Variable(batch['real_img'].type(torch.FloatTensor).cuda())
            real_label = torch.ones(size=(real_img.shape[0],1), dtype=torch.float32, device='cuda')
            fake_label = torch.zeros(size=(fake_img.shape[0],1), dtype=torch.float32, device='cuda')
            optimizer.zero_grad()
            G_img = net(fake_img)
            f_D = net_D(fake_img)
            loss_mse = criterion(real_img,G_img) # mean((real_img - G_img)**2)
            loss_cae = criterionCAE(real_img,G_img)
            loss_vgg = get_style_loss(real_img,G_img)
            loss_adv = l_adv(f_D,fake_label)
            loss = loss_mse + loss_cae + loss_vgg + loss_adv
            loss_mse_mean += loss_mse
            loss_cae_mean += loss_cae
            loss_vgg_mean += loss_vgg

            loss.backward()
            optimizer.step()


            if iter == 0:
                if step_n % 5 == 0:
                    display_images(fake_img, real_img, G_img)
                step_n += 1

            fake_img = Variable(batch['fake_img'].type(torch.FloatTensor).cuda())
            real_img = Variable(batch['real_img'].type(torch.FloatTensor).cuda())
            optimizer_D.zero_grad()
            G_img = net(fake_img)
            f_D = net_D(fake_img)
            r_D = net_D(real_img)


            loss_adv_D = l_adv(r_D, real_label)
            loss_adv_f = l_adv(f_D, fake_label)
            loss_D =  loss_adv_D + loss_adv_f

            loss_D.backward()
            optimizer_D.step()

            if iter == 0:
                if step_n % 5 == 0:
                    display_images(fake_img, real_img, G_img)
                step_n += 1

        # load data
        scheduler.step()
        scheduler_D.step()
    torch.save(net, 'G1.pth')
    torch.save(net_D, 'D.pth')
train(datasets, net,net_D,criterion,epoch)