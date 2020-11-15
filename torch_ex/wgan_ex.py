import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch import nn
import torch


# 模型上与gan无区别
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


if __name__ == '__main__':
    # 训练次数以及分块大小
    n_epochs = 200
    batch_size = 100
    # 学习率
    lr = 0.00005
    # 设定的噪声维度
    latent_dim = 100
    # 图片大小及通道数
    channels = 1
    img_size = 28
    # 生成器训练次数限制
    n_critic = 5
    # 参数的范围限制
    clip_value = 0.01
    sample_interval = 200

    img_shape = (channels, img_size, img_size)

    # 初始化生成网络与判别网络，这里为了能够储存结果，只有初始化的时候调用了定义的网络
    # generator = Generator()
    # discriminator = Discriminator()
    generator = torch.load('torch_ex/models/wgan_gen.pkl')
    discriminator = torch.load('torch_ex/models/wgan_dis.pkl')

    # 判断是否可以使用gpu
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    # 获取训练集数据
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "torch_ex/data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    # 优化器，这里选择了RMSprop算法
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    batches_done = 0
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = Variable(imgs.type(Tensor))
            # 训练判别器
            # 清空梯度
            optimizer_D.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
            fake_imgs = generator(z).detach()
            # 计算loss值
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
            # 判别器反向传播
            loss_D.backward()
            optimizer_D.step()

            # 限制判别器的参数范围
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # 调节生成器的训练次数
            if i % n_critic == 0:
                # 训练判别器
                # 清空梯度
                optimizer_G.zero_grad()
                gen_imgs = generator(z)
                # 计算loss
                loss_G = -torch.mean(discriminator(gen_imgs))
                # 生成器反向传播
                loss_G.backward()
                optimizer_G.step()
            # 每200次训练生成器后生成一张生成器生成的图片
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:100], "torch_ex/images/%d.png" % batches_done, nrow=10, normalize=True)
            batches_done += 1
    # 储存模型
    torch.save(generator, 'torch_ex/models/wgan_gen.pkl')
    torch.save(discriminator, 'torch_ex/models/wgan_dis.pkl')
