import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch import nn
import torch


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # block函数包含三部分，全连接层，可选择的归一化层，还有relu激活函数层
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 使用sequent打包四个block后，使用全连接层生成图像，并调用激活函数
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            # prod函数是使一个可迭代对象中的数据进行连乘
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        # 修改生成的数据使其变为二维
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 使用sequent，相当于三层全连接层
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


if __name__ == '__main__':
    # 训练次数以及分块大小
    n_epochs = 200
    batch_size = 100
    # 优化器参数
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    # 设定的噪声维度
    latent_dim = 100
    # 图片大小及通道数
    img_size = 28
    channels = 1
    # 展示生成的图片
    sample_interval = 200

    img_shape = (channels, img_size, img_size)

    # 损失函数
    adversarial_loss = torch.nn.BCELoss()

    # 初始化生成网络与判别网络，这里为了能够储存结果，只有初始化的时候调用了定义的网络
    # generator = Generator()
    # discriminator = Discriminator()
    generator = torch.load('torch_ex/models/gan_gen.pkl')
    discriminator = torch.load('torch_ex/models/gan_dis.pkl')

    # 判断是否可以使用gpu
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
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

    # 优化器，这里选择了adam算法
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # imgs.size = (100, 1, 28, 28)
            # 生成真与假的标签，之后用于计算loss
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
            real_imgs = Variable(imgs.type(Tensor))

            # 训练生成器
            # 清空梯度
            optimizer_G.zero_grad()
            # np.random.normal用于生成正则化随机数，第一个参数是均值，第二个参数是方差，第三个参数是形状
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
            gen_imgs = generator(z)
            # 计算判别器的loss值
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            # 生成器反向传播
            g_loss.backward()
            optimizer_G.step()

            # 训练判别器
            # 清空梯度
            optimizer_D.zero_grad()
            # 计算判别器区分真假图片的loss值并取均值
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            # 判别器反向传播
            d_loss.backward()
            optimizer_D.step()
            # 每200个batch后生成一张生成器生成的图片
            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:100], "torch_ex/images/%d.png" % batches_done, nrow=10, normalize=True)
    # 保存模型
    torch.save(generator, 'torch_ex/models/gan_gen.pkl')
    torch.save(discriminator, 'torch_ex/models/gan_dis.pkl')



