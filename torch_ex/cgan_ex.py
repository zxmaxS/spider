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

        # Embedding函数用于将词转变为词向量，第一个参数是输入词的个数，第二个参数是生成的词向量的维度
        self.label_emb = nn.Embedding(n_classes, n_classes)

        # 与gan中的block函数相同
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # 这里将生成的词向量与随机噪声合并后传入网络
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # 将传入的词转变为词向量后与传入的噪声合并，cat函数用于在某一维上合并tensor，默认为0维，传入-1代表倒数第一维
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        # 这里相当于四个全连接层
        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            # dropout层用于随机丢弃参数，防止过拟合，后面的参数为丢弃的概率
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # 将传入的词转变为词向量后与传入的噪声合并
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


def sample_image(n_row, batches_done):
    # 获取100个随机噪声
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row * n_row, latent_dim))))
    # 获得有序的从0到9的标签
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "torch_ex/images/%d.png" % batches_done, nrow=n_row, normalize=True)


if __name__ == '__main__':
    # 训练次数以及分块大小
    n_epochs = 50
    batch_size = 100
    # 优化器参数
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    # 设定的噪声维度
    latent_dim = 100
    # 输入词的长度以及生成变量的维度
    n_classes = 10
    # 图片大小及通道数
    channels = 1
    img_size = 32
    # 展示生成的图片
    sample_interval = 200

    img_shape = (channels, img_size, img_size)

    # 损失函数
    adversarial_loss = torch.nn.MSELoss()

    # 初始化生成网络与判别网络，这里为了能够储存结果，只有初始化的时候调用了定义的网络
    # generator = Generator()
    # discriminator = Discriminator()
    generator = torch.load('torch_ex/models/cgan_gen.pkl')
    discriminator = torch.load('torch_ex/models/cgan_dis.pkl')

    # 判断是否可以使用gpu
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
    else:
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor

    # 获取训练集数据
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "torch_ex/data",
            train=True,
            download=True,
            transform=transforms.Compose([
                # mnist数据集的图片大小为28*28，这里需要进行填充
                transforms.Resize(img_size),
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
        for i, (imgs, labels) in enumerate(dataloader):
            # 生成真与假的标签，之后用于计算loss
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # 训练生成器
            # 清空梯度
            optimizer_G.zero_grad()
            # 随机生成噪声以及标签
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
            gen_imgs = generator(z, gen_labels)
            # 计算判别器的loss值
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            # 生成器反向传播
            g_loss.backward()
            optimizer_G.step()

            # 训练判别器
            # 清空梯度
            optimizer_D.zero_grad()
            # 计算真与假图片的loss值，并取均值
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            # 判别器反向传播
            d_loss.backward()
            optimizer_D.step()

            # 每200个batch后生成一张生成器生成的图片
            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)
    # 保存模型
    torch.save(generator, 'torch_ex/models/cgan_gen.pkl')
    torch.save(discriminator, 'torch_ex/models/cgan_dis.pkl')
