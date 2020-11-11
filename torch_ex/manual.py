import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 卷积与池化均分为1d，2d，3d，分别代表1，2，3维的数据，下面就是一个二维卷积
        # 第一个参数是输入通道数，第二个参数是输出通道数，第三个参数是卷积核大小，padding参数是用于填充图像
        # 填充2后卷积得到28*28
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        # BatchNorm函数是一个归一化函数，其作用是将卷积后的数据分布重新变为均值为0的正太分布，由于sigmod等函数在
        # 0附近的梯度较大，从而增加模型收敛速度，传入一个参数，为数据的通道数
        self.bn1 = nn.BatchNorm2d(6)
        # 这里是最大池化，除此之外还有平均池化AvgPool
        # 第一个参数是范围大小，第二个参数是步长，默认为范围大小
        # 池化后变为14*14
        self.pool1 = nn.MaxPool2d(2, 2)
        # 再次卷积变为10*10
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        # 再次池化变为5*5
        self.pool2 = nn.MaxPool2d(2, 2)
        # 再次卷积就成为了一维的变量，下面直接使用全连接层
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        # dropout层用于随机丢弃参数，防止过拟合，后面的参数为丢弃的概率
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # 激活层在卷积和池化之间
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = out.squeeze()
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)

        return out


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    result = []
    for batch_idx, (data, target) in enumerate(train_loader):
        # to函数是tensor进行设备类别转换的函数
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 30 == 0:
            result.append(loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100.0 * batch_idx / len(train_loader), loss.item()))
    return result


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            # item函数是将tensor转变为int或者float的函数
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    model = ConvNet()
    epoch = 75
    batch = 20
    learning_rate = 0.1
    # 损失函数，这里是交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器，这里是选用了随机梯度下降的方法
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)
    train_loader = torch.utils.data.DataLoader(
        # 第二个参数表示是否是训练集，第三个参数表示是否下载
        datasets.MNIST('torch_ex/data', train=True, download=True,
                       # transforms类是用于对传入的图片进行一系列处理的类，需要从torchvision中引入
                       # Compose函数用于将一系列操作进行打包，内部步骤依次执行
                       transform=transforms.Compose([
                           # 该操作是将图片转变为tensor格式
                           transforms.ToTensor(),
                           # 这和网络里的规则化层相同，也是进行归一化用的，传入的是灰度图像，所以只有一个维度
                           # 第一个参数是均值，第二个参数是方差，如果多通道的话需要在元组中继续添加
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('torch_ex/data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=batch, shuffle=True
    )
    # 深度学习与神经网络计算任务之间不相互依赖，gpu可以进行大量的并行计算，而cpu除了计算任务，还包含其他的逻辑任务
    # 所以cpu的计算能力不如gpu
    # device类是一种用来将torch类型数据转变为使用于硬件计算的类，使用字符串来进行硬件分配，其中cuda代表gpu，
    # 冒号后面为所选设备的标号，从0开始计算，需要注意的是device函数不会检查你的硬件是否满足条件，等到后面转换会进行报错
    # 模型和数据的计算设备需要相同，两者默认设备均为cpu，如果想要更换为gpu，需要先判断gpu是否可用
    if torch.cuda.is_available():
        # 将模型中的参数转为gpu计算，可传入参数为gpu序号，默认参数为当前设备
        model.cuda(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    # 如果想使用cpu计算，则传入cpu
    # device = torch.device("cpu")
    # 如果想使用当前设备，则不需进行标号
    # device = torch.device("cuda")
    # 查看当前设备有多少gpu
    # num = torch.cuda.device_count()
    # print(num)
    # 查看当前设备正在使用的gpu标号
    # num = torch.cuda.current_device()
    # print(num)
    result = train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    plt.plot(range(100), result, 'ro-')
    plt.title('Conv')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()














