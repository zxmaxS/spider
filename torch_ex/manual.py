import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)

        return out


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # to函数是tensor进行设备类别转换的函数
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100.0 * batch_idx / len(train_loader), loss.item()))


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

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=batch, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False,
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
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)















