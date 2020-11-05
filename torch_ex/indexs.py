import numpy as np
import torch
from torch import nn


class Perceptron(nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super(Perceptron, self).__init__()
        # 使用模块内置的全连接层
        self.net = nn.Linear(in_dim, out_dim)
        # 参数初始化
        for params in self.net.parameters():
            # 使用正态分布进行参数初始化，均值为0，方差为0.01
            nn.init.normal_(params, mean=0, std=0.01)

    # 输入数据在模型中前向传播的计算过程
    def forward(self, x):
        x = self.net(x)
        return x


def load_data():
    data = np.loadtxt('data/wdbc_binary/wdbc.txt', dtype=str, skiprows=0, delimiter=',')

    feature = [data[i][2:] for i in range(len(data))]
    label = [0 if data[i][1] == 'M' else 1 for i in range(len(data))]
    # 将数据集分为训练集与测试集，这里是3:1
    train_feature = torch.tensor(np.array([feature[i] for i in range(len(feature)) if i % 3 != 0]).astype(float),
                                 dtype=torch.float32)
    test_feature = torch.tensor(np.array([feature[i] for i in range(len(feature)) if i % 3 == 0]).astype(float),
                                dtype=torch.float32)

    train_label = torch.tensor(np.array([label[i] for i in range(len(label)) if i % 3 != 0]),
                               dtype=torch.int64)
    test_label = torch.tensor(np.array([label[i] for i in range(len(label)) if i % 3 == 0]),
                              dtype=torch.int64)

    # train_label = train_label.type(torch.LongTensor)
    # test_label = test_label.type(torch.LongTensor)

    # 将训练集与测试集的特征与标签合并
    train_data = torch.utils.data.TensorDataset(train_feature, train_label)
    test_data = torch.utils.data.TensorDataset(test_feature, test_label)
    return train_data, test_data


# 训练权重
def train(train_loader, model, criterion, optimizer):
    # 启用batchNormalization和dropout
    model.train()
    for i, (feature, label) in enumerate(train_loader):
        # 获得训练结果
        output = model(feature)
        # 计算损失
        loss = criterion(output, label)
        # 清零梯度
        optimizer.zero_grad()
        # 计算反向传播梯度
        loss.backward()
        # 执行优化
        optimizer.step()


# 验证结果
def test(test_loader, model):
    # 禁用batchNormalization和dropout
    model.eval()
    correct = 0
    # 在测试时不需要计算梯度，从而减少计算代价
    with torch.no_grad():
        for i, (feature, label) in enumerate(test_loader):
            output = model(feature)
            # 获取计算最大值的下标
            _, pred = output.topk(1, 1, True, True)
            # view函数改变label变量的形状，使其与pred形状相同，然后使用sum函数进行求和，算出正确的个数
            correct += pred.eq(label.view(-1, 1)).sum(0, keepdim=True)
    return correct[0] * 100.0/len(test_data)


if __name__ == '__main__':
    # 读取数据
    train_data, test_data = load_data()
    model = Perceptron(in_dim=30, out_dim=2)
    epoch = 75
    batch_size = 20
    learning_rate = 0.1
    # 损失函数，这里是交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器，这里是选用了随机梯度下降的方法
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 将训练数据分成一个个batch
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    result = []
    for i in range(epoch):
        train(train_loader, model, criterion, optimizer)
        correct = test(test_loader, model)
        print("epoch: {} acc: {}".format(i, correct))
        result.append(correct)
    # 输出最大准确率
    print(max(result))














