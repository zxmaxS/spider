import numpy as np
import torch
from torch import nn


class Perceptron(nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super(Perceptron, self).__init__()
        # 使用模块内置的全连接层
        self.layer1 = nn.Linear(in_dim, 40)
        # inplace函数代表是否进行内存覆盖
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(40, out_dim)
        # 参数初始化
        for params in self.parameters():
            # 使用正态分布进行参数初始化，均值为0，方差为0.01
            nn.init.normal_(params, mean=0, std=0.01)

    # 输入数据在模型中前向传播的计算过程
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def load_data():
    data = np.loadtxt('torch_ex/data/soybean_multi/soybean-large.txt', dtype=str, skiprows=0, delimiter=',')
    # 因为数据集中包含缺失，将缺失改为-1
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] == '?':
                data[i][j] = '-1'

    feature = [data[i][1:] for i in range(len(data))]
    # 使用字典存储特征类别信息
    label = []
    num = 0
    label_dict = {}
    for i in range(len(data)):
        if data[i][0] in label_dict:
            label.append(label_dict[data[i][0]])
        else:
            label_dict[data[i][0]] = num
            label.append(num)
            num = num + 1
    train_feature = torch.tensor(np.array(feature[0:307]).astype(float), dtype=torch.float32)
    test_feature = torch.tensor(np.array(feature[307:]).astype(float), dtype=torch.float32)

    train_label = torch.tensor(np.array(label[0:307]), dtype=torch.int64)
    test_label = torch.tensor(np.array(label[307:]), dtype=torch.int64)

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
    model = Perceptron(in_dim=35, out_dim=19)
    epoch = 100
    batch_size = 80
    learning_rate = 0.8
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














