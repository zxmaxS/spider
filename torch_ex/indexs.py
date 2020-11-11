import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


# 自定义的类本身也可以作为其他类的一部分，虽然这里只有一层全连接层，如果存在两到三层，就可以减化深度网络的编写，
# 还可以对神经网络进行分块处理
class Perceptron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Perceptron, self).__init__()
        # 使用模块内置的全连接层
        self.net = nn.Linear(in_dim, out_dim)
        # 参数初始化
        # parameters()函数会返回该类中的所有参数，如果你只想对某一层的参数进行初始化，可以使用self.net.parameters()
        # 除此之外，还可以用self.named_parameters()同时返回name和params
        # for name, params in self.named_parameters():
        for params in self.parameters():
            # print(params)
            # 使用正态分布进行参数初始化，均值为0，方差为0.01
            nn.init.normal_(params, mean=0, std=0.01)
            # print(params)

    # 输入数据在模型中前向传播的计算过程
    def forward(self, x):
        x = self.net(x)
        return x


def load_data():
    # np读取txt，后面的参数应该能看懂，读入后是一个np_array
    data = np.loadtxt('torch_ex/data/wdbc_binary/wdbc.txt', dtype=str, skiprows=0, delimiter=',')

    # 这是列表生成式，详情可以看list_ex
    feature = [data[i][2:] for i in range(len(data))]
    label = [0 if data[i][1] == 'M' else 1 for i in range(len(data))]
    # 将数据集分为训练集与测试集，这里是3:1
    # 这里的torch.tensor与np.array类似，是将一个矩阵转变为一个tensor对象，该对象与np_array都具有dtype属性，
    # 可以查看其自身类型，详细类型可以自己去查，astype函数是np_array用来进行变量类型转换的函数
    # tensor的创建总是复制数据
    train_feature = torch.tensor(np.array([feature[i] for i in range(len(feature)) if i % 3 != 0]).astype(float),
                                 dtype=torch.float32)
    test_feature = torch.tensor(np.array([feature[i] for i in range(len(feature)) if i % 3 == 0]).astype(float),
                                dtype=torch.float32)

    # 除此之外，还可以直接由np_array生成torch，生成的torch与np_array共享内存，即两者是别名的关系
    # dataset = torch.from_numpy(feature)
    train_label = torch.tensor(np.array([label[i] for i in range(len(label)) if i % 3 != 0]),
                               dtype=torch.int64)
    test_label = torch.tensor(np.array([label[i] for i in range(len(label)) if i % 3 == 0]),
                              dtype=torch.int64)

    # 将训练集与测试集的特征与标签合并，生成的为TensorDataset类，用于后面进行切分，实际上就是将特征与标签放到一个元组中
    train_data = torch.utils.data.TensorDataset(train_feature, train_label)
    test_data = torch.utils.data.TensorDataset(test_feature, test_label)
    return train_data, test_data


# 训练权重
def train(train_loader, model, criterion, optimizer):
    # 启用batchNormalization和dropout
    model.train()
    # enumerate函数用途是将一个可迭代对象的下标作为第一个变量输出出来
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
            # fork函数用于获取最大值及其下标，第一个参数为前k个最大值，这里只需要最大的值，第二个参数为所需的维度，从
            # 0开始算，如果觉得维度蒙的话建议好好想一想，第三个参数为True返回最大值，否则为最小值，第四个参数为True
            # 返回的结果会进行排序，否则不会，返回的_代表最大值的值，pred代表最大值的下标
            _, pred = output.topk(1, 1, True, True)
            # view函数改变label变量的形状，使其与pred形状相同，然后使用sum函数进行求和，算出正确的个数
            # eq函数用于比较两个tensor是否完全相同，如果是会返回1
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
    # 将训练数据分成一个个batch，需要TensorDataset类作为输入数据，返回一个迭代器，后面的shuffle参数代表
    # 每次取后是否打乱顺序，这个函数会将所有的数据分为batch_size大小，你每次调用train_loader，他都会重新进行
    # 分块
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
    plt.plot(range(epoch), result, 'ro-')
    plt.title('Perceptron')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()














