import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
import csv


class rnn_classify(nn.Module):
    def __init__(self, in_feature=1, hidden_feature=50, num_class=1, num_layers=2):
        super(rnn_classify, self).__init__()
        self.rnn = nn.LSTM(input_size=in_feature, hidden_size=hidden_feature, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        # 先要将 维度为 (batch, 1, 28, 28)的x转换为 (28, batch, 28)
        x = x.squeeze()        # (batch, 1, 28, 28)——(batch, 28, 28)
        x = x.permute(2, 0, 1)     # 将最后一维放到第一维，变成(28, batch, 28)
        out, _ = self.rnn(x)     # 使用默认的隐藏状态，即全0，得到的out是 (28, batch, hidden_feature)
        out = out[-1, :, :]
        out = self.classifier(out)
        return out



def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().data
    # print(num_correct, total)
    return num_correct

def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    for i in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())
                label = Variable(label.cuda())
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            total = output.shape[0]
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data.cpu().numpy()/float(total)
            train_acc += get_acc(output, label).cpu().numpy()/float(total)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = Variable(im.cuda(), volatile=True)
                    label = Variable(label.cuda(), volatile=True)
                else:
                    im = Variable(im, volatile=True)
                    label = Variable(label, volatile=True)
                output = net(im)
                total = output.shape[0]
                loss = criterion(output, label)
                valid_loss += loss.data.cpu().numpy()/float(total)
                valid_acc += get_acc(output, label).cpu().numpy()/float(total)
            print("epoch: %d, train_loss: %f, train_acc: %f, valid_loss: %f, valid_acc:%f"
                  % (i, train_loss/len(train_data),  train_acc/len(train_data),
                  valid_loss/len(valid_data),  valid_acc/len(valid_data)))

        else:
            print("epoch= ", i, "train_loss= ", train_loss/len(train_data), "train_acc= ", train_acc/len(train_data))


def data_load():
    file = open('torch_ex/data/cba.csv', 'r', encoding='utf-8')
    reader = csv.reader(file)
    train = []
    test = []
    num = 0
    for item in reader:
        if num < 50:
            train.append(item)
        else:
            test.append(item)
        num += 1
    file.close()

    train = torch.tensor(train).t()
    test = torch.tensor(test).t()


if __name__ == '__main__':
    data_load()
    # train_data = DataLoader(train_set, batch_size=64, shuffle=True)
    # test_data = DataLoader(test_set, batch_size=64, shuffle=False)
    # net = rnn_classify()
    # criterion = nn.CrossEntropyLoss()
    # optimzier = torch.optim.Adadelta(net.parameters(), 1e-1)
    # # 开始训练
    # train(net, train_data, test_data, 10, optimzier, criterion)