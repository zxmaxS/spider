import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import csv
from torch.nn.utils.rnn import pad_sequence
import os


def train(device, net, criterion, optimizer):
    # net.load_state_dict(torch.load('torch_ex/models/rnn_net/net_{}.pkl'.format(http_i), map_location=lambda storage, loc: storage))
    var_x = torch.tensor(train_feather, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_label, dtype=torch.float32, device=device)

    batch_var_x = list()
    batch_var_y = list()

    for i in range(batch_size):
        j = 75 - i
        batch_var_x.append(var_x[j:])
        batch_var_y.append(var_y[j:])

    batch_var_x = pad_sequence(batch_var_x)
    batch_var_y = pad_sequence(batch_var_y)

    for e in range(500):
        out = net(batch_var_x)

        loss = criterion(out, batch_var_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            print('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))
    torch.save(net.state_dict(), 'torch_ex/models/rnn_net/net_{}.pkl'.format(http_i))


def evaluate(device, net):
    net.load_state_dict(torch.load('torch_ex/models/rnn_net/net_{}.pkl'.format(http_i), map_location=lambda storage, loc: storage))
    net = net.eval()
    var_x = torch.tensor(test_feather, dtype=torch.float32, device=device)
    var_y = torch.tensor(result, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = net(var_x)
        result_to_show = net(var_y)
    pred = pred.squeeze().cpu()
    result_to_show = result_to_show.squeeze()
    plt.plot(pred, 'r', label='pred')
    plt.plot(test_label, 'b', label='real', alpha=0.3)
    plt.legend(loc='best')
    plt.savefig('torch_ex/images/{}/{}.png'.format(picture, http_i))
    plt.close()
    return result_to_show.item()


class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y


def load_data():
    file = open('torch_ex/data/cba.csv', 'r', encoding='utf-8')
    reader = csv.reader(file)
    seq_number = []
    for item in reader:
        seq_number.append(item[http_i].replace('\ufeff', ''))
    file.close()
    seq_number = np.array(seq_number, dtype=np.float32)

    return seq_number


if __name__ == '__main__':
    batch_size = 30
    result_to_save = []
    picture = 6
    if not os.path.exists('torch_ex/images/{}'.format(picture)):
        os.mkdir('torch_ex/images/{}'.format(picture))
    for http_i in range(14):
        data = load_data()
        feather = data[:-1]
        label = data[1:]
        result = data[-1]

        train_feather = feather[:75]
        train_label = label[:75]
        test_feather = feather[75:]
        test_label = label[75:]
        train_feather = train_feather.reshape((75, 1))
        train_label = train_label.reshape((75, 1))
        test_feather = test_feather.reshape((1, len(feather)-75, 1))
        result = result.reshape((1, 1, 1))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = RegLSTM(1, 1, 8, 2).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
        train(device, net, criterion, optimizer)
        result_to_save.append(evaluate(device, net))

    with open('torch_ex/data/cba_{}.txt'.format(picture), 'w') as file:
        for item in result_to_save:
            file.write(str(item))
            file.write('\n')









            