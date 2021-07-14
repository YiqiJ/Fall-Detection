import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch import optim
import torch.nn.functional as F

from data_utils import read_data 

class Net(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_hidden_layer, output_size, bptt):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layer = num_hidden_layer
        self.output_size = output_size
        self.bptt = bptt

        self.lstm = nn.LSTM(input_size, hidden_size, num_hidden_layer)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()


    def forward(self, input, hidden):
        input = input.view(-1, self.batch_size, self.input_size)
        output, hidden = self.lstm(input, hidden)
        output = output.view(-1, self.hidden_size)
        output = self.softmax(self.linear(output))

        return output, hidden


    def init_hidden(self, cuda):
        h =  Variable(torch.zeros(self.num_hidden_layer, self.batch_size, self.hidden_size))
        c =  Variable(torch.zeros(self.num_hidden_layer, self.batch_size, self.hidden_size))
        if torch.cuda.is_available and cuda: 
            h, c = h.cuda(), c.cuda()
        return (h, c)


    def read_sequence(self, path, num_components):
        datas = read_data(path, num_components) 
        sequences = []
        for x_, y_ in datas:
            xs, ys = [], []
            for i in range(x_.shape[0] // self.bptt):
                x = x_[i*self.bptt:(i+1)*self.bptt,:]
                y = y_[i*self.bptt:(i+1)*self.bptt]
                x = torch.from_numpy(x).float()
                y = torch.from_numpy(y).long()
                xs.append(x)
                ys.append(y)

            sequences.append((xs, ys))
        return sequences

    def split_train_valid(self, sequences):
        train_seq, valid_seq = [], []
        for (idx, seq) in enumerate(sequences):
            if (idx + 1) % 3 == 0:
                valid_seq.append(seq)
            else:
                train_seq.append(seq)
        return train_seq, valid_seq


def train(sequences, net, error, optimizer, cuda, prob, is_training = True):
    losses = 0.0
    cnt = 0
    num_samples, num_correct, num_predicted, num_fall = 0.0, 0.0, 0.0, 0.0
    for (idx, seqs) in enumerate(sequences):
        hidden = net.init_hidden(cuda)
        xs, ys = seqs
        loss = 0.0
        num_x = 0
        for i in range(len(xs)):
            x, y = Variable(xs[i]), Variable(ys[i]).view(-1)
            if cuda: x, y = x.cuda(), y.cuda()
            
            pred_y, hidden = net(x, hidden)
            if np.random.rand(1,1) < prob:
                hidden = net.init_hidden(cuda)
                cnt += 1
            # cross entropy
            loss += error(pred_y, y)
            num_x += len(x)
            
            # accuracy
            _, predicted = torch.max(pred_y.data, 1)
            num_samples += y.size()[0]
            num_correct += (predicted == y.data).sum()
        losses += (loss / num_x)
    if len(sequences) > 0 and is_training:
        loss.backward()
        optimizer.step()
    num_correct = num_correct.item()
    return losses / len(sequences), num_correct / num_samples


def trainEpochs(train_seq, num_epochs, net, error, cuda, prob, learning_rate):
    for i in range(num_epochs):
        optimizer = optim.Adam(net.parameters(), lr = learning_rate)
        
        print ("epoch {}".format(i + 1))
        train_loss, _ = train(train_seq, net, error, optimizer, cuda, prob)
        print ("train set loss is {}".format(train_loss.data[0]))


def validEpochs(valid_seq, net, error, cuda, prob):
    
    learning_rate = 0.001
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    valid_loss, valid_accuracy = train(valid_seq, net, error, optimizer, cuda, prob, False)
    print ("valid set loss is {}".format(valid_loss.data[0]))
    print ("valid set accuracy is {}".format(valid_accuracy))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action = 'store_true', 
                        help = 'use cuda')
    parser.add_argument('--path', type = str, default = 'data', 
                        help = 'location of data')
    parser.add_argument('--label_path', type = str, default = 'data/label.txt', 
                        help = 'location of label')
    parser.add_argument('--num_epochs', type = int, default = 20,
                        help = 'number of epochs')
    parser.add_argument('--batch_size', type = int, default = 1,
                        help = 'batch size')
    parser.add_argument('--input_size', type = int, default = 10,
                        help = 'input size')
    parser.add_argument('--hidden_size', type = int, default = 100,
                        help = 'hidden size')
    parser.add_argument('--num_hidden_layer', type = int, default = 2,
                        help = 'num hidden layer')
    parser.add_argument('--output_size', type = int, default = 2,
                        help = 'output size')
    parser.add_argument('--bptt', type = int, default = 1,
                        help = 'bptt size')
    parser.add_argument('--prob', type = float, default = 0.15,
                        help = 'reset probability')
    parser.add_argument('--num_components', type = int, default = 10)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--is_training', action = 'store_true',
                        help = 'is training')
    parser.add_argument('--model_name', type = str , default = 'model/net')
    args = parser.parse_args()
    args.cuda = False
    args.model_name = 'model/net' + str(args.prob)
    # define network
    net = Net(args.batch_size, args.input_size, args.hidden_size, args.num_hidden_layer, args.output_size, args.bptt)
    if torch.cuda.is_available and args.cuda: net = net.cuda()

    # cross entropy error
    error = nn.CrossEntropyLoss()
    
    # read data
    folders = ['1', '2', '3', '4', '5', '6']
    sequences = net.read_sequence(args.path, args.num_components)
    train_seq, valid_seq = net.split_train_valid(sequences)
    if args.is_training:
        print ("Train")
        trainEpochs(train_seq, args.num_epochs, net, error, args.cuda, args.prob, args.lr)
        torch.save(net, args.model_name)
    else:
        print ("Valid")
        net = torch.load(args.model_name)
        #validEpochs(valid_seq, net, error, args.cuda, args.prob)
        validEpochs(sequences, net, error, args.cuda, args.prob)
        


