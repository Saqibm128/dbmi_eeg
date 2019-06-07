from sacred import Experiment
ex = Experiment('simple_lstm_eeg')
from sacred.observers import MongoObserver

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

import util_funcs
import pandas as pd
import numpy as np
from models.lstm import LSTM
import sklearn.model_selection
from sklearn.metrics import f1_score

#sources:https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e

@ex.capture
def get_data(use_1, use_both, num_files):
    if use_both:
        raise NotImplementedError("Not Implemented Yet")
        # x_1_data, y_data = get_data(True, False)
        # x_2_data, y_data_dup = get_data(False, False)
        # assert (y_data == y_data_dup).all().all()
        # x_data = np.hstack([x_1_data, x_2_data])
        # return x_data, y_data
    elif not use_1:
        data = util_funcs.read_all(use_1, num_files=num_files)
        lengths = [data[i].data.shape[0] for i in range(len(data))]
        width = 900
        paddedX = np.zeros((num_files, max(lengths), width))
        for i in range(len(data)):
            paddedX[i, 0:lengths[i]] = data[i].data
        paddedY = np.zeros((num_files, max(lengths)))
        for i in range(len(data)):
            paddedY[i, 0:lengths[i]] = util_funcs.get_seizure_types().index(data[i].seizure_type)
        return paddedX, paddedY, lengths
    else:
        data = util_funcs.read_all(use_1, num_files=num_files)
        lengths = [data[i].data.shape[0] for i in range(len(data))]
        paddedX = np.zeros((num_files, max(lengths), 20, 24))
        for i in range(len(data)):
            paddedX[i, 0:lengths[i]] = data[i].data
        paddedY = np.zeros((num_files, max(lengths)))
        for i in range(len(data)):
            paddedY[i, 0:lengths[i]] = util_funcs.get_seizure_types().index(data[i].seizure_type)
        return paddedX, paddedY, lengths

@ex.capture
def create_model(input_size, hidden_size_factor, num_lstm_layers, num_classes):
    return LSTM(input_size, input_size*hidden_size_factor, num_lstm_layers, num_classes)

@ex.config
def config():
    use_1 = True
    num_files = 150
    use_both = False
    num_epochs = 100
    test_size = 0.2
    valid_size = 0.25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 50
    hidden_size_factor = 10
    num_lstm_layers = 1
    num_classes = len(util_funcs.get_seizure_types())
    lr = 0.001
    gamma = 1 #turn off lr scheduler
    momentum = 0.9
    step_size = 50
    num_epochs = 100
    clf_name = "simple_lstm_eeg.pt"

@ex.automain
def main(test_size, valid_size, device, batch_size, lr, gamma, momentum, step_size, num_epochs, clf_name):
    paddedX, paddedY, lengths = get_data()
    net = create_model(paddedX.shape[2])

    best_acc = -100
    x_train_plus, x_test, y_train_plus, y_test, lengths_plus, lengths_test = \
        sklearn.model_selection.train_test_split(paddedX,
                                                 paddedY,
                                                 lengths,
                                                 test_size=test_size,
                                                 # stratify=y_data
                                                 )

    x_train, x_valid, y_train, y_valid, lengths_train, lengths_valid = \
        sklearn.model_selection.train_test_split(
            x_train_plus,
            y_train_plus,
            lengths_plus,
            test_size=valid_size)

    trainloader = DataLoader(TensorDataset(torch.Tensor(x_train).type(torch.FloatTensor).to(device),
                                           torch.Tensor(y_train).type(torch.LongTensor).to(device),
                                           torch.Tensor(lengths_train).type(torch.LongTensor).to(device),
                                           ),
                             batch_size=batch_size)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    #based off of https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        net.train()

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, batch_lengths = data

            rnn_padded_input = torch.nn.utils.rnn.pack_padded_sequence(inputs, batch_lengths, batch_first=True, enforce_sorted=False)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            return "early"

            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()


            # print statistics
            running_loss += loss.item()

            tensor_x_valid = torch.Tensor(x_valid).to(device)
            tensor_y_valid = torch.Tensor(y_valid).to(device)
            outputs = net(tensor_x_valid)



            if i % 5 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        net.eval()
        valid_score = f1_score(torch.max(tensor_y_valid, 1)[1].cpu().detach().numpy(),
                       outputs.cpu().detach().numpy().argmax(axis=1),
                       average="weighted"
                       )
        if valid_score > best_acc:
            torch.save(net, clf_name)
            best_acc = valid_score
        if epoch % batch_print_size == 0:
            print('Validation F1: %.3f' % valid_score)
        scheduler.step()

    net.eval()
    best_model = torch.load(clf_name)
    ex.add_artifact(clf_name) #add once to avoid overloading mongodb
    tensor_x_test = torch.Tensor(x_test).to(device)
    tensor_y_test = torch.Tensor(y_test).to(device)
    outputs = net(tensor_x_test)

    test_score = f1_score(torch.max(tensor_y_test, 1)[1].cpu().detach().numpy(),
                   outputs.cpu().detach().numpy().argmax(axis=1),
                   average="weighted"
                   )
    print('Test F1: %.3f' % test_score)

    print('Finished Training')
    return test_score, \
        (torch.max(tensor_y_test, 1)[1].cpu().detach().numpy(), \
            outputs.cpu().detach().numpy().argmax(axis=1))
