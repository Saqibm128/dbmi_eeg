from sacred import Experiment
ex = Experiment('simple_nn_eeg')
from sacred.observers import MongoObserver

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

import util_funcs
import pandas as pd
import numpy as np
from models.ffn import NeuralNet
import sklearn.model_selection
from sklearn.metrics import f1_score


@ex.capture
def get_data(use_1, use_both, num_files):
    if use_both:
        x_1_data, y_data = get_data(True, False)
        x_2_data, y_data_dup = generate_x_y(False, False)
        assert (y_data == y_data_dup).all().all()
        x_data = np.hstack([x_1_data, x_2_data])
    elif not use_1:
        data_all = util_funcs.read_all(use_1, num_files=num_files)
        x_data = np.vstack([instance.data.mean(axis=0) for instance in data_all])
    else:
        data_all = util_funcs.read_all(use_1, num_files=num_files)
        x_data = np.vstack([instance.data.mean(axis=0, keepdims=True) for instance in data_all])
        x_data = x_data.reshape(x_data.shape[0], -1)
    y_data_strings = [instance.seizure_type for instance in data_all]
    y_data = pd.DataFrame(
        index=range(num_files),
        columns=util_funcs.get_seizure_types()
        ).fillna(0)
    for i in range(num_files):
        y_data.loc[i, y_data_strings[i]] = 1
    return x_data, y_data.to_numpy()

@ex.named_config
def debug_config():
    batch_print_size = 1
    num_epochs = 10

@ex.config
def config():
    use_1 = False
    use_both = False
    num_files = util_funcs.TOTAL_NUM_FILES
    clf_step = None
    use_expanded_y = True
    clf_name = "simple_nn.pt"
    num_epochs = 1000
    ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_size = 0.2
    valid_size = 0.25
    batch_size = 50
    batch_print_size = 10
    lr = 0.001
    momentum = 0.9
    step_size = 50
    gamma = 0.8

@ex.automain
def main(test_size, valid_size, batch_size, num_epochs, batch_print_size, device, clf_name, lr, momentum, step_size, gamma):
    x_data, y_data = get_data()
    net = NeuralNet(
            x_data.shape[1],
            x_data.shape[1] * 10,
            len(util_funcs.get_seizure_types())
        ).to(device)
    best_model = net
    best_acc = -100
    x_train_plus, x_test, y_train_plus, y_test = \
        sklearn.model_selection.train_test_split(x_data,
                                                 y_data,
                                                 test_size=test_size,
                                                 # stratify=y_data
                                                 )
    x_train, x_valid, y_train, y_valid = \
        sklearn.model_selection.train_test_split(
            x_train_plus,
            y_train_plus,
            test_size=valid_size)

    trainloader = DataLoader(TensorDataset(torch.Tensor(x_train).type(torch.FloatTensor).to(device),
                                           torch.Tensor(y_train).type(torch.LongTensor).to(device)),
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
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
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
            ex.add_artifact(clf_name)
            best_acc = valid_score
        if epoch % batch_print_size == 0:
            print('Validation F1: %.3f' % valid_score)
        scheduler.step()

    net.eval()
    best_model = torch.load(clf_name)
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
