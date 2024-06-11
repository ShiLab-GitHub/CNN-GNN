import numpy as np
import pandas as pd
import sys, os
import torch
import torch.nn as nn
from model.cnn_gnn import CNN_GNN
from sklearn import metrics
from utils import *
import matplotlib.pyplot as plt


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
            loss_history[epoch-1] = loss.item()


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def calc_auc(y_labels, y_scores):
    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

modeling = [CNN_GNN][0]
model_st = modeling.__name__

cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 300

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
print('\nrunning on ', model_st + '_bace')
processed_data_file_train = 'data/processed/bace_train.pt'
processed_data_file_test = 'data/processed/bace_test.pt'
if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
    print('please run create_data.py to prepare data in pytorch format!')
else:
    train_data = TestbedDataset(root='data', dataset='bace_train')
    test_data = TestbedDataset(root='data', dataset='bace_test')
    data = pd.read_csv("data/bace_test.csv")
    class_test = data.Class.values.tolist()
    y_test = data.pIC50.values.tolist()

    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_mse = 1000
    best_ci = 0
    best_rate = 0
    best_auc = 0
    best_epoch = -1
    model_file_name = 'model_' + model_st + '_bace.model'
    result_file_name = 'result_' + model_st + '_bace.csv'
    Rate = np.zeros(NUM_EPOCHS)
    AUC_rate = np.zeros(NUM_EPOCHS)
    loss_history = np.zeros(NUM_EPOCHS)
    torch.cuda.empty_cache()
    Class_G = class_test
    for i in range(len(class_test)):
        if Class_G[i] == 'active':
            Class_G[i] = 1
        else:
            Class_G[i] = 0
    for epoch in range(NUM_EPOCHS):
        if epoch-best_epoch>50:
            LR = LR/10
        train(model, device, train_loader, optimizer, epoch + 1)
        G, P = predicting(model, device, test_loader)
        r = 0
        for i in range(len(P)):
            if P[i] >= 6:
                temp = 1
            else:
                temp = 0
            if temp == class_test[i]:
                r +=1
        rate = r/len(P)
        if rate>best_rate:
            best_rate = rate
        print('precision rate:',rate)
        Rate[epoch] = rate
        Score_P = P/(max(P))
        AUC_rate[epoch] = calc_auc(Class_G, Score_P)
        print('AUC:',AUC_rate[epoch])
        if AUC_rate[epoch]>best_auc:
            best_auc = AUC_rate[epoch]
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
        if ret[1] < best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_epoch = epoch + 1
            best_mse = ret[1]
            best_ci = ret[-1]
            print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci)
            print('best_accuracy:', best_rate, ';best_auc:', best_auc, model_st)
        else:
            print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci)
            print('best_accuracy:', best_rate, ';best_auc:', best_auc, model_st)
    n = range(NUM_EPOCHS)
    plt.figure()
    plt.plot(n,Rate)
    plt.xlabel('epoch')
    plt.title('ACC')
    plt.show()

    plt.figure()
    plt.plot(n, AUC_rate)
    plt.xlabel('epoch')
    plt.title('AUC')
    plt.show()

    plt.figure()
    plt.plot(n, loss_history)
    plt.xlabel('epoch')
    plt.title('loss')
    plt.show()