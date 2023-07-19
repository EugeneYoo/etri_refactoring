import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from QDS import QDS
from modelDir import datasetDir

torch.manual_seed(1)

data = pd.read_csv("./dataset/QDSRawDataset.csv", encoding='utf-8-sig', index_col=False)

#shuffle
data = data.sample(frac=1).reset_index(drop=True)



def makeData(data):

    train_data = data[:]
    train_x, train_y = train_data.iloc[:, :600], train_data.iloc[:, 600]

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(train_y)

    train_y = train_y.unsqueeze(1)

    return train_x, train_y

def makeDataforName(data):
    data = pd.read_csv("./data/{}".format(data), encoding='utf-8-sig', index_col=False)

    train_data = data[:]
    train_x, train_y = train_data.iloc[:, :600], train_data.iloc[:, 600]

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(train_y)

    train_y = train_y.unsqueeze(1)

    return train_x, train_y

def train_QDS():
    #try:
    train_x, train_y = makeData(data)

    model = QDS()

    # optimizer 설정
    optimizer = optim.SGD(model.parameters(), lr=1)

    nb_epochs = 1000
    for epoch in range(nb_epochs + 1):

        # H(x) 계산
        hypothesis = model(train_x)

        # cost 계산
        cost = F.binary_cross_entropy(hypothesis, train_y)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 20번마다 로그 출력
        if epoch % 10 == 0:
            prediction = hypothesis >= torch.FloatTensor([0.5])  # 예측값이 0.5를 넘으면 True로 간주
            correct_prediction = prediction.float() == train_y  # 실제값과 일치하는 경우만 True로 간주
            accuracy = correct_prediction.sum().item() / len(correct_prediction)  # 정확도를 계산
            print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(  # 각 에포크마다 정확도를 출력
                epoch, nb_epochs, cost.item(), accuracy * 100,
            ))

            torch.save(model.state_dict(), "./model/qds/[1023_{:.6f}]qds.pt".format(cost.item()))

            '''
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': cost.item()
            }, "../models/qds/[1004_{:.6f}]qds.pt".format(cost.item()))
            '''
    last_file_name = "./model/qds/[1023_{:.6f}]qds.pt".format(cost.item())
    last_rename_file = r"C:\Users\hi\Desktop\khu_2\khu\Datahub\model\qds.pt"

    shutil.copy(last_file_name, last_rename_file)

    print("complete train")

    return True


def train_QDS_with_Name(data, model_id):
    #try:
    train_x, train_y = makeDataforName(data)

    model = QDS()

    # optimizer 설정
    optimizer = optim.SGD(model.parameters(), lr=1)

    nb_epochs = 1000
    for epoch in range(nb_epochs + 1):

        # H(x) 계산
        hypothesis = model(train_x)

        # cost 계산
        cost = F.binary_cross_entropy(hypothesis, train_y)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 20번마다 로그 출력
        if epoch % 10 == 0:
            prediction = hypothesis >= torch.FloatTensor([0.5])  # 예측값이 0.5를 넘으면 True로 간주
            correct_prediction = prediction.float() == train_y  # 실제값과 일치하는 경우만 True로 간주
            accuracy = correct_prediction.sum().item() / len(correct_prediction)  # 정확도를 계산
            print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(  # 각 에포크마다 정확도를 출력
                epoch, nb_epochs, cost.item(), accuracy * 100,
            ))

            torch.save(model.state_dict(), "./model/qds/[1023_{:.6f}]qds.pt".format(cost.item()))
            last_file_name = "./model/qds/[1023_{:.6f}]qds.pt".format(cost.item())

    last_rename_file = "./data/{}".format(model_id)

    shutil.copy(last_file_name, last_rename_file)

    print("complete train")

    return True
