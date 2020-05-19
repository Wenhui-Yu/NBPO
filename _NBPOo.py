## NBPOo (NBPO with original loss)
## Wenhui Yu 2020.04.26
## author @Wenhui Yu, yuwh16@mails.tsinghua.edu.cn

## Parameters setting
dataset = 0                         # datasets, 0 for Amazon, 1 for movielens
eta = [0.005, 0.001][dataset]       # learning rate
lambda_r = [1, 1][dataset]          # coefficient of regularization term
lambda_phi = [0.5, 5][dataset]      # coefficient of regularization term
K0 = 50                             # length of latent factors
K1 = 50                             # length of latent factors for probability
vali_test = 0                       # 0 for validation set, 1 for test set
sample_rate = 4                     # sample rate, the number of negative samples foreach positive one
batch_size = 5000                   # batch size
epoch = 200                         # number of epochs
top_k = [2, 5, 10, 20, 50, 100]     # top k items to recommend
Model = 'NBPOo'
## Paths to save and read
# list for datasets
dataset_list = ['amazon', 'movielens']
path_train = 'dataset\\' + dataset_list[dataset] + '\\train_data.json'
path_validation = 'dataset\\' + dataset_list[dataset] + '\\validation_data.json'
path_test = 'dataset\\' + dataset_list[dataset] + '\\test_data.json'

import numpy as np
from numpy import *
import xlwt
import time
from Library import readdata
from Library import evaluation_F1
from Library import evaluation_NDCG
from Library import save_parameters
from Library import save_df

import xlrd, xlwt
from xlutils.copy import copy as xl_copy
from openpyxl import load_workbook
from openpyxl import Workbook
import pandas as pd

def d(x):
    # sigmoid(x)
    if x > 10:
        return 1
    if x < -10:
        return 0
    if x >= -10 and x <= 10:
        return 1.0 / (1.0 + exp(-x))

def test_Model(U, V):
    # test the precision
    k_num = len(top_k)
    # k_num-length list to record F1 and NDCG
    F1 = np.zeros(k_num)
    NDCG = np.zeros(k_num)

    # test all test samples
    user_num = 0
    for u in range(M):
        # the data in test set is [[i, i, i, i],...,[i, i]]
        test_item = Test[u]
        if len(test_item) > 0:
            user_num += 1
            # score all items
            score = np.dot(V, U[u])
            # order
            b = zip(score, range(N))
            b.sort(key=lambda x: x[0])
            order = [x[1] for x in b]
            order.reverse()
            # remove the training samples from the recommendations
            train_positive = train_data_aux[u]
            for item in train_positive:
                order.remove(item)
            # for each k, calculate top_k
            for i in range(len(top_k)):
                F1[i] += evaluation_F1(order, top_k[i], test_item)
                NDCG[i] += evaluation_NDCG(order, top_k[i], test_item)
    # calculate the average
    F1 = (F1 / user_num).tolist()
    NDCG = (NDCG / user_num).tolist()

    return F1, NDCG

def train_Model(eta):
    # training the model
    # initialization
    U = np.array([np.array([(random.random() / math.sqrt(K0)) for j in range(K0)]) for i in range(M)])
    V = np.array([np.array([(random.random() / math.sqrt(K0)) for j in range(K0)]) for i in range(N)])
    P = 5 * np.array([np.array([(random.random() / math.sqrt(K1)) for j in range(K1)]) for i in range(M)])
    Q = 5 * np.array([np.array([(-random.random() / math.sqrt(K1)) for j in range(K1)]) for i in range(N)])
    e = 10 ** 10
    # output a result without training
    print 'iteration ', 0,
    [F1, NDCG] = test_Model(U, V)
    Fmax = 0
    if F1[0] > Fmax:
        Fmax = F1[0]
    print Fmax, 'F1: ', F1, '  ', 'NDCG: ', NDCG
    # save in .xls file
    F1_df = pd.DataFrame(columns=top_k)
    NDCG_df = pd.DataFrame(columns=top_k)
    F1_df.loc[0] = F1
    NDCG_df.loc[0] = NDCG
    save_df([[F1_df, 'F1'], [NDCG_df, 'NDCG']], path_excel, first_sheet=False)  # @x
    # get the numer of training samples
    Re = len(train_data)
    # split the training samples with batch_size
    bs = range(0, Re, batch_size)
    bs.append(Re)
    # begin iterating
    for ep in range(epoch):
        print 'iteration ', ep + 1,
        eta = eta * 0.99
        # for each iterating, we user all training samples to train
        for ii in range(len(bs) - 1):
            if abs(U.sum()) < e:
                # input the samples as batches
                # initialize dU and dC to record the gradient
                dU = np.zeros((M, K0))
                dV = np.zeros((N, K0))
                dP = np.zeros((M, K1))
                dQ = np.zeros((N, K1))
                for re in range(bs[ii], bs[ii + 1]):
                    # iterate each batch
                    # train data, [u, i]
                    u = train_data[re][0]
                    i = train_data[re][1]
                    Ri = d(np.dot(U[u], V[i]))
                    gammai = d(np.dot(P[u], Q[i]))
                    # select negative samples randomly
                    num = 0
                    while num < sample_rate:
                        j = int(random.uniform(0, N))
                        # check if the current sample is positive sample
                        if not (j in train_data_aux[u]):
                            num += 1
                            Rj = np.dot(U[u], V[j])
                            gammaj = np.dot(P[u], Q[j])
                            dU[u] += d(-Ri) * V[i]
                            dV[i] += d(-Ri) * U[u]
                            dP[u] += -d(gammai) * Q[i]
                            dQ[i] += -d(gammai) * P[u]

                            D = ((d(gammaj) - 1) * d(Rj) * d(-Rj)) / (d(-Rj) + d(gammaj) * d(Rj) + 0.1 ** 10)
                            dU[u] += D * V[j]
                            dV[j] += D * U[u]
                            D = (d(gammaj) * d(-gammaj) * d(Rj)) / (d(-Rj) + d(gammaj) * d(Rj) + 0.1 ** 10)
                            dP[u] += D * Q[j]
                            dQ[j] += D * P[u]

                # update the matrices
                U += eta * (dU - lambda_r * U)
                V += eta * (dV - lambda_r * V)
                P += eta * (dP - lambda_phi * P)
                Q += eta * (dQ - lambda_phi * Q)
        if abs(U.sum()) < e:
            [F1, NDCG] = test_Model(U, V)
            if F1[0] > Fmax:
                Fmax = F1[0]
            F1_df.loc[ep + 1] = F1
            NDCG_df.loc[ep + 1] = NDCG
            print Fmax, 'F1: ', F1, '  ', 'NDCG: ', NDCG
            save_df([[F1_df, 'F1'], [NDCG_df, 'NDCG']], path_excel, first_sheet=False)  # @x
        else:
            break
    return F1_df, NDCG_df

def print_parameter():
    # print all parameters
    print 'model:', Model
    print 'dataset:', dataset_list[dataset]
    print 'eta:', eta
    print 'lambda_r:', lambda_r, 'lambda_phi:', lambda_phi
    print 'K0:', K0, 'K1:', K1
    print 'vali_test:', ['validation', 'test'][vali_test]
    print 'sample_rate:', sample_rate
    print 'batch_size:', batch_size
    print 'epoch:', epoch
    print 'top_k:', top_k
    print

'''**************************main_function***************************'''
'''**************************main_function***************************'''
# load the data
[train_data, train_data_aux, M, N] = readdata(path_train)
validation_data = readdata(path_validation)[1]
test_data = readdata(path_test)[1]
# choose validation or test set
if vali_test == 0:
    Test = validation_data
else:
    Test = test_data

data = [
    ["Model", [Model]],
    ["dataset", [dataset_list[dataset]]],
    ["eta", [eta]],
    ["lambda_r", [lambda_r]],
    ["lambda_phi", [lambda_phi]],
    ["K0", [K0]],
    ["K1", [K1]],
    ["vali_test", [['validation', 'test'][vali_test]]],
    ["sample_rate", [sample_rate]],
    ["batch_size", [batch_size]],
    ['epoch', [epoch]],
    ["top_k", top_k]
]
path_excel = 'experiment_result\\' + dataset_list[dataset] + '_' + Model + '_' + str(int(time.time())) + str(int(random.uniform(100, 900))) + '.xlsx'
print_parameter()
save_parameters(data, path_excel)
F1_df, NDCG_df = train_Model(eta)
