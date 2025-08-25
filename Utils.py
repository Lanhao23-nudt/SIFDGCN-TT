import configparser
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy.sparse.linalg import eigs


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + 'Created successfully')
        return True
    else:
        print(path + 'Folder already exists')
        return False


# Add context to the origin data and label
# EOG, ...
def AddContext_New(x, context, dtype=float):

    cut = context - 1
    x_padding = x[0]
    x_data_left_padding = np.tile(x_padding, (cut, 1))
    # print(x_data_left_padding)
    # print(x_data_left_padding.shape)
    x_data_all = np.concatenate((x_data_left_padding, x), axis=0)

    x_Data = np.zeros([x.shape[0], context, x.shape[1]], dtype=dtype)
    for i in range(x.shape[0]):
        x_Data[i] = x_data_all[i: i+context]

    ret = x_Data
    # print(ret.shape)

    return ret


# EEG
def AddContext_New_EEG(x, context, dtype=float):
    cut = context - 1
    x_padding = x[0]
    x_padding = np.expand_dims(x_padding, 0)
    # print(x_padding.shape)
    x_data_left_padding = np.tile(x_padding, (cut, 1, 1))  # 沿着第一维，重复cut次，其他维度不变
    # print(x_data_left_padding)
    # print(x_data_left_padding.shape)
    x_data_all = np.concatenate((x_data_left_padding, x), axis=0)  # 第一个数据前补充cut个数据，与第一个数据保持一致，方便后续添加上下文
    # print(x_data_all.shape)

    x_Data = np.zeros([x.shape[0], context, x.shape[1], x.shape[2]], dtype=dtype)
    for i in range(x.shape[0]):
        x_Data[i] = x_data_all[i: i+context]  # 取后续context个数据，此处设置为5，即取包括自己在内的后5个数据，构成一组数据

    ret = x_Data
    # print(ret.shape)

    return ret