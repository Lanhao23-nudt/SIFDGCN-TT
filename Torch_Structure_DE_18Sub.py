import numpy as np
import torch
import mat73
import os
import h5py
import random
from sklearn.model_selection import StratifiedKFold, KFold  # StratifiedKFold主要用于分类任务，类别不平衡时，让每一折总体比例是平衡的
from torch.utils.tensorboard import SummaryWriter

import torch.utils.data as Data
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy import io

from Utils import *
from SIFDGCN_TT import SIFDGCN_TT

def setup_seed(seed): # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_model(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_train_loss = 0
    # total_train_acc = 0

    total = 0
    predict = []
    label_y = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        predict.append(outputs.cpu().detach().numpy())
        label_y.append(labels.cpu().detach().numpy())

        outputs = outputs.squeeze(1)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total += labels.size(0)
        total_train_loss += loss.item()*labels.size(0)

    # Calculate average loss and Print
    epoch_loss = total_train_loss / total
    print("train average loss：{:.6f}".format(epoch_loss))

    predict = np.concatenate(predict)
    label_y = np.concatenate(label_y)

    predict = predict.squeeze(1)
    coeff, p_value = pearsonr(predict, label_y)
    rmse = np.sqrt(np.mean((predict - label_y) ** 2))

    return coeff, rmse, epoch_loss

def evaluate_model(model, test_dataloader, loss_fn, device):
    model.eval()
    total_test_loss = 0
    # total_test_acc = 0

    running_loss = 0.0
    total = 0

    predict = []
    label_y = []
    with torch.no_grad():
        for data, label in test_dataloader:
            data, label = data.to(device), label.to(device)
            outputs = model(data)

            predict.append(outputs.cpu().detach().numpy())
            label_y.append(label.cpu().detach().numpy())

            total += label.size(0)
            outputs = outputs.squeeze(1)
            test_loss = loss_fn(outputs, label)
            total_test_loss = total_test_loss + test_loss.item()*label.size(0)
    epoch_loss = total_test_loss / total
    print("test average loss：{:.6f}".format(epoch_loss))

    predict = np.concatenate(predict)
    label_y = np.concatenate(label_y)

    predict = predict.squeeze(1)
    coeff, p_value = pearsonr(predict, label_y)
    rmse = np.sqrt(np.mean((predict - label_y) ** 2))

    return predict, coeff, rmse, epoch_loss


if __name__ == '__main__':
    # subjects
    sub_num = 18
    Phy_type = 'eeg'
    root_path = './SIFDGCN_TT_General'   # store path
    data_path = 'D:/MMV dataset/'  # dataset path
    # [train] parameters
    fold_num = 10
    num_epochs = 200
    batch_size = 20
    context = 5
    lr = 0.0001
    Fre_num = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # RMSE, COR and P
    VAE_RMSE2 = np.array([])
    VAE_COR2 = np.array([])
    VAE_P2 = np.array([])

    VAE_RMSE1 = np.array([])
    VAE_COR1 = np.array([])
    VAE_P1 = np.array([])

    # RMSE, COR and P
    VAE_RMSE2_final = np.array([])
    VAE_COR2_final = np.array([])
    VAE_P2_final = np.array([])

    VAE_RMSE1_final = np.array([])
    VAE_COR1_final = np.array([])
    VAE_P1_final = np.array([])

    # Preparation for LGG
    original_order = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
                      'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
                      'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
                      'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
                      'O2', 'CB2']

    graph_general_MMV = [['FP1', 'FPZ', 'FP2'], ['AF3', 'AF4'], ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
                         ['FT7', 'T7', 'TP7'], ['FT8', 'T8', 'TP8'], ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6'],
                         ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6'], ['CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'],
                         ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
                         ['PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8'],
                         ['O1', 'OZ', 'O2'], ['CB1', 'CB2']]

    graph_Frontal_MMV = [['FP1', 'AF3'], ['FP2', 'AF4'], ['FPZ', 'FZ', 'FCZ'], ['F7', 'F5', 'F3', 'F1'], ['F2', 'F4', 'F6', 'F8'],
                         ['FT7', 'T7', 'TP7'], ['FT8', 'T8', 'TP8'], ['FC5', 'FC3', 'FC1'], ['FC2', 'FC4', 'FC6'],
                         ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6'], ['CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'],
                         ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
                         ['PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8'],
                         ['O1', 'OZ', 'O2'], ['CB1', 'CB2']]

    graph_Hemisphere_MMV = [['FP1', 'AF3'], ['FP2', 'AF4'], ['F7', 'F5', 'F3', 'F1'], ['F2', 'F4', 'F6', 'F8'],
                            ['FT7', 'T7', 'TP7'], ['FT8', 'T8', 'TP8'], ['FC5', 'FC3', 'FC1'], ['FC2', 'FC4', 'FC6'],
                            ['C5', 'C3', 'C1'], ['C2', 'C4', 'C6'], ['FPZ', 'FZ', 'FCZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ'],
                            ['CP5', 'CP3', 'CP1'], ['CP2', 'CP4', 'CP6'], ['P7', 'P5', 'P3', 'P1'],
                            ['P2', 'P4', 'P6', 'P8'], ['PO7', 'PO5', 'PO3', 'O1', 'CB1'],
                            ['PO4', 'PO6', 'PO8', 'O2', 'CB2']]

    graph_idx = graph_general_MMV  # The general graph definition for DEAP is used as an example.
    idx = []
    num_chan_local_graph = []
    for i in range(len(graph_idx)):
        num_chan_local_graph.append(len(graph_idx[i]))
        for chan in graph_idx[i]:
            idx.append(original_order.index(chan))

    sub_list = list(range(1, 19))
    ses_list = [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1]
    sub_list = sub_list[:sub_num]
    ses_list = ses_list[:sub_num]
    for subname, session_num in zip(sub_list, ses_list):
    # for subname in range(sub_num):
    #     for session_num in range(4):
        path = os.path.join(root_path, 'model_pth/sub_%02d' % (subname), 'session_%02d' % (session_num))
        mkdir(path)  # 存储参数文件

        path_log = os.path.join(root_path, 'model_Log/sub_%02d' % (subname), 'session_%02d' % (session_num))
        mkdir(path_log)  # 存储训练log文件

        # load PERCLOS
        avearge_PERCLOS_path = os.path.join(data_path, 'Feature Data', 'subject-%02d' % (subname), 'session-%02d' % (session_num), 'sub-%02d_sess-%02d_blk-02_eye.mat' % (subname, session_num))
        dataset_PERCLOS = mat73.loadmat(avearge_PERCLOS_path)
        data_PERCLOS = dataset_PERCLOS['Data']['PERCLOS_sliding_average_data']
        # dataset_PERCLOS.close()

        # load physiological signals' features
        Phy_feature_path = os.path.join(data_path, 'Feature Data', 'subject-%02d' % (subname), 'session-%02d' % (session_num), 'sub-%02d_sess-%02d_blk-02_%s.mat' % (subname, session_num, Phy_type))
        dataset_Phy = h5py.File(Phy_feature_path)
        x_Phy = dataset_Phy['Data']['data']
        # X_Phy = x_Phy.transpose()
        X_Phy = np.transpose(np.array(x_Phy))

        # dataset_Phy.close()

        print(X_Phy.shape)
        print(data_PERCLOS.shape)

        X_DE_reshape = X_Phy

        # Preparation for LGG
        X_DE_reshape = X_DE_reshape[:, idx, :]

        X_data1 = X_DE_reshape
        Y_data = data_PERCLOS

        # Train the network
        setup_seed(13206)
        # np.random.seed(0)
        # torch.manual_seed(26551)
        # torch.cuda.manual_seed_all(26551)

        skf = KFold(n_splits=fold_num)
        pred = np.zeros((X_data1.shape[0]))

        pred_final = np.zeros((X_data1.shape[0]))

        # Run 10-fold CV
        n = 0
        for train_idx, test_idx in skf.split(X_data1):
            n = n + 1
            print('$$$$$$$$$$$$$$$$$$$ name_location: ', subname)
            print('$$$$$$$$$$$$$$$$$$$ Session : ', session_num)
            print('$$$$$$$$$$$$$$$$$$$ fold: ', n)

            Write = SummaryWriter(path_log + f"/Fold_{n}")
            # labels
            training_score1 = Y_data[train_idx]
            testing_score1 = Y_data[test_idx]
            train_y = training_score1
            test_y = testing_score1

            # EEG
            training_feature_EEG = X_data1[train_idx, :]
            testing_feature_EEG = X_data1[test_idx, :]
            train_x1 = training_feature_EEG
            test_x1 = testing_feature_EEG
            
            # min_max
            for ch in range(train_x1.shape[1]):  # 遍历每个通道
                min_val = np.min(train_x1[:, ch, :], axis=0)
                max_val = np.max(train_x1[:, ch, :], axis=0)
                train_x1[:, ch, :] = (train_x1[:, ch, :] - min_val) / (max_val - min_val)
                test_x1[:, ch, :] = (test_x1[:, ch, :] - min_val) / (max_val - min_val)
            
            train_x1 = AddContext_New_EEG(train_x1, context)
            test_x1 = AddContext_New_EEG(test_x1, context)  # 添加上下文信息，使得每个样本所具备的感受野更广一点，但是标签保持不变

            # Preparation for SFT
            # train_x1_4D = Data_4D_Generation(train_x1, Fre_num=Fre_num, Seg_num=context)
            # test_x1_4D = Data_4D_Generation(test_x1, Fre_num=Fre_num, Seg_num=context)
            # train_x1 = train_x1_4D
            # test_x1 = test_x1_4D

            # dataloader
            train_x1 = torch.FloatTensor(train_x1)
            test_x1 = torch.FloatTensor(test_x1)
            train_y = torch.FloatTensor(train_y)
            test_y = torch.FloatTensor(test_y)

            train_dataloader = DataLoader(Data.TensorDataset(train_x1, train_y), batch_size=batch_size, shuffle=True, num_workers=4)
            test_dataloader = DataLoader(Data.TensorDataset(test_x1, test_y), batch_size=batch_size, num_workers=4)

            # model
            myModel = SIFDGCN_TT(num_chan_local_graph, eletrode_num=62, d_model=32, segment_num=context, heads=1, Fre_num=5)
            myModel = myModel.to(device)
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.AdamW(myModel.parameters(), lr=lr, weight_decay=0.02)

            Cor_max = 0
            RMSE_min = 1
            # model train and test
            for epoch in range(num_epochs):
                # train
                coeff_train, rmse_train, loss_train = train_model(myModel, train_dataloader, loss_fn, optimizer, device)

                # test
                predict, coeff_test, rmse_test, loss_test = evaluate_model(myModel, test_dataloader, loss_fn, device)

                # Tensorboard record
                Write.add_scalars("COR",
                                  {'train_COR': coeff_train,
                                   'test_COR': coeff_test}, epoch + 1)  # 可视化表、线图
                Write.add_scalars("RMSE",
                                  {'train_RMSE': rmse_train,
                                   'test_RMSE': rmse_test}, epoch + 1)
                Write.add_scalars("Loss",
                                  {'train_Loss': loss_train,
                                   'test_Loss': loss_test}, epoch + 1)

                if rmse_test < RMSE_min:  # 保存最佳结果
                    Cor_max = coeff_test
                    RMSE_min = rmse_test

                    torch.save(myModel.state_dict(), path + '/Final_model_fold' + str(n) + '.pth')
                    pred1 = predict

            pred1 = np.squeeze(pred1)
            # print(pred1.shape)
            pred[test_idx] = pred1  # 每折最优结果

            Write.close()
            # Fold finish
            print(128 * '_')

        # %% 保存每一被试的所有预测结果
        # 训练到底结果
        y_results = np.squeeze(Y_data)
        # 每折最优结果
        pre_results = pred
        # y_results = np.squeeze(Y_data)

        coeff_VAE, p_value_VAE = pearsonr(pre_results, y_results)

        VAE_RMSE1 = np.append(VAE_RMSE1, np.sqrt(np.mean((pre_results - y_results) ** 2)))
        VAE_COR1 = np.append(VAE_COR1, coeff_VAE)
        VAE_P1 = np.append(VAE_P1, p_value_VAE)

        # save results_matrix
        mkdir(root_path + 'inter_regression_results/sub_' + str(subname) + '/session_' + str(session_num))
        mat_path = root_path + 'inter_regression_results/sub_' + str(subname) + '/session_' + str(session_num) + '/inter_results_NoAVE.mat'
        io.savemat(mat_path, {'inter_results1': pre_results})


        # 每折最优结果 + 滑动平均
        sliding_data1 = pre_results
        sliding_data11 = np.zeros((sliding_data1.shape[0]))
        for sliding_j in range(sliding_data1.shape[0]):
            if sliding_j > 6 and sliding_j < sliding_data1.shape[0] - 8:
                sliding_data11[sliding_j] = np.mean(sliding_data1[sliding_j - 7:sliding_j + 8])
            elif sliding_j <= 6:
                sliding_data11[sliding_j] = np.mean(sliding_data1[:sliding_j + 1])
            else:
                sliding_data11[sliding_j] = np.mean(sliding_data1[sliding_j:])

        coeff_VAE2, p_value_VAE2 = pearsonr(sliding_data11, y_results)
        VAE_RMSE2 = np.append(VAE_RMSE2, np.sqrt(np.mean((sliding_data11 - y_results) ** 2)))
        VAE_COR2 = np.append(VAE_COR2, coeff_VAE2)
        VAE_P2 = np.append(VAE_P2, p_value_VAE2)

        # save results_matrix
        pre_results_ave = sliding_data11
        mat_path = root_path + 'inter_regression_results/sub_' + str(subname) + '/session_' + str(session_num) + '/inter_results_AVE.mat'
        io.savemat(mat_path, {'inter_results1': pre_results_ave})

    # %% 将各个被试计算的COR与RMSE保存到同一文件
    # 保存所有被试，无滑动平均结果
    results1 = np.vstack((VAE_RMSE1, VAE_COR1, VAE_P1))  # 没有经过滑动平均的RMSE、COR和COR的显著性p
    results1 = results1.transpose()  # 将行列转置，实验完之后为，三列数据
    mean_results1 = np.array(
        [[VAE_RMSE1.mean(), (VAE_COR1[~np.isnan(VAE_COR1)]).mean(), (VAE_P1[~np.isnan(VAE_P1)]).mean()]])
    all_results1 = np.vstack((results1, mean_results1))  # 最后一列为平均值
    all_results1 = all_results1.round(3)

    print('results 1:')
    print(all_results1)

    mat_path1 = root_path + 'all_results1.mat'
    io.savemat(mat_path1, {'all_results1': all_results1})  # results1 无滑动平均


    # 保存所有被试，滑动平均结果
    results2 = np.vstack((VAE_RMSE2, VAE_COR2, VAE_P2))
    results2 = results2.transpose()
    mean_results2 = np.array([[VAE_RMSE2.mean(), (VAE_COR2[~np.isnan(VAE_COR2)]).mean(),
                               (VAE_P2[~np.isnan(VAE_P2)]).mean()]])
    all_results2 = np.vstack((results2, mean_results2))
    all_results2 = all_results2.round(3)

    print('results 2:')
    print(all_results2)

    mat_path3 = root_path + 'all_results2.mat'
    io.savemat(mat_path3, {'all_results2': all_results2})  # results2 有滑动平均
    # 标签首先采取了滑动平均的方式进行了平滑，故将预测值进行相同的滑动平均，必然会提升效果；
    # 但滑动平均会干扰和削弱模型的影响力，无法确定最终的结果是模型的结果还是滑动平均产生的影响；
    # 故最后论文中的结果，统一按照未滑动平均结果进行对比

    # The labels were first smoothed using a moving average,
    # so applying the same moving average to the predicted values will inevitably enhance the results
    # However, the moving average can interfere with and diminish the model's influence,
    # making it difficult to determine whether the final result is due to the model or the effect of the moving average.
    # Therefore, the results presented in the paper are uniformly compared based on the non-moving average results.