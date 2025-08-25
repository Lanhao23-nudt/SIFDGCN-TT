import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers import GraphConvolution,Linear
# from utils import normalize_A, generate_cheby_adj
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_A(A,lmax=2):
    A = F.relu(A)
    N = A.shape[0]
    A = A*(torch.ones(N,N).cuda()-torch.eye(N,N).cuda())  # 此处去除了自环边
    A = A+A.T  # 表示无向图
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))  # 根号2度矩阵
    D = torch.diag_embed(d)  # 化为度矩阵
    L = torch.eye(N,N).cuda()-torch.matmul(torch.matmul(D, A), D)
    # L = D^{-1/2} (D-A) D^{-1/2}=I - D^{-1/2} A D^{-1/2} 归一化到0-1之间
    Lnorm=(2*L/lmax)-torch.eye(N,N).cuda()  # 符合论文中的归一化方式，但是论文中的lmax应该是动态计算得到的
    return Lnorm


def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).cuda())
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L,support[-1],)-support[-2]
            support.append(temp)
    return support  # 完全符合论文中的计算过程


class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc = nn.ModuleList()
        for i in range(K):
            self.gc.append(GraphConvolution( in_channels,  out_channels))

    def forward(self, x,L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc)):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])  # Tk*x
        result = F.relu(result)
        return result

class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).cuda())
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).cuda())
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)

class DGCNN(nn.Module):
    def __init__(self, in_channels,num_electrodes, k_adj, out_channels, num_classes=3):
        #in_channels(int): The feature dimension of each electrode.
        #num_electrodes(int): The number of electrodes.
        #k_adj(int): The number of graph convolutional layers.
        #out_channel(int): The feature dimension of  the graph after GCN.
        #num_classes(int): The number of classes to predict.
        super(DGCNN, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).to(device))
        nn.init.uniform_(self.A,0.01,0.5)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)  # data can also be standardized offline

        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = self.fc(result)
        # result = result.view(result.shape[0], 1, -1)
        return result


class Aggregator():
    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area
        self.chan_in_area = idx_area
        self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)

    def forward(self, x):
        # x: batch x channel x data
        data = []
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
                # 将每个电极区域的EEG信号进行了取平均操作,从而形成新的节点信号
            else:
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)

    def get_idx(self, chan_in_area):
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)
        return idx_[1:]

    def aggr_fun(self, x, dim):
        # return torch.max(x, dim=dim).values
        return torch.mean(x, dim=dim)


class SIFDGCN(nn.Module):
    def __init__(self, channel_num, feature_num, idx_graph, out_channel):
        super(SIFDGCN, self).__init__()
        self.idx = idx_graph
        self.channel = channel_num
        self.brain_area = len(self.idx)
        self.brain_graph = channel_num + len(self.idx)

        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, feature_num),
                                                requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)

        self.aggregate = Aggregator(self.idx)

        self.DGCN = DGCNN(in_channels=feature_num, num_electrodes=self.brain_graph, out_channels=out_channel, k_adj=3, num_classes=out_channel)
    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)  # 对逐点相乘+偏置
        return x

    def forward(self, x):
        shallow = self.local_filter_fun(x, self.local_filter_weight)
        deep = self.aggregate.forward(x)

        out = torch.cat((shallow, deep), dim=1)
        out = self.DGCN(out)
        return out


if __name__ == '__main__':  # qiebixuefu
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

    graph_idx = graph_Hemisphere_MMV  # The general graph definition for DEAP is used as an example.
    idx = []
    num_chan_local_graph = []
    for i in range(len(graph_idx)):
        num_chan_local_graph.append(len(graph_idx[i]))
        for chan in graph_idx[i]:
            idx.append(original_order.index(chan))

    eeg = torch.randn(15, 62, 5)
    eeg = eeg[:, idx, :]
    eeg = eeg.to(device)
    model = SIFDGCN(channel_num=62, feature_num=5, out_channel=32, idx_graph=num_chan_local_graph)
    model = model.to(device)
    pred = model(eeg)
    print("Input shape: {}".format(eeg.shape))
    print("output shape: {}".format(pred.shape))