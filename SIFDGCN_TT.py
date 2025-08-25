import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import copy
import math
from SIFDGCN import SIFDGCN
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Linear_Embedding(nn.Module):
    def __init__(self, feature_size, d_model=64):
        super(Linear_Embedding, self).__init__()
        self.linear1_0 = nn.Linear(feature_size, int(d_model / 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # default dropout
        self.linear1_1 = nn.Linear(int(d_model / 2), d_model)

    def forward(self, x):
        x = self.linear1_0(x)
        x = self.relu(x)
        out = self.dropout(x)
        out = self.linear1_1(out)
        return out


class PositionalEncoder_1dsincos(nn.Module):
    def __init__(self, d_model, max_seq_len=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


class PositionalEncoder_2dsincos(nn.Module):
    def __init__(self, h=6, w=9, eletrode_num=17, d_model=64, dropout=0.1, temperature: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y = y - 2
        x = x - 4
        assert (d_model % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(d_model // 4) / (d_model // 4 - 1)
        omega = 1.0 / (temperature ** omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        # pe_tmp = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

        pe_tmp = torch.zeros(len(y), d_model)
        for i in range(len(omega)):
            # 将 A, B, C, D 的第 i 列数据放入 result 对应的 4 列区域
            pe_tmp[:, 4 * i:4 * (i + 1)] = torch.stack([x.sin()[:, i], x.cos()[:, i], y.sin()[:, i], y.cos()[:, i]],
                                                       dim=1)

        pe = torch.zeros(eletrode_num, d_model)
        # order of 17 channels
        # ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2', 'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2']
        order = [0, 8, 9, 17, 18, 26, 21, 23, 30, 31, 32, 39, 40, 41, 48, 49, 50]
        for i in range(eletrode_num):
            pe[i, :] = pe_tmp[order[i], :]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        pe = Variable(self.pe, requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class Learned_PE(nn.Module):
    def __init__(self, Channel_num, d_model, dropout=0.1):
        self.Learned_PE = nn.Parameter(torch.rand(1, Channel_num, d_model))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.Learned_PE
        return self.dropout(x)


class Aten_Encoder(nn.Module):
    def __init__(self, Channel_num, d_model, heads, dropout, N):
        super().__init__()
        self.N = N
        self.src_mask = None
        self.layers = get_clones(EncoderLayer(Channel_num, d_model, heads, dropout), N)

    def forward(self, src):
        x = src
        attn = []
        for i in range(self.N):
            x, atten = self.layers[i](x, src_mask=self.src_mask)
            attn.append(atten)
        attn_tensor = torch.stack(attn)
        attn_tensor_transpose = torch.transpose(attn_tensor, 0, 1)
        return x, attn_tensor_transpose


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class EncoderLayer_Spatial(nn.Module):
    def __init__(self, eletrode_num, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        # self.norm_1 = nn.BatchNorm1d(d_model)
        # self.norm_2 = nn.BatchNorm1d(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    # def forward(self, x, mask):
    #     x2 = self.norm_1(x)
    #     x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
    #     x2 = self.norm_2(x)
    #     x = x + self.dropout_2(self.ff(x2))
    #     return x
    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x, atten = self.attn(q=x, k=x, v=x, mask=src_mask)
        # 2. add and norm
        x = self.dropout_1(x)
        x = self.norm_1(x + _x)
        # x = self.norm_1((x+_x).transpose(1,2))
        # x = x.transpose(1,2)
        # 3. positionwise feed forward network
        _x = x
        x = self.ff(x)
        # 4. add and norm
        x = self.dropout_2(x)
        x = self.norm_2(x + _x)
        # x = self.norm_2((x + _x).transpose(1, 2))
        # x = x.transpose(1, 2)
        return x, atten


class EncoderLayer(nn.Module):
    def __init__(self, Channel_num, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Layer_Norm(d_model)
        self.norm_2 = Layer_Norm(d_model)
        # self.norm_1 = nn.BatchNorm1d(Channel_num)
        # self.norm_2 = nn.BatchNorm1d(Channel_num)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    # def forward(self, x, mask):
    #     x2 = self.norm_1(x)
    #     x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
    #     x2 = self.norm_2(x)
    #     x = x + self.dropout_2(self.ff(x2))
    #     return x
    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x, atten = self.attn(q=x, k=x, v=x, mask=src_mask)
        # 2. add and norm
        x = self.dropout_1(x)
        x = self.norm_1(x + _x)
        # 3. positionwise feed forward network
        _x = x
        x = self.ff(x)
        # 4. add and norm
        x = self.dropout_2(x)
        x = self.norm_2(x + _x)
        return x, atten


class Layer_Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    # def forward(self, x):
    #     norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
    #            / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
    #     return norm
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.alpha * out + self.bias
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # self.in_proj_weight_q = nn.Parameter(torch.Tensor(d_model, d_model))
        # self.in_proj_weight_k = nn.Parameter(torch.Tensor(d_model, d_model))
        # self.in_proj_weight_v = nn.Parameter(torch.Tensor(d_model, d_model))

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.reset_parameters()

    # def init_params(self):
    #     init.xavier_uniform_(self.linear.weight)
    def reset_parameters(self):
        # init.kaiming_uniform_(self.in_proj_weight_q, a=math.sqrt(5))
        # init.kaiming_uniform_(self.in_proj_weight_k, a=math.sqrt(5))
        # init.kaiming_uniform_(self.in_proj_weight_v, a=math.sqrt(5))

        init.xavier_uniform_(self.q_linear.weight)
        init.xavier_uniform_(self.k_linear.weight)
        init.xavier_uniform_(self.v_linear.weight)
        init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # q = torch.matmul(q, self.in_proj_weight_q).view(bs, -1, self.h, self.d_k)
        # k = torch.matmul(k, self.in_proj_weight_k).view(bs, -1, self.h, self.d_k)
        # v = torch.matmul(v, self.in_proj_weight_v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores, atten = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output, atten


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    # scores denotes the attention scores
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output, scores


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.2):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class Tem_extract(nn.Module):  # Temporal feature extraction
    def __init__(self, num_classes=1, segment_num=16, d_model=64, hidden_size=32, heads=4):
        super(Tem_extract, self).__init__()
        # Tim-GE-LSTM module
        self.position_enco1 = PositionalEncoder_1dsincos(d_model=d_model, max_seq_len=100, dropout=0.1)
        self.aten_enco1 = Aten_Encoder(Channel_num=segment_num, d_model=d_model, heads=heads, dropout=0.1, N=1)
        self.aten_enco2 = Aten_Encoder(Channel_num=segment_num, d_model=hidden_size, heads=heads, dropout=0.1, N=1)  # N=4 --> N=2

        # Classification module
        self.linear2_1 = nn.Linear(hidden_size * segment_num, 120)
        self.dropout = nn.Dropout(0.6)  # default dropout
        self.linear2_2 = nn.Linear(120, num_classes)

    def forward(self, out):  # x.shaoe = [batch, 16, 64]1
        # out = self.norm(out)
        out = self.position_enco1(out)
        out, _ = self.aten_enco1(out)
        out, _ = self.aten_enco2(out)

        # flatten                       # [batch, 16*32]
        out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])

        # Classification Output
        out = self.linear2_1(out)
        out = self.dropout(out)
        out = self.linear2_2(out)
        return out

class SIFDGCN_TT(nn.Module):
    def __init__(self, num_chan_local_graph, eletrode_num, d_model, segment_num, heads, Fre_num=5):
        super(SIFDGCN_TT, self).__init__()
        self.segment_num = segment_num
        self.linear_enco = Linear_Embedding(Fre_num, d_model=d_model)
        self.SIFDGCN = SIFDGCN(channel_num=eletrode_num, feature_num=d_model, out_channel=d_model, idx_graph=num_chan_local_graph)
        self.Tim = Tem_extract(num_classes=1, segment_num=segment_num, d_model=32, hidden_size=32, heads=heads)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = []
        for i in range(self.segment_num):
            xi = torch.squeeze(x[:, i, :, :], 1)  # Extract the feature at each time step
            xi = self.linear_enco(xi)
            xi = self.SIFDGCN(xi)  # Apply Spa_Fre extraction
            xi = xi.view(xi.shape[0], 1, -1)
            out.append(xi)  # Append to the list

        # Concatenate all the outputs from time steps
        out = torch.cat(out, dim=1)

        # Apply the temporal extraction layer
        out = self.Tim(out)
        #
        return out


if __name__ == '__main__':
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
            idx.append(original_order.index(chan))  # 获取将各个电极分组抽象成“特殊节点后”，按顺序电极的序号

    context = 5
    data = torch.rand((32, context, 62, 5))  # (batch_size=1, cnn_channel=1, EEG_channel=32, data_points=512)
    # data = data[:, :, idx, :]  # (batch_size=1, cnn_channel=1, EEG_channel=32, data_points=512)

    data = data.to(DEVICE)
    net = SIFDGCN_TT(num_chan_local_graph, eletrode_num=62, d_model=32, segment_num=context, heads=1, Fre_num=5)
    net = net.to(DEVICE)
    output = net(data)
    print("Input shape     : ", data.shape)
    print("Output shape    : ", output.shape)
