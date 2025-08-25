# %% EEGNet
# # Sources: We adopted the version publicly available on https://github.com/yi-ding-cs/TSception
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class eegNet(nn.Module):
#     def initialBlocks(self, dropoutP, *args, **kwargs):
#         block1 = nn.Sequential(
#                 nn.Conv2d(1, self.F1, (1, self.C1),
#                           padding=(0, self.C1 // 2), bias=False),
#                 nn.BatchNorm2d(self.F1),
#                 Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
#                                      padding=0, bias=False, max_norm=1,
#                                      groups=self.F1),
#                 nn.BatchNorm2d(self.F1 * self.D),
#                 nn.ELU(),
#                 nn.AvgPool2d((1, 4), stride=4),
#                 nn.Dropout(p=dropoutP))
#         block2 = nn.Sequential(
#                 nn.Conv2d(self.F1 * self.D, self.F1 * self.D,  (1, 22),
#                                      padding=(0, 22//2), bias=False,
#                                      groups=self.F1 * self.D),
#                 nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
#                           stride=1, bias=False, padding=0),
#                 nn.BatchNorm2d(self.F2),
#                 nn.ELU(),
#                 nn.AvgPool2d((1, 8), stride=8),
#                 nn.Dropout(p=dropoutP)
#                 )
#         return nn.Sequential(block1, block2)
#
#     def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
#         return nn.Sequential(
#                 nn.Conv2d(inF, outF, kernalSize, *args, **kwargs))
#
#     def calculateOutSize(self, model, nChan, nTime):
#         '''
#         Calculate the output based on input size.
#         model is from nn.Module and inputSize is a array.
#         '''
#         data = torch.rand(1, 1, nChan, nTime)
#         model.eval()
#         out = model(data).shape
#         return out[2:]
#
#     def __init__(self, nChan, nTime, nClass=2,
#                  dropoutP=0.25, F1=8, D=2,
#                  C1=64, *args, **kwargs):
#         super(eegNet, self).__init__()
#         self.F2 = D*F1
#         self.F1 = F1
#         self.D = D
#         self.nTime = nTime
#         self.nClass = nClass
#         self.nChan = nChan
#         self.C1 = C1
#
#         self.firstBlocks = self.initialBlocks(dropoutP)
#         self.fSize = self.calculateOutSize(self.firstBlocks, nChan, nTime)
#         self.lastLayer = self.lastBlock(self.F2, nClass, (1, self.fSize[1]))
#
#     def forward(self, x):
#         x = self.firstBlocks(x)
#         x = self.lastLayer(x)
#         x = torch.squeeze(x, 3)
#         x = torch.squeeze(x, 2)
#         return x
#
#
# class Conv2dWithConstraint(nn.Conv2d):
#     def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
#         self.max_norm = max_norm
#         self.doWeightNorm = doWeightNorm
#         super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
#
#     def forward(self, x):
#         if self.doWeightNorm:
#             self.weight.data = torch.renorm(
#                 self.weight.data, p=2, dim=0, maxnorm=self.max_norm
#             )
#         return super(Conv2dWithConstraint, self).forward(x)
#
#
# class LinearWithConstraint(nn.Linear):
#     def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
#         self.max_norm = max_norm
#         self.doWeightNorm = doWeightNorm
#         super(LinearWithConstraint, self).__init__(*args, **kwargs)
#
#     def forward(self, x):
#         if self.doWeightNorm:
#             self.weight.data = torch.renorm(
#                 self.weight.data, p=2, dim=0, maxnorm=self.max_norm
#             )
#         return super(LinearWithConstraint, self).forward(x)
#
# if __name__ == '__main__':
#     data = torch.randn(64, 1, 62, 800)  # (batch_size=1, cnn_channel=1, EEG_channel=32, data_points=512)
#
#     EEG_net = eegNet(62,800, nClass=1)
#
#     preds = EEG_net(data)
#     print("The shape of input:", data.shape)
#     print("The shape of output:", preds.shape)


# %% EEG-Conformer
# # Sources: https://github.com/eeyhsong/EEG-Conformer
# import torch
# import torch.nn as nn
# from torch import Tensor
# import os
# from einops import rearrange, reduce, repeat
# from einops.layers.torch import Rearrange, Reduce
# import torch.nn.functional as F
#
# class PatchEmbedding(nn.Module):
#     def __init__(self, emb_size=40):
#         super().__init__()
#
#         self.eegnet = nn.Sequential(
#             nn.Conv2d(1, 8, (1, 125), (1, 1)),
#             nn.BatchNorm2d(8),
#             nn.Conv2d(8, 16, (22, 1), (1, 1)),
#             nn.BatchNorm2d(16),
#             nn.ELU(),
#             nn.AvgPool2d((1, 4), (1, 4)),
#             nn.Dropout(0.5),
#             nn.Conv2d(16, 16, (1, 16), (1, 1)),
#             nn.BatchNorm2d(16),
#             nn.ELU(),
#             nn.AvgPool2d((1, 8), (1, 8)),
#             nn.Dropout2d(0.5)
#         )
#
#         self.shallownet = nn.Sequential(
#             nn.Conv2d(1, 40, (1, 25), (1, 1)),
#             nn.Conv2d(40, 40, (62, 1), (1, 1)),  # 17-->17 channel; 62 --> 62channels
#             nn.BatchNorm2d(40),
#             nn.ELU(),
#             nn.AvgPool2d((1, 75), (1, 15)),
#             nn.Dropout(0.5),
#         )
#
#         self.projection = nn.Sequential(
#             nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # 5 is better than 1
#             Rearrange('b e (h) (w) -> b (h w) e'),
#         )
#
#     def forward(self, x: Tensor) -> Tensor:
#         b, _, _, _ = x.shape
#
#         for i in range(len(self.shallownet)):
#             x = self.shallownet[i](x)
#         # x = self.shallownet(x)  #
#         x = self.projection(x)
#         return x
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, emb_size, num_heads, dropout):
#         super().__init__()
#         self.emb_size = emb_size
#         self.num_heads = num_heads
#         self.keys = nn.Linear(emb_size, emb_size)
#         self.queries = nn.Linear(emb_size, emb_size)
#         self.values = nn.Linear(emb_size, emb_size)
#         self.att_drop = nn.Dropout(dropout)
#         self.projection = nn.Linear(emb_size, emb_size)
#
#     def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
#         queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
#         keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
#         values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
#         energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
#         if mask is not None:
#             fill_value = torch.finfo(torch.float32).min
#             energy.mask_fill(~mask, fill_value)
#
#         scaling = self.emb_size ** (1 / 2)
#         att = F.softmax(energy / scaling, dim=-1)
#         att = self.att_drop(att)
#         out = torch.einsum('bhal, bhlv -> bhav ', att, values)
#         out = rearrange(out, "b h n d -> b n (h d)")
#         out = self.projection(out)
#         return out
#
#
# class ResidualAdd(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         res = x
#         x = self.fn(x, **kwargs)
#         x += res
#         return x
#
#
# class FeedForwardBlock(nn.Sequential):
#     def __init__(self, emb_size, expansion, drop_p):
#         super().__init__(
#             nn.Linear(emb_size, expansion * emb_size),
#             nn.GELU(),
#             nn.Dropout(drop_p),
#             nn.Linear(expansion * emb_size, emb_size),
#         )
#
#
# class GELU(nn.Module):
#     def forward(self, input: Tensor) -> Tensor:
#         return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
#
#
# class TransformerEncoderBlock(nn.Sequential):
#     def __init__(self,
#                  emb_size,
#                  num_heads=5,
#                  drop_p=0.5,
#                  forward_expansion=4,
#                  forward_drop_p=0.5):
#         super().__init__(
#             ResidualAdd(nn.Sequential(
#                 nn.LayerNorm(emb_size),
#                 # nn.BatchNorm1d(7),
#                 MultiHeadAttention(emb_size, num_heads, drop_p),
#                 nn.Dropout(drop_p)
#             )),
#             ResidualAdd(nn.Sequential(
#                 nn.LayerNorm(emb_size),
#                 # nn.BatchNorm1d(7),
#                 FeedForwardBlock(
#                     emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
#                 nn.Dropout(drop_p)
#             )
#             ))
#
#
# class TransformerEncoder(nn.Sequential):
#     def __init__(self, depth, emb_size):
#         super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])
#
#
# class ClassificationHead(nn.Sequential):
#     def __init__(self, emb_size, n_classes):
#         super().__init__()
#         self.cov = nn.Sequential(
#             nn.Conv1d(190, 1, 1, 1),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.5)
#         )
#         self.clshead = nn.Sequential(
#             Reduce('b n e -> b e', reduction='mean'),
#             nn.LayerNorm(emb_size),
#             nn.Linear(emb_size, n_classes)
#         )
#         self.clshead_fc = nn.Sequential(
#             Reduce('b n e -> b e', reduction='mean'),
#             nn.LayerNorm(emb_size),
#             nn.Linear(emb_size, 32),
#             nn.ELU(),
#             nn.Dropout(0.5),
#             nn.Linear(32, n_classes)
#         )
#         self.fc = nn.Sequential(
#             # nn.Linear(280, 32),
#             nn.Linear(1880, 32),  # 1880 <==> 800; 4040 <==> 1600
#             nn.ELU(),
#             nn.Dropout(0.3),
#             nn.Linear(32, 1)
#         )
#
#     def forward(self, x):
#         x = x.contiguous().view(x.size(0), -1)
#         out = self.fc(x)
#
#         # return x, out
#
#         return out
#
#
# # ! Rethink the use of Transformer for EEG signal
# class ViT(nn.Sequential):
#     def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
#         super().__init__(
#
#             PatchEmbedding(emb_size),
#             TransformerEncoder(depth, emb_size),
#             ClassificationHead(emb_size, n_classes)
#         )
#
#
# if __name__ == '__main__':
#     data = torch.randn(64, 1, 62, 800)  # (batch_size=1, cnn_channel=1, EEG_channel=32, data_points=512)
#     Comformer = ViT()
#     preds = Comformer(data)
#     print("The shape of input:", data.shape)
#     print("The shape of output:", preds.shape)


# %% TSception
# # Sources: https://github.com/yi-ding-cs/TSception
# import torch
# import torch.nn as nn
# def generate_TS_channel_order(original_order: list):
#     """
#     This function will generate the channel order for TSception
#     Parameters
#     ----------
#     original_order: list of the channel names
#
#     Returns
#     -------
#     TS: list of channel names which is for TSception
#     """
#     chan_name, chan_num, chan_final = [], [], []
#     for channel in original_order:
#         chan_name_len = len(channel)
#         k = 0
#         for s in [*channel[:]]:  # 判断名字中有无数字，有数字+1
#             if s.isdigit():
#                k += 1
#         if k != 0:  # 有数字，不在中心线，则添加入列表
#             chan_name.append(channel[:chan_name_len-k])  # 保留通道前名称
#             chan_num.append(int(channel[chan_name_len-k:]))  # 保留通道编码
#             chan_final.append(channel)  # 保留完整通道编码
#     chan_pair = []
#     for ch, id in enumerate(chan_num):  # ch为索引，id为保存的具体序号
#         if id % 2 == 0:
#             chan_pair.append(chan_name[ch] + str(id-1))
#         else:
#             chan_pair.append(chan_name[ch] + str(id+1))  # 奇偶数换了个位置
#     chan_no_duplicate = []
#     # [chan_no_duplicate.extend([f, chan_pair[i]]) for i, f in enumerate(chan_final) if f not in chan_no_duplicate]
#     for i, f in enumerate(chan_final):
#         if f not in chan_no_duplicate:
#             chan_no_duplicate.extend([f, chan_pair[i]])  # 把同样名字的放在一起
#
#     chan_output = chan_no_duplicate[0::2] + chan_no_duplicate[1::2]  # 由于上一步则可将电极全部分开成两边（单数在左半脑，双数在右半脑）
#     return chan_output
#
# class TSception(nn.Module):
#     def conv_block(self, in_chan, out_chan, kernel, step, pool):
#         return nn.Sequential(
#             nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
#                       kernel_size=kernel, stride=step),
#             nn.LeakyReLU(),
#             nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))
#
#     def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
#         # input_size: 1 x EEG channel x datapoint
#         super(TSception, self).__init__()
#         self.inception_window = [0.5, 0.25, 0.125]
#         self.pool = 8
#         # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
#         # achieve the 1d convolution operation
#         self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
#         self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
#         self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)
#
#         self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool*0.25))
#         self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
#                                          int(self.pool*0.25))
#         self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
#         self.BN_t = nn.BatchNorm2d(num_T)
#         self.BN_s = nn.BatchNorm2d(num_S)
#         self.BN_fusion = nn.BatchNorm2d(num_S)
#
#         self.fc = nn.Sequential(
#             nn.Linear(num_S, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden, num_classes)
#         )
#
#     def forward(self, x):
#         y = self.Tception1(x)
#         out = y
#         y = self.Tception2(x)
#         out = torch.cat((out, y), dim=-1)
#         y = self.Tception3(x)
#         out = torch.cat((out, y), dim=-1)
#         out = self.BN_t(out)
#         z = self.Sception1(out)
#         out_ = z
#         z = self.Sception2(out)
#         out_ = torch.cat((out_, z), dim=2)
#         out = self.BN_s(out_)
#         out = self.fusion_layer(out)
#         out = self.BN_fusion(out)
#         out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
#         out = self.fc(out)
#         return out
#
# if __name__ == '__main__':
#     original_order = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
#                       'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
#                       'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
#                       'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
#                       'O2', 'CB2']
#
#     TS_order = generate_TS_channel_order(
#         original_order)  # generate proper channel orders for the asymmetric spatial layer in TSception
#
#     data = torch.randn(64, 1, 62, 800)  # (batch_size=1, cnn_channel=1, EEG_channel=32, data_points=512)
#     idx = []
#     for chan in TS_order:
#         idx.append(original_order.index(chan))  # 返回出现chan时，原始数据中的索引序号
#     data = data[:, :, idx, :]  # (batch_size=1, cnn_channel=1, EEG_channel=28, data_points=512) Some channels are not selected, hence EEG channel becomes 28.
#     # 数据转换为TS_order中的数据顺序
#
#     TS = TSception(
#         num_classes=1,
#         input_size=(1, len(idx), 800),
#         sampling_rate=200,
#         num_T=15,  # num_T controls the number of temporal filters
#         num_S=15,
#         # num_S controls the size of hidden embedding due to the global average pooling. Please increase it if you need a larger model capacity, e.g., subject-independent case
#         hidden=32,
#         dropout_rate=0.5
#     )
#
#     preds = TS(data)
#     print("The shape of input:", data.shape)
#     print("The shape of output:", preds.shape)

# %% primary LGG
# # Sources: https://github.com/yi-ding-cs/LGG
# import torch
# import math
# import os
# import mat73
# import torch.nn as nn
# import torch.nn.functional as F
# # from layers import GraphConvolution
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# class GraphConvolution(nn.Module):
#     """
#     simple GCN layer
#     """
#
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
#         if bias:
#             self.bias = nn.Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
#         else:
#             self.register_parameter('bias', None)
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, x, adj):
#         output = torch.matmul(x, self.weight) - self.bias  # 这里的weight权重，设置的隐藏输出维度为32
#         output = F.relu(torch.matmul(adj, output))  # 此处的图卷积，仅仅与可学习的参数进行了相乘加上一个偏执，然后与邻接矩阵矩阵相乘再激活
#         return output
#
# class PowerLayer(nn.Module):
#     '''
#     The power layer: calculates the log-transformed power of the data
#     '''
#     def __init__(self, dim, length, step):
#         super(PowerLayer, self).__init__()
#         self.dim = dim
#         self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))
#
#     def forward(self, x):
#         return torch.log(self.pooling(x.pow(2)))
#
#
# class LGGNet(nn.Module):
#     def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
#         return nn.Sequential(
#             nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
#             PowerLayer(dim=-1, length=pool, step=int(pool_step_rate*pool))
#         )
#
#     def __init__(self, num_classes, input_size, sampling_rate, num_T,
#                  out_graph, dropout_rate, pool, pool_step_rate, idx_graph):
#         # input_size: EEG frequency x channel x datapoint
#         super(LGGNet, self).__init__()
#         self.idx = idx_graph
#         self.window = [0.5, 0.25, 0.125]
#         self.pool = pool
#         self.channel = input_size[1]
#         self.brain_area = len(self.idx)
#
#         # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
#         # achieve the 1d convolution operation
#         self.Tception1 = self.temporal_learner(input_size[0], num_T,
#                                                (1, int(self.window[0] * sampling_rate)),
#                                                self.pool, pool_step_rate)
#         self.Tception2 = self.temporal_learner(input_size[0], num_T,
#                                                (1, int(self.window[1] * sampling_rate)),
#                                                self.pool, pool_step_rate)
#         self.Tception3 = self.temporal_learner(input_size[0], num_T,
#                                                (1, int(self.window[2] * sampling_rate)),
#                                                self.pool, pool_step_rate)
#         self.BN_t = nn.BatchNorm2d(num_T)
#         self.BN_t_ = nn.BatchNorm2d(num_T)
#         self.OneXOneConv = nn.Sequential(
#             nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
#             nn.LeakyReLU(),
#             nn.AvgPool2d((1, 2)))  # 融合64个核 (Batch_size, Num_kernel, Channel, Timpoints)
#         # diag(W) to assign a weight to each local areas
#         size = self.get_size_temporal(input_size)
#         self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
#                                                 requires_grad=True)
#         nn.init.xavier_uniform_(self.local_filter_weight)
#         self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
#                                               requires_grad=True)
#
#         # aggregate function
#         self.aggregate = Aggregator(self.idx)
#
#         # trainable adj weight for global network
#         self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
#         nn.init.xavier_uniform_(self.global_adj)
#         # to be used after local graph embedding
#         self.bn = nn.BatchNorm1d(self.brain_area)
#         self.bn_ = nn.BatchNorm1d(self.brain_area)
#         # learn the global network of networks
#         self.GCN = GraphConvolution(size[-1], out_graph)
#
#         self.fc = nn.Sequential(
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(int(self.brain_area * out_graph), num_classes))
#
#     def forward(self, x):
#         y = self.Tception1(x)
#         out = y
#         y = self.Tception2(x)
#         out = torch.cat((out, y), dim=-1)
#         y = self.Tception3(x)
#         out = torch.cat((out, y), dim=-1)
#         out = self.BN_t(out)
#         out = self.OneXOneConv(out)  # 融合64个核 (Batch_size, Num_kernel, Channel, Timpoints)
#         out = self.BN_t_(out)
#         out = out.permute(0, 2, 1, 3)  # (Batch_size, Channel, Num_kernel, Timpoints) 准备对每个Channel进行处理
#         out = torch.reshape(out, (out.size(0), out.size(1), -1))  # 把每个电极的特征拉成了向量
#
#         # 局部提取图信息：实际就是可学习的矩阵相乘+偏置；然后将每个电极信息取平均，作为大节点信息
#         out = self.local_filter_fun(out, self.local_filter_weight)
#         # self.local_filter_weight为和Out形状完全一致的可学习参数矩阵
#         # 其中self.local_filter_fun()操作，让结果类似于Attention了一下，与一个参数矩阵逐点相乘，再加上一个可学习的偏执矩阵
#         out = self.aggregate.forward(out)  # 将每个节点取平均，获得了新的节点信号
#
#         # 然后求新的图的相似矩阵（矩阵自相乘）；再次用可学习的参数矩阵WA+B，获得新的邻接矩阵；再+eye矩阵，增加自关联程度；
#         adj = self.get_adj(out)
#         out = self.bn(out)  # 对新节点，再次归一化
#
#         # 新节点，直接WX-B；再左乘邻接矩阵并激活；这就是图卷积？用可学习的参数类比于全连接
#         out = self.GCN(out, adj)
#         out = self.bn_(out)
#
#         out = out.view(out.size()[0], -1)
#         out = self.fc(out)
#         return out
#
#     def get_size_temporal(self, input_size):
#         # input_size: frequency x channel x data point
#         data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
#         z = self.Tception1(data)
#         out = z
#         z = self.Tception2(data)
#         out = torch.cat((out, z), dim=-1)
#         z = self.Tception3(data)
#         out = torch.cat((out, z), dim=-1)
#         out = self.BN_t(out)
#         out = self.OneXOneConv(out)
#         out = self.BN_t_(out)
#         out = out.permute(0, 2, 1, 3)
#         out = torch.reshape(out, (out.size(0), out.size(1), -1))
#         size = out.size()
#         return size
#
#     def local_filter_fun(self, x, w):
#         w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
#         x = F.relu(torch.mul(x, w) - self.local_filter_bias)  # 对逐点相乘+偏置
#         return x
#
#     def get_adj(self, x, self_loop=True):
#         # x: b, node, feature
#         adj = self.self_similarity(x)   # b, n, n
#         # 通过矩阵自相乘，获得批量矩阵，类比于Transformer的Attention机制
#         num_nodes = adj.shape[-1]  # 获取抽象电极数
#         adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
#         # 按照论文，是通过可学习的矩阵，进一步获取全局关联；取转置后相加，是考虑对称性，也考虑电极之间的相互关联性
#
#         if self_loop:
#             adj = adj + torch.eye(num_nodes).to(DEVICE)
#             # 按照论文是添加自循环
#         rowsum = torch.sum(adj, dim=-1)  # 求得每个电极的关联值总和
#         mask = torch.zeros_like(rowsum)
#         mask[rowsum == 0] = 1  # 保证邻接矩阵归一化时，度矩阵根号2分之1不为0
#         rowsum += mask
#         d_inv_sqrt = torch.pow(rowsum, -0.5)  # torch.pow是求幂的公式，根号分之一
#         d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # 将向量转换为对角矩阵，此处即“度”矩阵
#         adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)  # 对图的标准化，左乘右乘度矩阵
#         return adj
#
#     def self_similarity(self, x):
#         # x: b, node, feature
#         x_ = x.permute(0, 2, 1)
#         s = torch.bmm(x, x_)
#         return s  # 通过矩阵自相乘，获得批量矩阵
#
#
# class Aggregator():
#
#     def __init__(self, idx_area):
#         # chan_in_area: a list of the number of channels within each area
#         self.chan_in_area = idx_area
#         self.idx = self.get_idx(idx_area)
#         self.area = len(idx_area)
#
#     def forward(self, x):
#         # x: batch x channel x data
#         data = []
#         for i, area in enumerate(range(self.area)):
#             if i < self.area - 1:
#                 data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
#                 # 将每个电极区域的EEG信号进行了取平均操作,从而形成新的节点信号
#             else:
#                 data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
#         return torch.stack(data, dim=1)
#
#     def get_idx(self, chan_in_area):
#         idx = [0] + chan_in_area
#         idx_ = [0]
#         for i in idx:
#             idx_.append(idx_[-1] + i)
#         return idx_[1:]
#
#     def aggr_fun(self, x, dim):
#         # return torch.max(x, dim=dim).values
#         return torch.mean(x, dim=dim)
#
# if __name__ == '__main__':
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     data_path = 'D:/MMV dataset/'
#     subname = 1
#     session_num = 0
#
#     # load trial data
#     avearge_trial_data_path = os.path.join(data_path, 'Trial Data', 'subject-%02d' % (subname + 1),
#                                            'session-%02d' % (session_num + 1),
#                                            'sub-%02d_sess-%02d_blk-02_eeg.mat' % (subname + 1, session_num + 1))
#     dataset_trial_data = mat73.loadmat(avearge_trial_data_path)
#     original_order = dataset_trial_data['Data']['channels_notation']
#
#     original_order = [item[0] for item in original_order]
#     print(original_order)
#
#     original_order = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3',
#      'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5',
#      'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
#      'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
#
#     original_order2 = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
#                        'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
#                        'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
#                        'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
#                        'O2', 'CB2']
#
#
#
#
#     graph_general_MMV = [['FP1', 'FPZ', 'FP2'], ['AF3', 'AF4'], ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
#                          ['FT7', 'T7', 'TP7'], ['FT8', 'T8', 'TP8'], ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6'],
#                          ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6'], ['CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'],
#                          ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
#                          ['PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8'],
#                          ['O1', 'OZ', 'O2'], ['CB1', 'CB2']]
#
#     graph_idx = graph_general_MMV  # The general graph definition for DEAP is used as an example.
#     idx = []
#     num_chan_local_graph = []
#     for i in range(len(graph_idx)):
#         num_chan_local_graph.append(len(graph_idx[i]))
#         for chan in graph_idx[i]:
#             idx.append(original_order.index(chan))  # 获取将各个电极分组抽象成“特殊节点后”，按顺序电极的序号
#
#     data = torch.randn(32, 1, 62, 800)  # (batch_size=1, cnn_channel=1, EEG_channel=32, data_points=512)
#     data = data[:, :, idx, :]  # (batch_size=1, cnn_channel=1, EEG_channel=32, data_points=512)
#
#     data = data.to(DEVICE)
#     LGG = LGGNet(
#         num_classes=1,
#         input_size=(1, 62, 800),
#         sampling_rate=200,
#         num_T=64,  # num_T controls the number of temporal filters
#         out_graph=32,
#         pool=16,
#         pool_step_rate=0.25,
#         idx_graph=num_chan_local_graph,
#         dropout_rate=0.5
#     )
#     LGG = LGG.to(DEVICE)
#     preds = LGG(data)
#
#     print("Input data shape: {}".format(data.shape))
#     print("Output data shape: {}".format(preds.shape))


# %% LGG adapted for DE feature
# import torch
# import math
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import mat73
#
# # from layers import GraphConvolution
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#
# class GraphConvolution(nn.Module):
#     """
#     simple GCN layer
#     """
#
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
#         if bias:
#             self.bias = nn.Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
#         else:
#             self.register_parameter('bias', None)
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, x, adj):
#         output = torch.matmul(x, self.weight) - self.bias  # 这里的weight权重，设置的隐藏输出维度为32
#         output = F.relu(torch.matmul(adj, output))  # 此处的图卷积，仅仅与可学习的参数进行了相乘加上一个偏执，然后与邻接矩阵矩阵相乘再激活
#         return output
#
#
# class PowerLayer(nn.Module):
#     '''
#     The power layer: calculates the log-transformed power of the data
#     '''
#     def __init__(self, dim, length, step):
#         super(PowerLayer, self).__init__()
#         self.dim = dim
#         self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))
#
#     def forward(self, x):
#         return torch.log(self.pooling(x.pow(2)))
#
#
# class LGG_Graph(nn.Module):
#     def __init__(self, num_classes, input_size, out_graph, dropout_rate, idx_graph):
#         # input_size: EEG frequency x channel x datapoint
#         super(LGG_Graph, self).__init__()
#         self.idx = idx_graph
#         self.channel = input_size[1]
#         self.brain_area = len(self.idx)
#
#         self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, input_size[-1]),
#                                                 requires_grad=True)
#         nn.init.xavier_uniform_(self.local_filter_weight)
#         self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
#                                               requires_grad=True)
#
#         # aggregate function
#         self.aggregate = Aggregator(self.idx)
#
#         # trainable adj weight for global network
#         self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
#         nn.init.xavier_uniform_(self.global_adj)
#         # to be used after local graph embedding
#         self.bn = nn.BatchNorm1d(self.brain_area)
#         self.bn_ = nn.BatchNorm1d(self.brain_area)
#         # learn the global network of networks
#         self.GCN = GraphConvolution(input_size[-1], out_graph)
#
#         self.fc = nn.Sequential(
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(int(self.brain_area * out_graph), num_classes))
#
#     def forward(self, x):
#         x = self.local_filter_fun(x, self.local_filter_weight)
#         # self.local_filter_weight为和Out形状完全一致的可学习参数矩阵
#         # 其中self.local_filter_fun()操作，让结果类似于Attention了一下，与一个参数矩阵逐点相乘，再加上一个可学习的偏置矩阵
#         out = self.aggregate.forward(x)  # 将每个节点取平均，获得了新的节点信号
#         # 局部提取图信息：实际就是可学习的矩阵相乘+偏置；然后将每个电极信息取平均，作为大节点信息
#
#         # 然后求新的图的相似矩阵（矩阵自相乘）；再次用可学习的参数矩阵WA+B，获得新的邻接矩阵；再+eye矩阵，增加自关联程度；
#         adj = self.get_adj(out)
#         out = self.bn(out)  # 对新节点，再次归一化
#
#         # 新节点，直接WX-B；再左乘邻接矩阵并激活；这就是图卷积？用可学习的参数类比于全连接
#         out = self.GCN(out, adj)
#         out = self.bn_(out)
#
#         out = out.view(out.size()[0], -1)
#         out = self.fc(out)
#         # out = out.view(out.shape[0], 1, -1)
#         return out
#
#     def local_filter_fun(self, x, w):
#         w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
#         x = F.relu(torch.mul(x, w) - self.local_filter_bias)  # 对逐点相乘+偏置
#         return x
#
#     def get_adj(self, x, self_loop=True):
#         # x: b, node, feature
#         adj = self.self_similarity(x)   # b, n, n
#         # 通过矩阵自相乘，获得批量矩阵，类比于Transformer的Attention机制
#         num_nodes = adj.shape[-1]  # 获取抽象电极数
#
#
#         adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
#         # 按照论文，是通过可学习的矩阵，进一步获取全局关联；取转置后相加，是考虑对称性，也考虑电极之间的相互关联性
#
#         if self_loop:
#             adj = adj + torch.eye(num_nodes).to(DEVICE)
#             # 按照论文是添加自循环
#         rowsum = torch.sum(adj, dim=-1)  # 求得每个电极的关联值总和
#         mask = torch.zeros_like(rowsum)
#         mask[rowsum == 0] = 1  # 保证邻接矩阵归一化时，度矩阵根号2分之1不为0
#         rowsum += mask
#         d_inv_sqrt = torch.pow(rowsum, -0.5)  # torch.pow是求幂的公式，根号分之一
#         d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # 将向量转换为对角矩阵，此处即“度”矩阵
#         adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)  # 对图的标准化，左乘右乘度矩阵
#         return adj
#
#     def self_similarity(self, x):
#         # x: b, node, feature
#         x_ = x.permute(0, 2, 1)
#         s = torch.bmm(x, x_)
#         return s  # 通过矩阵自相乘，获得批量矩阵
#
#
# class Aggregator():
#
#     def __init__(self, idx_area):
#         # chan_in_area: a list of the number of channels within each area
#         self.chan_in_area = idx_area
#         self.idx = self.get_idx(idx_area)
#         self.area = len(idx_area)
#
#     def forward(self, x):
#         # x: batch x channel x data
#         data = []
#         for i, area in enumerate(range(self.area)):
#             if i < self.area - 1:
#                 data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
#                 # 将每个电极区域的EEG信号进行了取平均操作,从而形成新的节点信号
#             else:
#                 data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
#         return torch.stack(data, dim=1)
#
#     def get_idx(self, chan_in_area):
#         idx = [0] + chan_in_area
#         idx_ = [0]
#         for i in idx:
#             idx_.append(idx_[-1] + i)
#         return idx_[1:]
#
#     def aggr_fun(self, x, dim):
#         # return torch.max(x, dim=dim).values
#         return torch.mean(x, dim=dim)
#
#
# if __name__ == '__main__':
#     # data_path = 'D:/MMV dataset/'
#     # subname = 0
#     # session_num = 0
#     #
#     # # load trial data
#     # avearge_trial_data_path = os.path.join(data_path, 'Trial Data', 'subject-%02d' % (subname + 1),
#     #                                        'session-%02d' % (session_num + 1),
#     #                                        'sub-%02d_sess-%02d_blk-02_eeg.mat' % (subname + 1, session_num + 1))
#     # dataset_trial_data = mat73.loadmat(avearge_trial_data_path)
#     # original_order = dataset_trial_data['Data']['channels_notation']
#     #
#     # original_order = [item[0] for item in original_order]
#     # print(original_order)
#     original_order = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
#                       'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
#                       'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
#                       'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
#                       'O2', 'CB2']
#
#
#     graph_general_MMV = [['FP1', 'FPZ', 'FP2'], ['AF3', 'AF4'], ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
#                          ['FT7', 'T7', 'TP7'], ['FT8', 'T8', 'TP8'], ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6'],
#                          ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6'], ['CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'],
#                          ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
#                          ['PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8'],
#                          ['O1', 'OZ', 'O2'], ['CB1', 'CB2']]
#
#     graph_idx = graph_general_MMV  # The general graph definition for DEAP is used as an example.
#     idx = []
#     num_chan_local_graph = []
#     for i in range(len(graph_idx)):
#         num_chan_local_graph.append(len(graph_idx[i]))
#         for chan in graph_idx[i]:
#             idx.append(original_order.index(chan))  # 获取将各个电极分组抽象成“特殊节点后”，按顺序电极的序号
#
#     data = torch.randn(32, 62, 5)  # (batch_size=1, cnn_channel=1, EEG_channel=32, data_points=512)
#     data = data[:, idx, :]  # (batch_size=1, cnn_channel=1, EEG_channel=32, data_points=512)
#
#     data = data.to(DEVICE)
#     LGG = LGG_Graph(
#         num_classes=1,
#         input_size=(1, 62, 5),
#         out_graph=16,
#         idx_graph=num_chan_local_graph,
#         dropout_rate=0.5
#     )
#     LGG = LGG.to(DEVICE)
#     preds = LGG(data)
#
#     print("Input data shape: {}".format(data.shape))
#     print("Output data shape: {}".format(preds.shape))


# %% SFT-Net
# # Sources: https://github.com/wangkejie97/SFT-Net
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init
# import numpy as np
#
# # This is two parts of the attention module:
# ## Spatial_Attention in attention module
# class spatialAttention(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
#         self.norm = nn.Sigmoid()
#
#     def forward(self, U):
#         q = self.Conv1x1(U)
#         spaAtten = q
#         spaAtten = torch.squeeze(spaAtten, 1)
#         q = self.norm(q)
#         # In addition, return to spaAtten for visualization
#         return U * q, spaAtten
#
#
# ## Frequency Attention in attention module
# class frequencyAttention(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2,
#                                       kernel_size=1, bias=False)
#         self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels,
#                                          kernel_size=1, bias=False)
#         self.norm = nn.Sigmoid()
#
#     def forward(self, U):
#         z = self.avgpool(U)
#         z = self.Conv_Squeeze(z)
#         z = self.Conv_Excitation(z)
#         freqAtten = z
#         freqAtten = torch.squeeze(freqAtten, 3)
#         z = self.norm(z)
#         # In addition, return to freqAtten for visualization
#         return U * z.expand_as(U), freqAtten
#
#
# # Attention module:
# # spatial-frequency attention
# class sfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#
#         self.frequencyAttention = frequencyAttention(in_channels)
#         self.spatialAttention = spatialAttention(in_channels)
#
#     def forward(self, U):
#         U_sse, spaAtten = self.spatialAttention(U)
#         U_cse, freqAtten = self.frequencyAttention(U)
#         # Return new 4D features
#         # and the Frequency Attention and Spatial_Attention
#         return U_cse + U_sse, spaAtten, freqAtten
#
# # depthwise separable convolution(DS Conv):
# # depthwise conv + pointwise conv + bn + relu
# class depthwise_separable_conv(nn.Module):
#     def __init__(self, ch_in, ch_out, kernel_size):
#         super(depthwise_separable_conv, self).__init__()
#         self.ch_in = ch_in
#         self.ch_out = ch_out
#         self.kernal_size = kernel_size
#         self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size, padding=1, groups=ch_in)
#         self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)
#         self.bn = nn.BatchNorm2d(ch_out)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.depth_conv(x)
#         x = self.point_conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
#
#
# # Context module in DSC module
# class Conv3x3BNReLU(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(Conv3x3BNReLU, self).__init__()
#         self.conv3x3 = depthwise_separable_conv(in_channel, out_channel, 3)
#         self.bn = nn.BatchNorm2d(out_channel)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         return self.relu(self.bn(self.conv3x3(x)))
#
#
# class ContextModule(nn.Module):
#     def __init__(self, in_channel):
#         super(ContextModule, self).__init__()
#         self.stem = Conv3x3BNReLU(in_channel, in_channel // 2)
#         self.branch1_conv3x3 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)
#         self.branch2_conv3x3_1 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)
#         self.branch2_conv3x3_2 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)
#
#     def forward(self, x):
#         x = self.stem(x)
#         # branch1
#         x1 = self.branch1_conv3x3(x)
#         # branch2
#         x2 = self.branch2_conv3x3_1(x)
#         x2 = self.branch2_conv3x3_2(x2)
#         # concat
#         return torch.cat([x1, x2], dim=1)
#
#
# # Attention module + DSC module + LSTM module
# class SFT_Net(nn.Module):
#     def __init__(self, Fre_num=5, Seg_num=5, num_classes=1):
#         super(SFT_Net, self).__init__()
#         self.Atten = sfAttention(in_channels=Fre_num)
#         self.bneck = nn.Sequential(
#             #  begin x = [32, 16, 5, 6, 9], in fact x1 = [32, 5, 6, 9]
#             depthwise_separable_conv(Fre_num, 32, 3),
#             depthwise_separable_conv(32, 64, 3),
#             # default dropout
#             nn.Dropout2d(0.3),
#             depthwise_separable_conv(64, 128, 3),
#             # Context Module
#             ContextModule(128),
#             depthwise_separable_conv(128, 64, 3),
#             # default dropout
#             nn.Dropout2d(0.3),
#             depthwise_separable_conv(64, 32, 3),
#             nn.AdaptiveAvgPool2d((2, 2))  # [batch, 32, 2, 2]
#         )
#         self.linear = nn.Linear(32 * 2 * 2, 64)
#         self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=2, batch_first=True)  # [batch, input_size, -]
#         self.linear1 = nn.Linear(32 * Seg_num, 120)  # 8-->16-->Num_segment
#         self.dropout = nn.Dropout(0.4)  # default dropout
#         self.linear2 = nn.Linear(120, num_classes)
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # x1 - x16 [batch, 16, 5, 6, 9]
#         x1 = torch.squeeze(x[:, 0, :, :, :], 1)  # [batch, 5, 6, 9]
#         x2 = torch.squeeze(x[:, 1, :, :, :], 1)
#         x3 = torch.squeeze(x[:, 2, :, :, :], 1)
#         x4 = torch.squeeze(x[:, 3, :, :, :], 1)
#         x5 = torch.squeeze(x[:, 4, :, :, :], 1)
#         # x6 = torch.squeeze(x[:, 5, :, :, :], 1)
#         # x7 = torch.squeeze(x[:, 6, :, :, :], 1)
#         # x8 = torch.squeeze(x[:, 7, :, :, :], 1)
#         # x9 = torch.squeeze(x[:, 8, :, :, :], 1)
#         # x10 = torch.squeeze(x[:, 9, :, :, :], 1)
#         # x11 = torch.squeeze(x[:, 10, :, :, :], 1)
#         # x12 = torch.squeeze(x[:, 11, :, :, :], 1)
#         # x13 = torch.squeeze(x[:, 12, :, :, :], 1)
#         # x14 = torch.squeeze(x[:, 13, :, :, :], 1)
#         # x15 = torch.squeeze(x[:, 14, :, :, :], 1)
#         # x16 = torch.squeeze(x[:, 15, :, :, :], 1)
#         # spaAtten and freqAtten
#         x1, spaAtten1, freqAtten1 = self.Atten(x1)  # [batch, 5, 6, 9], [batch, 6, 9], [batch, 5, 1]
#         x2, spaAtten2, freqAtten2 = self.Atten(x2)
#         x3, spaAtten3, freqAtten3 = self.Atten(x3)
#         x4, spaAtten4, freqAtten4 = self.Atten(x4)
#         x5, spaAtten5, freqAtten5 = self.Atten(x5)
#         # x6, spaAtten6, freqAtten6 = self.Atten(x6)
#         # x7, spaAtten7, freqAtten7 = self.Atten(x7)
#         # x8, spaAtten8, freqAtten8 = self.Atten(x8)
#         # x9, spaAtten9, freqAtten9 = self.Atten(x9)
#         # x10, spaAtten10, freqAtten10 = self.Atten(x10)
#         # x11, spaAtten11, freqAtten11 = self.Atten(x11)
#         # x12, spaAtten12, freqAtten12 = self.Atten(x12)
#         # x13, spaAtten13, freqAtten13 = self.Atten(x13)
#         # x14, spaAtten14, freqAtten14 = self.Atten(x14)
#         # x15, spaAtten15, freqAtten15 = self.Atten(x15)
#         # x16, spaAtten16, freqAtten16 = self.Atten(x16)
#         # attention avg
#         # spaAtten = (spaAtten1 + spaAtten2 + spaAtten3 + spaAtten4 + spaAtten5 + spaAtten6 + spaAtten7 + spaAtten8
#         #             + spaAtten9 + spaAtten10 + spaAtten11 + spaAtten12 + spaAtten13 + spaAtten14 + spaAtten15 + spaAtten16) / 16
#         # freqAtten = (freqAtten1 + freqAtten2 + freqAtten3 + freqAtten4 + freqAtten5 + freqAtten6 + freqAtten7 + freqAtten8
#         #             + freqAtten9 + freqAtten10 + freqAtten11 + freqAtten12 + freqAtten13 + freqAtten14 + freqAtten15 + freqAtten16) / 16
#
#         # spaAtten = (spaAtten1 + spaAtten2 + spaAtten3 + spaAtten4 + spaAtten5 + spaAtten6 + spaAtten7 + spaAtten8) / 8
#         # freqAtten = (freqAtten1 + freqAtten2 + freqAtten3 + freqAtten4 + freqAtten5 + freqAtten6 + freqAtten7 + freqAtten8) / 8
#
#         spaAtten = (spaAtten1 + spaAtten2 + spaAtten3 + spaAtten4 + spaAtten5) / 8
#         freqAtten = (freqAtten1 + freqAtten2 + freqAtten3 + freqAtten4 + freqAtten5) / 8
#
#         # bneck
#         x1 = self.bneck(x1)
#         x2 = self.bneck(x2)
#         x3 = self.bneck(x3)
#         x4 = self.bneck(x4)
#         x5 = self.bneck(x5)
#         # x6 = self.bneck(x6)
#         # x7 = self.bneck(x7)
#         # x8 = self.bneck(x8)
#         # x9 = self.bneck(x9)
#         # x10 = self.bneck(x10)
#         # x11 = self.bneck(x11)
#         # x12 = self.bneck(x12)
#         # x13 = self.bneck(x13)
#         # x14 = self.bneck(x14)
#         # x15 = self.bneck(x15)
#         # x16 = self.bneck(x16)
#
#         x1 = self.linear(x1.view(x1.shape[0], 1, -1))  # [batch, 1, 32*2*2] -> [batch, 1, 64]
#         x2 = self.linear(x2.view(x2.shape[0], 1, -1))
#         x3 = self.linear(x3.view(x3.shape[0], 1, -1))
#         x4 = self.linear(x4.view(x4.shape[0], 1, -1))
#         x5 = self.linear(x5.view(x5.shape[0], 1, -1))
#         # x6 = self.linear(x6.view(x6.shape[0], 1, -1))
#         # x7 = self.linear(x7.view(x7.shape[0], 1, -1))
#         # x8 = self.linear(x8.view(x8.shape[0], 1, -1))
#         # x9 = self.linear(x9.view(x9.shape[0], 1, -1))
#         # x10 = self.linear(x10.view(x10.shape[0], 1, -1))
#         # x11 = self.linear(x11.view(x11.shape[0], 1, -1))
#         # x12 = self.linear(x12.view(x12.shape[0], 1, -1))
#         # x13 = self.linear(x13.view(x13.shape[0], 1, -1))
#         # x14 = self.linear(x14.view(x14.shape[0], 1, -1))
#         # x15 = self.linear(x15.view(x15.shape[0], 1, -1))
#         # x16 = self.linear(x16.view(x16.shape[0], 1, -1))
#
#         # cat 16 * [batch, 1, 32] -> [batch, 16, 32]
#         # out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16), dim=1)
#         # out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)
#         out = torch.cat((x1, x2, x3, x4, x5), dim=1)
#
#         # after LSTM                    [batch, 16, 64]
#         out, (h, c) = self.lstm(out)
#
#         # flatten                       [batch, 16*120]
#         out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])
#
#         # first linear                  [batch, 120]
#         out = self.linear1(out)
#         out = self.dropout(out)
#         # second linear                 [batch, ]
#         out = self.linear2(out)
#         # finnal feature and attention  [batch, 1]
#         # return out, spaAtten, freqAtten
#         return out
#
# def Data_4D_Generation(input, Fre_num=5, Seg_num=5):
#     data_4D = np.zeros([len(input), Seg_num, 10, 9, Fre_num])
#     data = input
#
#     # 'FP'
#     data_4D[:, :, 0, 3, :] = data[:, :, 0, :]
#     data_4D[:, :, 0, 4, :] = data[:, :, 1, :]
#     data_4D[:, :, 0, 5, :] = data[:, :, 2, :]
#
#     # 'AF'
#     data_4D[:, :, 1, 3, :] = data[:, :, 3, :]
#     data_4D[:, :, 1, 5, :] = data[:, :, 4, :]
#
#     # 'F' + 'FT' + 'T/C' + 'TP/CP' + 'P'
#     for i in range(45):
#         data_4D[:, :, i // 9 + 2, i % 9, :] = data[:, :, i + 5, :]
#
#     # 'PO'
#     j = 1
#     for i in range(50, 57):
#         data_4D[:, :, 7, j, :] = data[:, :, i, :]
#         j = j + 1
#
#     # 'O'
#     j = 3
#     for i in range(57, 60):
#         data_4D[:, :, 8, j, :] = data[:, :, i, :]
#         j = j + 1
#
#     # 'CB'
#     data_4D[:, :, 9, 3, :] = data[:, :, 60, :]
#     data_4D[:, :, 9, 5, :] = data[:, :, 61, :]
#     data_4D_reshape = data_4D.transpose((0, 1, 4, 2, 3))
#     return data_4D_reshape
#
# if __name__ == '__main__':
#     Segment_num = 5
#     Fre_num = 5
#     data = np.random.random((16, Segment_num, 62, Fre_num))
#
#     input = Data_4D_Generation(data, Fre_num=5, Seg_num=5)
#     input = torch.FloatTensor(input)
#     # input = torch.rand((32, Segment_num, Fre_num, 10, 9))  # batch_size, Num_segment, Fre_Bands, 2DMap_height, 2DMap_weight
#     net = SFT_Net()
#     output = net(input)
#     print("Input shape     : ", input.shape)
#     print("Output shape    : ", output.shape)


# %% STGCN
# STGCN is implemented in TensorFlow, and the specific code can be found at the release site of the MMV dataset
