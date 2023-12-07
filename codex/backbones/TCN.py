import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from codex.Enhancer.intergate_mechanism import Intergate


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)


        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()
        self.emb_dim = args.emb_dim
        self.in_dim = args.input_dim
        self.hid_dim = args.hid_dim
        self.num_nodes = args.num_nodes
        self.horizon = args.horizon
        self.Intergrate = Intergate(args)
        self.seq_len = args.lookback
        self.fc = nn.Linear(self.seq_len, self.horizon)
        self.act = nn.LeakyReLU()
        self.TCN = TemporalConvNet(num_inputs=self.num_nodes * self.in_dim, num_channels=[self.num_nodes, self.num_nodes], kernel_size=3, dropout=0.0)

    def forward(self, transform_data, invariant_pattern=None, variant_pattern=None, intervene=None, abla=False):
        b, n, t, d = transform_data.shape
        if intervene:
            pred_list = []
            feature_list = self.Intergrate(transform_data, variant_pattern, invariant_pattern, intervene)
            for feature in feature_list:
                feature = feature.permute(0, 1, 3, 2).reshape(b, -1, t)
                pred = self.act(self.fc(self.TCN(feature)))
                pred_list.append(pred[:, :, 0])
            return pred_list
        else:
            feature = self.Intergrate(transform_data, variant_pattern, invariant_pattern, intervene)
            feature = feature.permute(0, 1, 3, 2).reshape(b, -1, t)
            pred = self.act(self.fc(self.TCN(feature))).permute(2, 0, 1)[:self.horizon, :, :].squeeze(dim=0)
            return pred
