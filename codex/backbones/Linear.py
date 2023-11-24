import torch
import torch.nn as nn
import torch.nn.functional as F
from codex.Enhancer.intergate_mechanism import Intergate
from torch.autograd import Function
import math

class mlp(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb_dim = args.emb_dim
        self.device = args.device
        self.in_dim = args.input_dim
        self.look_back = args.lookback
        self.horizon = args.horizon
        self.Intergrate = Intergate(args).to(self.device)
        self.fc1 = nn.Linear(in_features=self.in_dim, out_features=self.emb_dim)
        self.fc = nn.Linear(in_features=self.emb_dim, out_features=1)
        self.fc2 = nn.Linear(in_features=self.look_back, out_features=self.horizon)

    def forward(self, transform_data, invariant_pattern=None, variant_pattern=None, intervene=None, abla=None):
        '''
        t_data: [b,N,T,emb_dim]
        variant_pattern: [T,N,in_dim]
        '''
        if intervene:
            pred_list = []
            feature_list = self.Intergrate(transform_data, variant_pattern, invariant_pattern, intervene)
            for feature in feature_list:
                pred = torch.sigmoid(self.fc(torch.sigmoid(self.fc1(feature))).squeeze(dim=-1))
                pred = self.fc2(pred).squeeze(dim=-1)
                pred_list.append(pred)
            return pred_list
        else:
            feature = self.Intergrate(transform_data, variant_pattern, invariant_pattern, intervene)
            pred = torch.sigmoid(self.fc(torch.sigmoid(self.fc1(feature))).squeeze(dim=-1))
            pred = self.fc2(pred).permute(2, 0, 1).squeeze(dim=0)
            return pred