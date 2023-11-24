import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
       super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class FAGCN(nn.Module):
    def __init__(self,args,order=2):
        super(FAGCN,self).__init__()
        self.nconv = nconv()
        c_in = (order * 2) * args.input_dim
        c_out = args.input_dim
        self.mlp = linear(c_in,c_out)
        self.dropout = args.dropout
        self.order = order
        self.eps = args.eps
        self.eye = torch.eye(args.num_nodes).to(args.device)

    def forward(self,x,adj):
        out = []
        L = self.eps * self.eye + adj
        H = self.eps * self.eye - adj
        support = [L,H]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class Intergate(nn.Module):
    def __init__(self, args):
        super(Intergate, self).__init__()
        self.emb_dim = args.emb_dim
        self.device = args.device
        self.in_dim = args.input_dim
        self.look_back = args.lookback
        self.aggregate = FAGCN(args)
        self.model_type = args.model_type

    def forward(self, transform_data, variant_pattern, invariant_pattern, intervene):
        '''
        t_data: [b,N,T,emb_dim]
        variant pattern: [T,N,N]
        invariant pattern: [T,N,N]
        '''
        transform_data = transform_data.permute(0, 3, 1, 2)
        if intervene:
            feature_list = []
            for ind in range(len(invariant_pattern)):
                feature_list.append(self.aggregate(transform_data, invariant_pattern[ind]+variant_pattern[ind]).permute(0, 2, 3, 1))
                feature_list.append(self.aggregate(transform_data, invariant_pattern[ind]).permute(0, 2, 3, 1))
            inv_rel_info = torch.sum(torch.stack(invariant_pattern, dim=0), dim=0) / len(invariant_pattern)  # [N, N]
            feature_list.append(self.aggregate(transform_data, inv_rel_info).permute(0, 2, 3, 1))
            return feature_list
        else:
            inv_rel_info = torch.sum(torch.stack(invariant_pattern, dim=0), dim=0) / len(invariant_pattern) # [N, N]
            res = self.aggregate(transform_data, inv_rel_info).permute(0, 2, 3, 1)
            return res