import torch.nn as nn
from codex.Enhancer.forecast_loss import *
import torch.nn.functional as F
import math

def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

class GraphLearner(nn.Module):
    def __init__(self, args):
        super(GraphLearner, self).__init__()
        self.lookback = args.lookback
        self.input_dim = args.input_dim
        self.num_nodes = args.num_nodes
        self.order = args.order
        self.device = args.device
        self.emb_dim = args.emb_dim
        self.feature_dim = args.feature_dim
        self.period = args.period
        self.coeff = []
        self.g = []
        self.E = nn.Parameter(torch.rand(self.num_nodes, self.feature_dim, device=self.device))
        self.emb = nn.Parameter(torch.rand(self.period, self.emb_dim, device=self.device))
        self.W_c = nn.Parameter(torch.rand(self.emb_dim, self.order, device=self.device))
        self.relu = nn.ReLU()
        self.ContextMatching = ContextMatching(args).to(self.device)

    def graph_norm(self, W):
        N, N = W.shape
        W = W + torch.eye(N).to(W.device)
        D = W.sum(axis=1)
        D = torch.diag(D ** (-0.5))
        out = D @ W @ D
        return out

    def forward(self, inputs, pretrain=False):
        graphs = []
        # print(dates)
        g = self.relu(torch.matmul(self.E, self.E.permute(1, 0)))
        coeff = torch.softmax(torch.mm(self.emb, self.W_c), dim=-1)
        for i in range(self.period):
            graph = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
            for j in range(self.order):
                graph = (graph + coeff[i][j] * torch.matrix_power(g, j) / (self.order))
            graphs.append(graph)
        graphs = torch.stack(graphs, dim=0)
        context = self.ContextMatching(inputs, graphs, pretrain=pretrain)
        return context

def compute_metric(time1, time2, correlation_matrix):
    # Flatten and reshape data for cosine similarity calculation
    loss = masked_mae_loss(torch.mm(correlation_matrix, time1), time2)
    return loss

class nconv(nn.Module):
    def __init__(self):
       super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('nv,vw->nw',(x,A))
        return x.contiguous()

class ContextMatching(nn.Module):
    def __init__(self, args):
        super(ContextMatching, self).__init__()
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.device = args.device
        self.emb_dim = args.emb_dim
        self.lookback = args.lookback
        self.period = args.period
        self.conv = nconv()
        self.gru = nn.GRU(input_size=self.num_nodes*self.num_nodes, hidden_size=self.period, batch_first=True)

    def forward(self, inputs, graphs, pretrain=False):
        b, n, t, d = inputs.shape
        if pretrain:
            lenx = torch.randint(0, self.period, (self.lookback,))
            context = graphs[lenx]
            return context
        else:
            sc = inputs.permute(2, 1, 0, 3).reshape(t, n, -1)
            context_list = []
            for idx in range(t-1):
                time_step_data = sc[idx]
                time_forward = sc[idx+1]
                metrics = torch.tensor([compute_metric(time_step_data, time_forward, graphs[p]) for p in range(self.period)])
                optimal_matrix_idx = metrics.argmin()
                # print('idx:{}'.format(optimal_matrix_idx))
                context_list.append(graphs[optimal_matrix_idx])
            context_list.append(torch.cov(sc[-1]))
            context = torch.stack(context_list, dim=0).reshape(t, -1).unsqueeze(0)
            output, _ = self.gru(context) # output.shape: [batch_size(1), seq_len, hidden_size]
            last_idx = F.softmax(torch.squeeze(output[:,-1,:]), dim=0).argmax()
            # print('last_idx:{}'.format(last_idx))
            context_list[-1] = graphs[last_idx]
            return torch.stack(context_list, dim=0)

class Decouple(nn.Module):
    def __init__(self, args, eps=0.95):
        super(Decouple, self).__init__()
        in_dim = args.input_dim
        hid_dim = args.hid_dim
        self.eps = eps
        self.device = args.device
        self.num_nodes = args.num_nodes
        self.lookback = args.lookback
        self.period = args.period
        self.dropout = args.dropout
        self.diff_step = args.max_diffusion_step
        self.q_linear = nn.Linear(self.K, hid_dim)
        self.k_linear = nn.Linear(self.K, hid_dim)
        self.v_linear = nn.Linear(self.K, hid_dim)
        self.feature_dim = args.feature_dim
        self.K = math.ceil(math.log(self.num_nodes)) # self.num_nodes/math.log(self.num_nodes)
        self.E1 = torch.nn.Parameter(torch.rand(self.num_nodes, self.feature_dim, device=self.device))
        self.E2 = torch.nn.Parameter(torch.rand(self.feature_dim, hid_dim, device=self.device))
        self.Gconv = nn.Linear(hid_dim * (self.diff_step + 1), self.num_nodes)
        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        self.layernorm3 = nn.LayerNorm(hid_dim)
        self.neighbor = torch.zeros(self.lookback, self.num_nodes, self.K, device=self.device, dtype=torch.long)
        self.point = torch.rand(self.lookback, self.num_nodes, self.K, device=self.device)
        self.fc = nn.Linear(self.num_nodes, self.K)
        self.eps = args.eps
        self.act = nn.ReLU(inplace=False)
        self.variant_pattern = []
        self.invariant_pattern = []

    def collect_negborhoods(self, graph_list):
        # collect the neighbor in time-span lookback
        # collect neighborhood for every node in every time-point
        for index in range(len(graph_list)):
            # iterative over time
            graph = torch.softmax(graph_list[index], dim=-1)
            self.point[index] = torch.topk(graph, k=math.ceil(math.log(self.num_nodes)))[0].detach()
            self.neighbor[index] = torch.topk(graph, k=math.ceil(math.log(self.num_nodes)))[1]
        return self.point

    def neighbor_propogate(self, node_embedding):
        # The code is divided into two part: in-time aggregate and across-time aggregate
        # across-time aggregate
        for idx in range(len(self.neighbor)):
            node_embedding[idx] = torch.sum(node_embedding[idx][self.neighbor[idx]] * (torch.unsqueeze(self.point[idx], dim=-1)), dim=1) + node_embedding[idx]
        return node_embedding

    def concat(self, x, x_):
        x_ = torch.unsqueeze(x_, dim=0)
        return torch.cat([x, x_], dim=0)

    def aggregate_feature(self, m, value):
        x1 = m@value
        x0 = value.clone()
        value = torch.unsqueeze(value, dim=0)
        x = self.concat(value, x1)
        for k in range(1, self.diff_step):
            x2 = 2 * torch.mm(m, x1) - x0
            x = self.concat(x, x2)
            x1, x0 = x2, x1
        x = self.Gconv(x.permute(1, 2, 0).reshape(self.num_nodes, -1))
        return x

    def forward(self, graph_list):
        variant_pattern, invariant_pattern = [], []
        self.collect_negborhoods(graph_list)
        node_embedding = self.fc(graph_list)
        # update node embeddings accroding to its neighbors
        node_embedding = self.neighbor_propogate(node_embedding)
        for idx in range(len(graph_list)):
            # decouple the invariant and variant part
            q, k, v = self.q_linear(node_embedding[idx]), self.k_linear(node_embedding[idx]), self.v_linear(node_embedding[idx])
            q, k ,v = self.layernorm1(q), self.layernorm2(k), self.layernorm3(v)
            m_I = torch.softmax(q@(k.transpose(1, 0)) / math.sqrt(self.num_nodes), dim=-1) # m_I : [N,N]
            m_V = torch.softmax(-q@(k.transpose(1,0)) / math.sqrt(self.num_nodes), dim=-1) # m_V : [N,N]
            z_I = self.aggregate_feature(m_I, v)
            z_V = self.aggregate_feature(m_V, v*torch.softmax(self.E1@self.E2, dim=-1))
            variant_pattern.append(z_V)
            invariant_pattern.append(z_I)
        return variant_pattern, invariant_pattern

class RelationalML(nn.Module):
    def __init__(self, args):
        super(RelationalML, self).__init__()
        self.lookback = args.lookback
        self.input_dim = args.input_dim
        self.decouple = Decouple(args)
        self.num_nodes = args.num_nodes
        self.feature_dim = args.feature_dim

    def forward(self, x, graph_learner):
        dyn_graph_list = graph_learner(x) # input date information for graph generate
        inv_pattern, var_pattern = self.decouple(dyn_graph_list)
        return inv_pattern, var_pattern