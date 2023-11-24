import torch.nn as nn
from codex.Enhancer.forecast_loss import *
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

    def graph_norm(self, W):
        N, N = W.shape
        W = W + torch.eye(N).to(W.device)
        D = W.sum(axis=1)
        D = torch.diag(D ** (-0.5))
        out = D @ W @ D
        return out

    def forward(self, dates):
        graphs = []
        # print(dates)
        g = self.relu(torch.matmul(self.E, self.E.permute(1, 0)))
        coeff = torch.softmax(torch.mm(self.emb, self.W_c), dim=-1)
        for i in range(self.lookback):
            graph = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
            date = torch.tensor(dates[i] % self.period, dtype=torch.long)
            for j in range(self.order):
                graph = (graph + coeff[date][j] * torch.matrix_power(g, j) / (self.order))
            graphs.append(graph)
        graphs = torch.stack(graphs, dim=0)
        return graphs


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
        self.q_linear = nn.Linear(in_dim, hid_dim)
        self.k_linear = nn.Linear(in_dim, hid_dim)
        self.v_linear = nn.Linear(in_dim, hid_dim)
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
        self.fc = nn.Linear(self.num_nodes, in_dim)
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
        # across-time aggregate: Temporal Hawkes Attention
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
        dyn_graph_list = graph_learner(x[0, 0, :, -1]) # input date information for graph generate
        inv_pattern, var_pattern = self.decouple(dyn_graph_list)
        return inv_pattern, var_pattern