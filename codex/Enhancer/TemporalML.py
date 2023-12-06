import torch
import torch.nn as nn
import math

class RPPsAtt(torch.nn.Module):

    def __init__(self, args, attention_type='general'):
        super(RPPsAtt, self).__init__()
        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')
        dimensions = args.input_dim
        self.num_nodes = args.num_nodes
        self.attention_type = attention_type

        if self.attention_type == 'general':
            self.linear_in = torch.nn.Linear(dimensions, dimensions, bias=False)
        self.linear_out = torch.nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh()
        self.ae = torch.nn.Parameter(torch.FloatTensor(self.num_nodes, 1, 1))
        self.ad = torch.nn.Parameter(torch.FloatTensor(self.num_nodes, 1, 1))
        self.ab = torch.nn.Parameter(torch.FloatTensor(self.num_nodes, 1, 1))
        self.device = args.device
        self.index = -1

    def forward(self, query, context, index):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        # Compute weights across every context sequence
        attention_scores = attention_scores.reshape(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.reshape(batch_size, output_len, query_len)

        mix = attention_weights * (context.permute(0, 2, 1))
        delta_t = torch.flip(torch.arange(0, query_len), [0]).type(torch.float32).to(self.device)
        delta_t = delta_t.repeat(self.num_nodes, 1).reshape(self.num_nodes, 1, query_len)
        bt = torch.exp(-1*self.ab * delta_t)
        term_2 = torch.abs(self.ae * mix * bt)
        term_3 = torch.abs(-1 * self.ad * mix * bt)
        mix = torch.tanh(torch.sum(term_2+mix-term_3, -1)).unsqueeze(1)

        combined = torch.cat((mix, query), dim=2)
        combined = combined.reshape(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).reshape(batch_size, output_len, dimensions)
        output = self.tanh(output)
        return output, attention_weights

class TemporalML(nn.Module):
    def __init__(self, args, data_dis):
        super().__init__()
        self.in_dim = args.input_dim
        self.emb_dim = args.emb_dim
        self.num_nodes = args.num_nodes
        self.lookback = args.lookback
        self.conv_list = []
        self.q_linear = nn.ModuleList()
        self.k_linear = nn.ModuleList()
        self.v_linear = nn.ModuleList()
        self.layernorm = nn.ModuleList()
        self.device = args.device
        self.relu = nn.LeakyReLU()
        self.process = RPPsAtt(args)
        self.mean = nn.Parameter(torch.tensor(data_dis[0], device=self.device))
        self.var = nn.Parameter(torch.tensor(data_dis[1], device=self.device))
        self.normal = nn.Parameter(torch.normal(size=(self.num_nodes, self.lookback, self.in_dim), mean=data_dis[0], std=data_dis[1], device=self.device))
        for i in range(1):
            self.q_linear.append(nn.Linear(self.in_dim * self.lookback, self.emb_dim))
            self.k_linear.append(nn.Linear(self.in_dim * self.lookback, self.emb_dim))
            self.v_linear.append(nn.Linear(self.in_dim * self.lookback, self.emb_dim))
            self.layernorm.append(nn.LayerNorm(self.emb_dim))
            self.layernorm.append(nn.LayerNorm(self.emb_dim))
            self.layernorm.append(nn.LayerNorm(self.emb_dim))

    def forward(self, x, index=None):
        '''
        params:
        x: [batch, Number_nodes, T, emb_dim]
        y: [b, N]
        '''
        b, n, t, d = x.size()
        RPPAtt = []
        dis = []
        for idx in range(b):
            q1, k1, v1 = self.q_linear[0](x[idx].reshape(n, -1)), self.k_linear[0](x[idx].reshape(n, -1)), \
                         self.v_linear[0](x[idx].reshape(n, -1))
            q1, k1, v1 = self.layernorm[0](q1), self.layernorm[1](k1), self.layernorm[2](v1)
            data = torch.mm(torch.softmax(torch.mm(q1, k1.permute(1, 0)) / math.sqrt(self.num_nodes), dim=-1),
                            self.normal.reshape(n, -1)).reshape(n, t, d)
            dis.append(data)
            _, weight = self.process(x[idx][:, -1:, :], data, index)
            RPPAtt.append(weight)
        data = torch.stack(dis, dim=0)
        res = torch.stack(RPPAtt, dim=0).permute(0, 1, 3, 2)
        data = data + res
        return data
