import torch
from codex.Enhancer.RelationalML import *
from codex.Enhancer.TemporalML import *
from codex.evaluator import evaluate
from codex.Enhancer.Pretrain_model import *
from codex.backbones.GRU import *
from codex.backbones.Linear import *
from codex.backbones.TCN import *
from codex.backbones.Transformer import *
from codex.Enhancer.forecast_loss import *
from codex.Enhancer.forecast_utils import *
import random
import copy

def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

class DoubleLearner(nn.Module):
    def __init__(self, args, standard_scaler=None, forecast=False):
        super().__init__()
        self.emb_dim = args.emb_dim
        self.hid_dim = args.hid_dim
        self.device = args.device
        self.forecast = forecast
        self.standard_scaler = standard_scaler
        if forecast:
            self.TemporalML = TemporalML(args, [self.standard_scaler.mean, self.standard_scaler.std]).to(self.device)
        else:
            self.TemporalML = TemporalML(args, [self.standard_scaler[0], self.standard_scaler[1]]).to(self.device)
        self.RelationalML = RelationalML(args).to(self.device)
        self.model_type = args.model_type
        self.num_nodes = args.num_nodes
        self.batch_size = args.batch_size
        self.in_dim = args.input_dim
        self.clip = 5
        self.dataset = args.dataset
        self.inner_lr = args.inner_lr
        self.noise_factor = args.noise_factor
        self.base_lr = args.base_lr
        self.outer_tem_lr = args.outer_tem_lr
        self.outer_rel_lr = args.outer_rel_lr
        self.epsilon = args.epsilon
        self.lr_decay_ratio = args.lr_decay_ratio
        self.steps = args.steps
        self.beta = args.beta
        self.mean_factor = args.mean_factor
        self.var_factor = args.var_factor
        self.graph_learner = GraphLearner(args).to(self.device)
        self.pretrain_model = Pretrain_Model(args).to(self.device)
        self.relu = nn.ReLU()

        if self.model_type == 'linear':
            self.feature_network = mlp(args).to(self.device)
        elif self.model_type == 'TCN':
            self.feature_network = TCN(args).to(self.device)
        elif self.model_type == 'GRU':
            self.feature_network = GRU_model(args).to(self.device)
        elif self.model_type == 'Transformer':
            self.feature_network = Transformer(args).to(self.device)
        else:
            raise Exception('Invalid model type!')

        # self.init_params()

    def _compute_loss(self, y_true, y_predicted, standard_scaler):
        y_true = standard_scaler.inverse_transform(y_true)
        y_predicted = standard_scaler.inverse_transform(y_predicted)
        return masked_rmse_loss(y_predicted, y_true)

    def init_params(self):
        for p in self.feature_network.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=1.414)
        for p in self.RelationalML.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=1.414)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def sample_data(self, total_num, batch_size):
        if total_num <= batch_size:
            return list(range(total_num))
        st = random.randint(0, total_num - batch_size)
        return list(range(st, st + batch_size))

    def weighted_mse_loss(self, input, target, weight):
        return torch.mean(weight * (input - target) ** 2)

    def loss_rank(self, pred, ground_truth, mask, alpha):
        n_batch = ground_truth.shape[0]
        return_ratio = pred.reshape(-1, 1) # [b*n,1]
        ground_truth = ground_truth.reshape(-1, 1)  # [b*n, 1]
        mask = mask.view(-1, 1) # [b*n, 1]
        reg_loss = self.weighted_mse_loss(return_ratio, ground_truth, mask)
        all_ones = torch.ones(return_ratio.shape[0], 1).to(self.device)

        pre_pw_dif = (torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1)) - torch.matmul(all_ones, torch.transpose(return_ratio, 0, 1)))
        gt_pw_dif = (torch.matmul(all_ones, torch.transpose(ground_truth, 0, 1)) - torch.matmul(ground_truth, torch.transpose(all_ones, 0, 1)))
        mask_pw = torch.matmul(mask, torch.transpose(mask, 0, 1)) # [b*n, b*n]

        rank_loss = torch.mean(F.relu(((pre_pw_dif * gt_pw_dif) * mask_pw)))
        loss = reg_loss + alpha * rank_loss
        loss /= n_batch

        del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
        return loss

    def mean_loss(self, inputs, input_list):
        loss = 0
        for i in range(len(input_list)):
            loss = loss + input_list[i]
        mean_loss = torch.abs(loss/len(input_list)-inputs)
        return self.mean_factor * mean_loss
    
    def var_loss(self, input_list):
        loss = 0
        var_loss = 0
        for i in range(len(input_list)):
            loss = loss +input_list[i]
        for i in range(len(input_list)):
            var_loss = var_loss + torch.pow(input_list[i]-(loss/len(input_list)), 2)
        return self.var_factor * var_loss

    def pre_train(self, inputs, label, batches_seen):
        criterion = nn.MSELoss(reduction='mean')
        self.graph_opt = torch.optim.Adam([{'params': self.graph_learner.parameters(), 'lr': self.base_lr},
                                           {'params': self.pretrain_model.parameters(), 'lr': self.base_lr}],
                                            lr=self.base_lr)
        self.graph_opt.zero_grad()
        outputs, adj = self.pretrain_model(inputs, self.graph_learner, label, batches_seen)
        loss = criterion(outputs, label)
        losses = loss
        losses.backward()
        self.graph_opt.step()
        return loss

    def get_noise(self, inputs):
        normalize_tensor = F.normalize(inputs, p=2, dim=0)
        temp_tensor = torch.zeros_like(normalize_tensor, device=self.device) + 0.0001
        normalize_tensor = torch.where(normalize_tensor > 0, normalize_tensor, temp_tensor)
        noise = inputs[:, torch.randperm(inputs.size(1)), :] / (self.noise_factor * normalize_tensor)
        return noise

    def train_meta(self, inputs, label, stock_mask, index, adv_samples=None):
        # training the feature extractor network
        # transform_data: [b, N, T, emb_dim]
        # invar_pattern, variant_pattern: list of len T [N, in_dim]
        if index in self.steps:
            self.inner_lr = self.inner_lr * self.lr_decay_ratio
        self.inner_opt = torch.optim.Adam(params=self.feature_network.parameters(), lr=self.inner_lr, eps=self.epsilon)
        self.inner_opt.zero_grad()
        transform_data = self.TemporalML(inputs, index)
        inv_pattern, var_pattern = self.RelationalML(inputs, self.graph_learner)
        x = self.feature_network(transform_data, inv_pattern, var_pattern, intervene=False)
        result_list = []
        if self.forecast:
            loss = self._compute_loss(label, x, self.standard_scaler)
        else:
            loss = self.loss_rank(x, label, stock_mask, float(4))

        for adv_sample in adv_samples:
            no_transform_data = transform_data + self.TemporalML(adv_sample, index)
            y = self.feature_network(no_transform_data, inv_pattern, var_pattern, intervene=False)
            if self.forecast:
                result_list.append(self._compute_loss(label, y, self.standard_scaler))
            else:
                result_list.append(self.loss_rank(y, label, stock_mask, float(4)))
        var_loss = self.var_loss(result_list)
        mean_loss = self.mean_loss(loss, result_list)
        # Separate Training
        losses = loss + var_loss + mean_loss
        losses.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.feature_network.parameters(), self.clip)
        self.inner_opt.step()
        return loss

    def valid_meta(self, inputs, label, stock_mask, index):
        # training the Temporal Meta-Learner and Relational Meta-Learner
        if index in self.steps:
            self.base_lr = self.base_lr * self.lr_decay_ratio
            self.outer_tem_lr = self.outer_tem_lr * self.lr_decay_ratio
            self.outer_rel_lr = self.outer_rel_lr * self.lr_decay_ratio
        self.outer_opt = torch.optim.Adam([{'params': self.graph_learner.parameters(), 'lr': self.base_lr},
                                           {'params' : self.TemporalML.parameters(), 'lr': self.outer_tem_lr},
                                           {'params' : self.RelationalML.parameters(), 'lr': self.outer_rel_lr}],
                                           lr=self.base_lr, eps=self.epsilon)
        self.outer_opt.zero_grad()
        transform_data = self.TemporalML(inputs, index)
        inv_pattern, var_pattern = self.RelationalML(inputs, self.graph_learner)
        x_list = self.feature_network(transform_data, inv_pattern, var_pattern, intervene=True)
        mean_loss = []
        var_loss = []
        losses = []
        if self.forecast:
            cur_loss = self._compute_loss(label, x_list[-1], self.standard_scaler)
        else:
            cur_loss = self.loss_rank(x_list[-1], label, stock_mask, float(4))
        for i in range(len(x_list)-1):
            if self.forecast:
                loss = self._compute_loss(label, x_list[i], self.standard_scaler)  # 所有信息，包含variant information信息
            else:
                loss = self.loss_rank(x_list[i], label, stock_mask, float(4)) # 只包含invariant的信息
            losses.append(loss)
            if i%2:
                mean_loss.append(loss)
            else:
                var_loss.append(loss)
        loss1, loss2, loss3, loss4 = self.mean_loss(cur_loss, mean_loss), self.mean_loss(cur_loss, losses), 0, self.var_loss(var_loss)
        for i in range(len(mean_loss)):
            loss3 = loss3 + torch.abs(mean_loss[i] - var_loss[i]) / len(mean_loss)
        loss = cur_loss + self.beta * (loss1 + loss2) + loss3 + loss4
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.TemporalML.parameters(), self.clip)
            torch.nn.utils.clip_grad_norm_(self.RelationalML.parameters(), self.clip)
        self.outer_opt.step()
        return cur_loss

    def test_forecast(self, inputs):
        with torch.no_grad():
            transform_data = self.TemporalML(inputs)
            inv_pattern, var_pattern = self.RelationalML(inputs, self.graph_learner)
            pred = self.feature_network(transform_data, inv_pattern, var_pattern, intervene=False)
        return pred

    def test_meta(self, input_price, input_label, eval_type=None):
        indices = np.arange(0, len(input_price))
        no_of_samples = len(indices)
        with torch.no_grad():
            cur_pred = np.zeros([self.num_nodes, no_of_samples], dtype=float)
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                idc = indices[start:end]
                eval_price = torch.FloatTensor(input_price[idc]).to(self.device)
                transform_data = self.TemporalML(eval_price)
                inv_pattern, var_pattern = self.RelationalML(eval_price, self.graph_learner)
                pred = self.feature_network(transform_data, inv_pattern, var_pattern, intervene=False)
                pred = torch.transpose(pred.detach().cpu(), 0, 1).numpy()
                cur_pred[:, idc] = copy.copy(pred)
            performance = evaluate(cur_pred, input_label.T, self.dataset, eval_type, self.model_type)
            return performance