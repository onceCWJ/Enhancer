import matplotlib.pyplot as plt
import random
import utils
from codex.Enhancer.DoubleMeta import *
from codex.Enhancer.forecast_loss import masked_mae_loss, masked_mape_loss, masked_mse_loss
import pandas as pd
import os
import time
torch.set_num_threads(4)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DMSupervisor:
    def __init__(self, args):
        self.args = args
        self.num_sample = args.num_sample
        self.device = args.device
        self.AR_list = []
        self.SR_list = []
        # data set
        self._data = utils.load_dataset(args.dataset_dir, args.batch_size)
        self.standard_scaler = self._data['scaler']
        self.T_N = args.T_N

        ### Feas
        # initialize input_dim:1 feas_dim:1 graph_input_dim:1 graph_feas_dim:1
        if args.dataset_dir == 'data/CSI300':
            self.dataset = 'CSI300'
            file = np.load('data/CSI300/CSI300.npy')
            arr = file[:,:,0]
        elif args.dataset_dir == 'data/CSI500':
            self.dataset = 'CSI500'
            file = np.load('data/CSI500/CSI500.npy')
            arr = file[:,:,0]


        args.dataset = self.dataset
        self.input_dim = args.input_dim
        args.num_nodes = arr.shape[1]
        self.num_nodes = args.num_nodes
        self.seq_len = args.lookback  # for the encoder
        self.output_dim = args.output_dim
        self.batch_size = args.batch_size
        self.model_type = args.model_type
        self.use_curriculum_learning = args.use_curriculum_learning
        self.horizon = args.horizon  # for the decoder
        self.best_val_loss = 9999
        self.maxm = -9999
        print(args)
        prior_graph = torch.rand(self.num_nodes, self.num_nodes, device=self.device)
        # setup model
        self.DoubleLearner = DoubleLearner(args, self.standard_scaler, True).to(self.device)
        print("Model created")
        print("Total Trainable Parameters: {}".format(count_parameters(self.DoubleLearner)))

    def save_test_model(self, dataset, epoch):
        if not os.path.exists('models_{}/'.format(dataset)):
            os.makedirs('models_{}/'.format(dataset))
        config = {}
        config['model_state_dict'] = self.DoubleLearner.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models_{}/epo{}.tar'.format(dataset, epoch))
        print("Saved model at {}".format(epoch))
        return 'models_{}/epo{}.tar'.format(dataset, epoch)

    def load_test_model(self, dataset, epoch):
        assert os.path.exists('models_{}/epo{}.tar'.format(dataset, epoch)), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load('models_{}/epo{}.tar'.format(dataset, epoch), map_location='cpu')
        self.DoubleLearner.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model at {}".format(epoch))

    def train(self, args):
        return self._train(args)

    def evaluate(self, dataset='val', epoch=None):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        self.dict = {}
        for i in range(self.num_nodes):
            self.dict[str(i)] = [0,0]
        self.topk_indices = []
        self.ratio_list = []
        with torch.no_grad():

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []
            mapes = []
            # rmses = []
            mses = []
            medae = []
            lenx = self.horizon
            l = [[] for i in range(lenx)]
            m = [[] for i in range(lenx)]
            r = [[] for i in range(lenx)]
            ae = [[] for i in range(lenx)]
            for batch_idx, (x_org, y_org) in enumerate(val_iterator):
                x, y = self._prepare_data(x_org, y_org)
                x = x.reshape(self.seq_len, self.batch_size, self.num_nodes, -1).permute(1, 2, 0, 3)
                output = self.DoubleLearner.test_forecast(x)
                y = y.squeeze(dim=0)
                loss = self._compute_loss(y, output)
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)

                if dataset == 'test':
                    top_k, ratio_list, dict = calc_AR(y_pred, y_org, x_org, self.topk_indices, self.dict)
                    self.ratio_list += ratio_list
                    self.topk_indices = top_k
                    self.dict = dict

                mapes.append(masked_mape_loss(y_pred, y_true).item())
                mses.append(masked_mse_loss(y_pred, y_true).item())
                medae.append(masked_medae_loss(y_pred, y_true).item())
                losses.append(loss.item())

                for i in range(lenx):
                    l[i].append(masked_mae_loss(y_pred[i:i + 1], y_true[i:i + 1]).item())
                    m[i].append(masked_mape_loss(y_pred[i:i + 1], y_true[i:i + 1]).item())
                    r[i].append(masked_mse_loss(y_pred[i:i + 1], y_true[i:i + 1]).item())
                    ae[i].append(masked_medae_loss(y_pred[i:i + 1], y_true[i:i + 1]).item())

            if dataset == 'test':
                AR, SR = calc_AR_SR(self.ratio_list)
                self.AR_list.append(AR)
                self.SR_list.append(SR)

            mean_loss = np.mean(losses)
            mean_mape = np.mean(mapes)
            mean_medae = np.mean(medae)
            mean_rmse = np.sqrt(np.mean(mses))
            return mean_loss, mean_mape, mean_rmse, mean_medae

    def test(self, args, epoch_num):
        self.dict = {}
        for i in range(self.num_nodes):
            self.dict[str(i)] = [0, 0]
        self.topk_indices = []
        self.ratio_list = []
        with torch.no_grad():
            self.load_test_model(self.dataset, epoch_num)
            test_iterator = self._data['test_loader'].get_iterator()
            losses = []
            mapes = []
            # rmses = []
            mses = []
            medae = []
            lenx = args.horizon
            l = [[] for i in range(lenx)]
            m = [[] for i in range(lenx)]
            r = [[] for i in range(lenx)]
            ae = [[] for i in range(lenx)]
            for batch_idx, (x_org, y_org) in enumerate(test_iterator):
                x, y = self._prepare_data(x_org, y_org)
                x = x.reshape(self.seq_len, self.batch_size, self.num_nodes, -1).permute(1, 2, 0, 3)
                output = self.DoubleLearner.test_forecast(x)
                y = y.squeeze(dim=0)
                loss = self._compute_loss(y, output)
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)

                top_k, ratio_list, dict = calc_AR(y_pred, y_org, x_org, self.topk_indices, self.dict)
                self.ratio_list += ratio_list
                self.topk_indices = top_k
                self.dict = dict

                mapes.append(masked_mape_loss(y_pred, y_true).item())
                mses.append(masked_mse_loss(y_pred, y_true).item())
                medae.append(masked_medae_loss(y_pred, y_true).item())
                losses.append(loss.item())

                for i in range(lenx):
                    l[i].append(masked_mae_loss(y_pred[i:i+1], y_true[i:i+1]).item())
                    m[i].append(masked_mape_loss(y_pred[i:i+1], y_true[i:i+1]).item())
                    r[i].append(masked_mse_loss(y_pred[i:i+1], y_true[i:i+1]).item())
                    ae[i].append(masked_medae_loss(y_pred[i:i + 1], y_true[i:i + 1]).item())

            AR, SR = calc_AR_SR(self.ratio_list)
            mean_loss = np.mean(losses)
            mean_mape = np.mean(mapes)
            mean_medae = np.mean(medae)
            mean_rmse = np.sqrt(np.mean(mses))

            print('test_mae: {:.6f}, test_mape: {:.6f}, test_rmse: {:.6f}, test medae: {:.6f} '.format(
                    mean_loss, mean_mape, mean_rmse, mean_medae))
            return mean_loss, mean_mape, mean_rmse


    def _train(self, args):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        patience = args.patience
        epochs = args.epochs
        log_every = args.log_every
        test_every_n_epochs = args.test_every_n_epochs
        best_idx = 0
        train_time = []
        val_time = []

        print('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        print("num_batches:{}".format(num_batches))

        batches_seen = 0
        indices = np.arange(0, len(self._data['x_train']))

        # Pretrain Stage
        for epoch_num in range(args.pretrain_epochs):
            print("Pretrain #Num of epoch:", epoch_num)
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []
            start_time = time.time()
            for batch_idx, (x, y) in enumerate(train_iterator):
                x, y = self._prepare_data(x, y)
                x = x.reshape(self.seq_len, self.batch_size, self.num_nodes, -1).permute(1, 2, 0, 3)
                y = y.squeeze(dim=0)
                loss = self.DoubleLearner.pre_train(x, y, batches_seen)
                losses.append(loss.item())
                batches_seen += 1
            mean_loss = np.mean(losses)
            print('MAE Loss:{}'.format(mean_loss))

        batches_seen = 0
        # Meta-Train Stage
        for epoch_num in range(args.epochs):
            # print("Meta Train #Num of epoch:", epoch_num)
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []
            start_time = time.time()
            for batch_idx, (x, y) in enumerate(train_iterator):
                x, y = self._prepare_data(x, y)
                x = x.reshape(self.seq_len, self.batch_size, self.num_nodes, -1).permute(1, 2, 0, 3)
                y = y.squeeze(dim=0)
                adv_sample = []
                for k in range(self.T_N):
                    rand_index = random.randint(0, round(len(self._data['x_train']) * 0.7) - args.batch_size)
                    end_index = rand_index + args.batch_size
                    idcx = indices[rand_index:end_index]
                    adv_sample.append(torch.FloatTensor(self._data['x_train'][idcx]).to(self.device).permute(0, 2, 1, 3))
                loss = self.DoubleLearner.train_meta(x, y, stock_mask=None, index=epoch_num, adv_samples=adv_sample)
                losses.append(loss.item())
                batches_seen += 1
            mean_loss = np.mean(losses)
            print('Meta-Train MAE Loss:{}'.format(mean_loss))

            losses = []
            train_iterator = self._data['train_loader'].get_iterator()
            for batch_idx, (x_org, y_org) in enumerate(train_iterator):
                x, y = self._prepare_data(x_org, y_org)
                x = x.reshape(self.seq_len, self.batch_size, self.num_nodes, -1).permute(1, 2, 0, 3)
                y = y.squeeze(dim=0)
                loss = self.DoubleLearner.valid_meta(x, y, stock_mask=None, index=epoch_num)
                losses.append(loss.item())
                batches_seen += 1
            mean_loss = np.mean(losses)
            print('Meta-Valid MAE Loss:{}'.format(mean_loss))
            print("evaluating now!")
            end_time = time.time()
            val_loss, val_mape, val_rmse, val_medae = self.evaluate(dataset='val', epoch=epoch_num)
            end_time2 = time.time()
            
            train_time.append(end_time-start_time)
            val_time.append(end_time2-end_time)

            if (epoch_num % log_every) == log_every - 1:
                print('Epoch [{}/{}] ({}) train_mae: {:.6f}, val_mae: {:.6f}, val_mape: {:.6f}, val_rmse: {:.6f} val_medae: {:.6f}' \
                          '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen, np.mean(losses), val_loss, val_mape, val_rmse, val_medae,
                                                    (end_time - start_time), (end_time2 - start_time)))

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss, test_mape, test_rmse, test_medae = self.evaluate(dataset='test', epoch=epoch_num)
                print('Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f}, test_mape: {:.4f}, test_rmse: {:.4f}, test_medae: {:.6f}' \
                          '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen, np.mean(losses), test_loss, test_mape, test_rmse, test_medae,
                                                    (end_time - start_time), (end_time2 - start_time)))

            if val_loss < self.best_val_loss:
                wait = 0
                model_file_name = self.save_test_model(self.dataset, epoch_num)
                best_idx = epoch_num
                print('Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(self.best_val_loss, val_loss, model_file_name))
                self.best_val_loss = val_loss

            elif val_loss >= self.best_val_loss:
                wait += 1
                if wait == patience:
                    print('Early stopping at epoch: %d' % epoch_num)
                    break

        self.test(args, best_idx)
        print('max_AR:{}'.format(np.max(np.array(self.AR_list))))
        print('max_SR:{}'.format(np.max(np.array(self.SR_list))))
        print('median_AR:{}'.format(np.median(np.array(self.AR_list))))
        print('median_SR:{}'.format(np.median(np.array(self.SR_list))))

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(self.device), y.to(self.device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)
