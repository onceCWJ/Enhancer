from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import torch
import copy
from rank_utils import load_sample, load_stocklist, load_date
from codex.Enhancer.DoubleMeta import DoubleLearner
from codex.evaluator import evaluate
import logging
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
torch.set_num_threads(4)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--dataset', type=str, default='NASDAQ', help='Dataset')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=None, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--pretrain_epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--num_nodes', type=int, default=1737, help='Stock num.')
parser.add_argument('--lookback', type=int, default=16, help='Look back.')
parser.add_argument('--input_dim', type=int, default=5, help='Dimension of price input')
parser.add_argument('--emb_dim', type=int, default=50, help='Dimension of embdedding size')
parser.add_argument('--max_len', type=int, default=500, help='Dimension of embdedding size')
parser.add_argument('--d_model', type=int, default=512, help='Dimension of embdedding size')
parser.add_argument('--dim_feedforward', type=int, default=256, help='Dimension of embdedding size')
parser.add_argument('--n_heads', type=int, default=4, help='Dimension of embdedding size')
parser.add_argument('--hid_dim', type=int, default=30, help='Dimension of hidden embdedding size')
parser.add_argument('--order', type=int, default=4, help='Temporal polynomial order')
parser.add_argument('--feature_dim', type=int, default=10, help='hidden dim of base adjacency matrix')
parser.add_argument('--num_layers', type=int, default=2, help='number layers of LSTM/Transformer')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--inner_lr', type=float, default=0.003, help='learning rate of the feature network')
parser.add_argument('--outer_tem_lr', type=float, default=0.003, help='learning rate of the temporal meta-learner')
parser.add_argument('--outer_rel_lr', type=float, default=0.003, help='learning rate of the relational meta-learner')
parser.add_argument('--base_lr', type=float, default=0.003, help='base learning rate')
parser.add_argument('--model_type', type=str, default='GRU', help='feature network type')
parser.add_argument('--save_rank_model_checkpoint_path', type=str, default='save_models/', help='save model path')
parser.add_argument('--only_test', action='store_true')
parser.add_argument('--noise_factor', type=float, default=1000, help='Normalizing factor of noise')
parser.add_argument('--mean_factor', type=float, default=100, help='Amplification of the mean loss function')
parser.add_argument('--var_factor', type=float, default=1e5, help='Amplification of the variance loss function')
parser.add_argument('--beta', default=3, type=int, help='balancing factor')
parser.add_argument('--cl_decay_steps', default=2000, type=int)
parser.add_argument('--filter_type', default='dual_random_walk', type=str)
parser.add_argument('--period', type=int, default=60, help='Temporal polynomial period')
parser.add_argument('--horizon', default=1, type=int)
parser.add_argument('--max_diffusion_step', default=2, type=int)
parser.add_argument('--num_rnn_layers', default=1, type=int)
parser.add_argument('--output_dim', default=1, type=int)
parser.add_argument('--epsilon',default=2.0e-3,type=float)
parser.add_argument('--eps', default=0.8, type=int)
parser.add_argument('--steps',default=[10, 30, 50],type=list)
parser.add_argument('--lr_decay_ratio',default=0.1,type=float)
parser.add_argument('--rnn_units', default=96, type=int)
parser.add_argument('--use_curriculum_learning', default=True, type=bool)
parser.add_argument('--embedding_size', default=256, type=int)
parser.add_argument('--clip', default=5, type=int)
parser.add_argument('--kernel_size', default=5, type=int)
args = parser.parse_args()
dataset = args.dataset
device = args.device

def model_train():
    # setup_seed(20)
    logger.info('loading sample data ... ')
    sample_path = "./data_{}/samples_{}/train_data".format(dataset, dataset)
    train_market, train_price, train_label, _, stock_mask = load_sample(sample_path)
    sample_path = "./data_{}/samples_{}/dev_data".format(dataset, dataset)
    dev_market, dev_price, dev_label, _, _ = load_sample(sample_path)
    sample_path = "./data_{}/samples_{}/test_data".format(dataset, dataset)
    test_market, test_price, test_label, _, _ = load_sample(sample_path)
    test_market, test_price, test_label, _, _ = load_sample(sample_path)

    logger.info(f'sample_path: samples_{dataset}')
    logger.info(f' train set: {train_market.shape[0]}, dev set: {dev_market.shape[0]}, '
                f'test set: {test_market.shape[0]}')
    args.num_nodes = train_price.shape[1]

    # prior_graph = torch.mean(torch.tensor(np.load('relation/sector_industry/{}_industry_relation.npy'.format(dataset)), device=args.device), dim=-1)
    prior_graph = torch.rand(args.num_nodes, args.num_nodes, device=args.device)
    train_price_data = torch.tensor(train_price, device=args.device)
    engine = DoubleLearner(args, [torch.mean(train_price_data), torch.std(train_price_data)], forecast=False).to(args.device)
    logger.info("start training...")

    best_irr, best_sr = -1, -1
    for i_epoch in range(args.pretrain_epochs):

        # --- Meta Training step ---
        t1 = time.time()
        indices = np.arange(0, len(train_price))
        np.random.shuffle(indices)
        train_loss = 0
        progress_bar_train = tqdm(range(0, round(len(indices)), args.batch_size))
        progress_bar_train.set_description(f'Epoch: {i_epoch}')

        for start in progress_bar_train:
            end = start + args.batch_size
            idc = indices[start:end]
            train_price_sample = torch.FloatTensor(train_price[idc]).to(device)
            train_label_sample = torch.FloatTensor(train_label[idc]).to(device)
            s_loss = engine.pre_train(train_price_sample, train_label_sample, start)
            train_loss += s_loss * len(idc)

        train_loss /= len(indices)
        t2 = time.time()
        logger.info(f'Pre Train MSE Loss: {train_loss}')
        logger.info('Pre Training Time: {:.4f} secs'.format(t2 - t1))
    
    test_sr = []
    test_irr =[]
    best_test_sr_list = []
    best_test_irr_list =[]
    training_time = []
    inference_time = []

    for i_epoch in range(round(args.epochs * 0.7)):
        
        # --- Meta Training step ---
        t1 = time.time()
        indices = np.arange(0, len(train_price))
        np.random.shuffle(indices)
        train_loss = 0
        train_time = 0
        progress_bar_train = tqdm(range(0, round(len(indices)*0.7), args.batch_size))
        progress_bar_train.set_description(f'Epoch: {i_epoch}')

        for start in progress_bar_train:
            noise_sample = []
            end = start + args.batch_size
            idc = indices[start:end]
            for k in range(10):
                rand_index = random.randint(0, round(len(indices)*0.7)-args.batch_size)
                end_index = rand_index + args.batch_size
                idcx = indices[rand_index:end_index]
                noise_sample.append(torch.FloatTensor(train_price[idcx]).to(device))
            train_price_sample = torch.FloatTensor(train_price[idc]).to(device)
            train_label_sample = torch.FloatTensor(train_label[idc]).to(device)
            stock_mask_sample = torch.FloatTensor(stock_mask[idc]).to(device)
            s_loss = engine.train_meta(train_price_sample, train_label_sample, stock_mask_sample, i_epoch, noise_sample)
            train_loss += s_loss*len(idc)

        train_loss /= len(indices)
        t2 = time.time()
        logger.info(f'Meta Train Rank Loss: {train_loss}')
        logger.info('Meta Training Time: {:.4f} secs'.format(t2 - t1))
        train_time += (t2 - t1)
        # --- Meta Valid step ---
        t1 = time.time()
        indices = np.arange(0, len(train_price))
        np.random.shuffle(indices)
        valid_loss = 0
        progress_bar_valid = tqdm(range(round(len(indices)*0.7), len(indices), args.batch_size))
        progress_bar_valid.set_description(f'Epoch: {i_epoch}')

        for start in progress_bar_valid:
            end = start + args.batch_size
            idc = indices[start:end]
            train_price_sample = torch.FloatTensor(train_price[idc]).to(device)
            train_label_sample = torch.FloatTensor(train_label[idc]).to(device)
            stock_mask_sample = torch.FloatTensor(stock_mask[idc]).to(device)
            s_loss = engine.valid_meta(train_price_sample, train_label_sample, stock_mask_sample, i_epoch)
            train_loss += s_loss * len(idc)

        train_loss /= len(indices)
        t2 = time.time()
        logger.info(f'Meta Valid Rank Loss: {train_loss}')
        logger.info('Meta Valid Time: {:.4f} secs'.format(t2 - t1))
        train_time += (t2 - t1)
        
        training_time.append(train_time)
        # --- Evaluate step ---    
        with torch.no_grad():
            valid_performance = engine.test_meta(dev_price, dev_label)
            logger.info(f"validing result: mse: {valid_performance['mse']}, sr: {'%.4f'%valid_performance['sr']}, irr: {'%.4f'%valid_performance['irr']}, ndcg: {'%.4f'%valid_performance['ndcg']},")
            
            s1 = time.time()
            test_performance = engine.test_meta(test_price, test_label)
            s2 = time.time()
            logger.info(f"testing result: mse: {test_performance['mse']}, sr: {'%.4f'%test_performance['sr']}, irr: {'%.4f'%test_performance['irr']}, ndcg: {'%.4f'%test_performance['ndcg']}")
            
            logger.info('Inference Time: {:.4f} secs'.format(s2 - s1))
            inference_time.append(s2-s1)
            test_sr.append(test_performance['sr'])
            test_irr.append(test_performance['irr'])
            if best_sr < valid_performance['sr'] or best_irr < valid_performance['irr']:
                best_sr = max(valid_performance['sr'], best_sr)
                best_irr = max(valid_performance['irr'], best_irr)
                best_test_sr_list.append(test_performance['sr'])
                best_test_irr_list.append(test_performance['irr'])
                if args.save_rank_model_checkpoint_path is not None:
                    torch.save(engine.feature_network, args.save_rank_model_checkpoint_path+'fea_{}_best_eval22.pth'.format(dataset))
                    torch.save(engine.TemporalML, args.save_rank_model_checkpoint_path + 'TML_{}_best_eval22.pth'.format(dataset))
                    torch.save(engine.RelationalML, args.save_rank_model_checkpoint_path + 'RML_{}_best_eval22.pth'.format(dataset))
    print('test_sr_mean:{}'.format(np.mean(np.array(test_sr))))
    print('test_irr_mean:{}'.format(np.mean(np.array(test_irr))))
    print('best_test_sr_mean:{}'.format(np.mean(np.array(best_test_sr_list))))
    print('best_test_irr_mean:{}'.format(np.mean(np.array(best_test_irr_list))))
    print('mean_train_time:{}'.format(np.mean(np.array(training_time))))
    print('mean_inference_time:{}'.format(np.mean(np.array(inference_time))))

def model_test():
    sample_path = "../data_{}/samples_{}/test_data".format(dataset, dataset)
    test_market, test_price, test_label, _, _ = load_sample(sample_path)
    stock_list_path = "../data_{}/code2ID.txt".format(dataset)
    stock_list = load_stocklist(stock_list_path)

    date_list_path = "../data_{}/samples_{}/test_date".format(dataset, dataset)
    date_list = load_date(date_list_path)

    logger.info(f'sample_path: samples_{dataset}')
    logger.info(f'sample_path: samples_{dataset}')

    best_model = torch.load(args.save_rank_model_checkpoint_path+'{}_best_eval.pth'.format(dataset), map_location=args.device)

    # --- Test step ---
    indices = np.arange(0, len(test_price))
    no_of_test_samples = len(indices)
    cur_test_pred = np.zeros([args.stock, no_of_test_samples], dtype=float)
    s1 = time.time()

    with torch.no_grad():
        best_model.eval()
        for start in range(0, len(indices), args.batch_size):
            end = start + args.batch_size
            idc = indices[start:end]
            test_market_sample = torch.FloatTensor(test_market[idc]).to(device)
            test_price_sample = torch.FloatTensor(test_price[idc]).to(device)
            cur_rr, _ = best_model(test_price_sample, test_market_sample, args.temperature, args.gumbel_soft)
            cur_rr = torch.transpose(torch.squeeze(cur_rr).detach().cpu(), 0, 1).numpy()
            cur_test_pred[:, idc] = copy.copy(cur_rr)

    s2 = time.time()
    test_performance = evaluate(cur_test_pred, test_label.T, dataset, 'test')
    logger.info(f"testing result: mse: {test_performance['mse']}, sr: {'%.4f'%test_performance['sr']}, irr: {'%.4f'%test_performance['irr']}, nDCG: {'%.4f'%test_performance['ndcg']}")
    logger.info('Inference Time: {:.4f} secs'.format(s2 - s1))
    with open(f'{dataset}_rank_result.txt', 'w') as f:
        f.write('sample_idx, date, rankth, stockno, pred_return\n')
        for i in range(cur_test_pred.shape[1]): # shape[1] each test sample
            rank_result = np.argsort(cur_test_pred[:, i]) # 从小到大排序的元素index
            for j in range(1,31): # 排名第j位的是哪只股票
                f.write(f"{i}, {date_list[i]}, {j}, {stock_list[rank_result[-1*j]]}, {cur_test_pred[rank_result[-1*j], i]}\n")
    logger.info('Saving Rank Pool Result Success. ')


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    if args.seed is not None:
        set_seed(args.seed)


    logger.info(args)
    logger.info(f'device: {torch.cuda.is_available(), device}')

    if args.only_test:
        model_test()
        exit(0)

    model_train()


