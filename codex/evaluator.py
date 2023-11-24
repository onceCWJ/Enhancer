import math
import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import matplotlib.pyplot as plt

def metric_compute(prediction, ground_truth, mask, dataset, eva_type, model_type, report=False):
    assert ground_truth.shape == prediction.shape == mask.shape, 'shape mis-match'
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2/ np.sum(mask)
    ndcg5 = []
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0
    top_1_ground_truth = []
    top_5_ground_truth = []
    top_10_ground_truth = []
    sharpe_li = []
    sharpe_li5 = []
    sharpe_li10 = []
    profit_list = [10000]


    for i in range(prediction.shape[1]): # shape[1] each test sample
        rank_gt = np.argsort(ground_truth[:, i]) # 从小到大排序的元素index
        gt_top1 = set()
        gt_top5 = set()
        gt_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j] # 排名第j位的是哪只股票
            if mask[cur_rank][i] < 0.5: # 如果这只股票要mask掉，不考虑
                continue
            if len(gt_top1) < 1: # 分别记录未标注为mask的收益率最大的前1只，5只，10只股票
                gt_top1.add(cur_rank)
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)
        rank_pre = np.argsort(prediction[:, i])
        pre_top1 = set()
        pre_top5 = set()
        pre_top10 = set()

        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)
        # 计算nDCG@5 比较 真实收益率前5的股票set 和 预测收益率前5的股票set
        if len(gt_top5)==5 and len(pre_top5)==5:
            ndcg5.append(ndcg_score(np.array(list(gt_top5)).reshape(1,-1), np.array(list(pre_top5)).reshape(1,-1)))

        top1_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1: # 比较实际收益从大到小的股票index在不在预测的top1股票set里，实际是看预测top1在真实list的排序（跳过mask）
                    break
        if top1_pos_in_gt == 0: # 这种情况说明预测的top1不在mask了低收益股票后剩下的股票列表里
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt # MRR:预测第一的股票咋哎实际列表中的倒排

        if len(pre_top1)==1:
            real_ret_rat_top = ground_truth[list(pre_top1)[0]][i] # 第i个样本预测top1股票的真实收益率
            bt_long += real_ret_rat_top # 预测top1只股票的投资收益
            sharpe_li.append(real_ret_rat_top)

        if len(pre_top5)==5:
            real_ret_rat_top5 = 0
            for pre in pre_top5:
                real_ret_rat_top5 += ground_truth[pre][i]
            real_ret_rat_top5 /= 5
            bt_long5 += real_ret_rat_top5 # 预测top5只股票的平均投资收益
            sharpe_li5.append(real_ret_rat_top5)
        profit_list.append(10000 * bt_long5)
        if len(pre_top10)==10:
            real_ret_rat_top10 = 0
            for pre in pre_top10:
                real_ret_rat_top10 += ground_truth[pre][i]
            real_ret_rat_top10 /= 10
            bt_long10 += real_ret_rat_top10
            sharpe_li10.append(real_ret_rat_top10)


    if dataset == 'NYSE':
        plt.plot(np.arange(len(profit_list)),np.array(profit_list),linestyle='--')
        plt.savefig('graphs/{}_NYSE_profit'.format(model_type), dpi=1000)
        np.save('graphs/{}_profit_{}'.format(model_type,dataset), np.array(profit_list))
        
        
    elif dataset == 'NASDAQ':
        plt.plot(np.arange(len(profit_list)),np.array(profit_list),linestyle='--')
        plt.savefig('graphs/{}_NASDAQ_profit'.format(model_type), dpi=1000)
        np.save('graphs/{}_profit_{}'.format(model_type,dataset), np.array(profit_list))

    performance['ndcg'] = round(np.mean(ndcg5), 4)
    performance['mrr'] = round(mrr_top / (prediction.shape[1] - all_miss_days_top), 4) # 这里分母去掉了压根预测不在mask外的高收益股票-》离谱
    # performance['irr'] = bt_long
    performance['irr'] = round(bt_long5, 4) # 起始值=1.0, 后面bt_long5 += real_ret_rat_top5没对不同样本取平均？emo...
    # performance['irr10'] = bt_long10
    sharpe_li = np.array(sharpe_li) # 每个test样本的预测top1的股票的实际收益-》列表
    # print(sharpe_li5)
    # print(np.mean(sharpe_li5))
    # print(np.std(sharpe_li5))
    # input('sr')
    #performance['sr'] = (np.mean(sharpe_li5)/np.std(sharpe_li5))*15.87 ##To annualize： math.sqrt(252) 最后夏普率是算的预测top5只股票的平均投资收益-无风险国债除标准差的情况
    sharpe_rf_dict = {'NASDAQ':0.0212, 'NYSE':0.0212, 'TSE':0.0008, 'Ashare':0.0246}
    free = np.power(1 + sharpe_rf_dict[dataset], 1 / 365)-1 # US数据集test区间的无风险美国年化国债平均值 （中国：0.0246）(日本0.0008) (US: 0.0212)
    performance['sr'] = round((np.mean(sharpe_li5)-free)/np.std(sharpe_li5), 4)
    return performance



def evaluate(prediction, ground_truth, dataset, eva_type, model_type, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    #print(prediction.shape, ground_truth.shape, mask.shape)
    # progress_bar = tqdm(range(prediction.shape[1]))
    # progress_bar.set_description(f'evaluating: {dataset}')
    mask_list = []
    wo_mask_list = []
    #for data_idx in progress_bar:
    for data_idx in range(prediction.shape[1]):
        # msk = np.array(ground_truth[:, data_idx] > -1e-3, dtype=np.float32) #0.0 -0.001
        # wo_msk = np.array(ground_truth[:, data_idx] > -0.03, dtype=np.float32) # -0.02
        # msk = np.array(ground_truth[:, data_idx] > 0.001, dtype=np.float32) #0.0 -0.001
        # wo_msk = np.array(ground_truth[:, data_idx] > -2e-2, dtype=np.float32) # -0.02
        msk = np.array(ground_truth[:, data_idx] > -1e-4, dtype=np.float32) # -1e-4
        wo_msk = np.array(ground_truth[:, data_idx] > -2e-2, dtype=np.float32) # -0.02
        # msk = np.array(label > 0.001, dtype=np.float32)
        # wo_msk = np.array(label > -10.0, dtype=np.float32)
        mask_list.append(msk)
        wo_mask_list.append(wo_msk)
    #print(len(mask_list), len(wo_mask_list))
    mask_list = np.array(mask_list)
    wo_mask_list = np.array(wo_mask_list)
    #print(mask_list.shape, wo_mask_list.shape)
    #mask  = wo_mask_list.T
    #print(mask.shape)

    performance = metric_compute(prediction, ground_truth, mask_list.T, dataset, eva_type, model_type)
    performance_wo_msk = metric_compute(prediction, ground_truth, wo_mask_list.T, dataset, eva_type, model_type)
    performance['irr'] = performance_wo_msk['irr']-1.0
    return performance


