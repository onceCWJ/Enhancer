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


    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top1 = set()
        gt_top5 = set()
        gt_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top1) < 1:
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
        if len(gt_top5)==5 and len(pre_top5)==5:
            ndcg5.append(ndcg_score(np.array(list(gt_top5)).reshape(1,-1), np.array(list(pre_top5)).reshape(1,-1)))

        top1_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt

        if len(pre_top1)==1:
            real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
            bt_long += real_ret_rat_top
            sharpe_li.append(real_ret_rat_top)

        if len(pre_top5)==5:
            real_ret_rat_top5 = 0
            for pre in pre_top5:
                real_ret_rat_top5 += ground_truth[pre][i]
            real_ret_rat_top5 /= 5
            bt_long5 += real_ret_rat_top5
            sharpe_li5.append(real_ret_rat_top5)
        profit_list.append(10000 * bt_long5)
        if len(pre_top10)==10:
            real_ret_rat_top10 = 0
            for pre in pre_top10:
                real_ret_rat_top10 += ground_truth[pre][i]
            real_ret_rat_top10 /= 10
            bt_long10 += real_ret_rat_top10
            sharpe_li10.append(real_ret_rat_top10)

    performance['ndcg'] = round(np.mean(ndcg5), 4)
    performance['mrr'] = round(mrr_top / (prediction.shape[1] - all_miss_days_top), 4)
    # performance['irr'] = bt_long
    performance['irr'] = round(bt_long5, 4)
    # performance['irr10'] = bt_long10
    sharpe_li = np.array(sharpe_li)
    sharpe_rf_dict = {'NASDAQ':0.0212, 'NYSE':0.0212, 'TSE':0.0008, 'Ashare':0.0246}
    free = np.power(1 + sharpe_rf_dict[dataset], 1 / 365)-1
    performance['sr'] = round((np.mean(sharpe_li5)-free)/np.std(sharpe_li5), 4)
    return performance

def evaluate(prediction, ground_truth, dataset, eva_type, model_type, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    mask_list = []
    wo_mask_list = []
    for data_idx in range(prediction.shape[1]):
        msk = np.array(ground_truth[:, data_idx] > -1e-4, dtype=np.float32) # -1e-4
        wo_msk = np.array(ground_truth[:, data_idx] > -2e-2, dtype=np.float32) # -0.02
        mask_list.append(msk)
        wo_mask_list.append(wo_msk)
    mask_list = np.array(mask_list)
    wo_mask_list = np.array(wo_mask_list)

    performance = metric_compute(prediction, ground_truth, mask_list.T, dataset, eva_type, model_type)
    performance_wo_msk = metric_compute(prediction, ground_truth, wo_mask_list.T, dataset, eva_type, model_type)
    performance['irr'] = performance_wo_msk['irr']-1.0
    return performance


