import torch
import numpy as np

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0   # delete Nan
    # print('mae_loss:{}'.format(loss))
    return loss.mean()

def masked_medae_loss(y_pred, y_true):
    loss = torch.median(torch.abs(y_pred - y_true), dim=1)[0]
    return loss.mean()

def masked_mape_loss(y_pred, y_true):
    mask = torch.gt(abs(y_true), 0.000000001)
    pred = torch.masked_select(y_pred, mask)
    true = torch.masked_select(y_true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def masked_rmse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())

def masked_mse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()

def calc_AR(y_pred, y_true, x_input, topk_indices, dict, device='cuda:0', top_k=20):
    # x_input.shape: [batch_size, seq_len, num_stocks, dim]
    mask1 = torch.gt(torch.tensor(x_input[:, -1, :, 4], device=device), 0.000001)
    y_true = y_true[:, -1, :, 4]
    mask2 = torch.gt(torch.tensor(y_true, device=device), 0.000001)
    y_pred = y_pred * mask1 * mask2
    y_pred = torch.where(abs(y_pred) < 0.00000001, -99999, y_pred)
    (point, ind) = torch.topk(y_pred, k=top_k, dim=1, largest=True)
    # print(point)
    ind = ind.detach().cpu().numpy()
    close_price = x_input[:, -1, :, 4]
    ratio_list = []
    for batch in range(y_pred.shape[0]):
        cl_p = close_price[batch]
        if torch.sum(point[batch]) < -999:
            ratio = 0
            if len(topk_indices) > 0:
                for k in range(len(topk_indices)):
                    ratio = ratio + cl_p[topk_indices[k]] * dict[str(topk_indices[k])][1]
                ratio_list.append(ratio)
            else:
                ratio_list.append(100000)
            continue
        ratio = 0
        del_list = []
        if topk_indices == []:
            for i in range(len(ind[batch])):
                topk_indices.append(ind[batch][i])
                dict[str(ind[batch][i])][0] = cl_p[ind[batch][i]]
                dict[str(ind[batch][i])][1] = (10000.0 / (cl_p[ind[batch][i]] * (1+0.0015)))
            continue
        for k in range(top_k):
            if topk_indices[k] not in ind[batch]:
                ratio = ratio + cl_p[topk_indices[k]] * dict[str(topk_indices[k])][1] * (1-0.0025)
                dict[str(topk_indices[k])][0], dict[str(topk_indices[k])][1] = 0, 0
                del_list.append(topk_indices[k])
        for s in range(len(del_list)):
            topk_indices.remove(del_list[s])
        if len(del_list) > 0:
            for k in range(top_k):
                if ind[batch][k] not in topk_indices:
                    topk_indices.append(ind[batch][k])
                    dict[str(ind[batch][k])][0] = cl_p[ind[batch][k]]
                    dict[str(ind[batch][k])][1] = ratio / (len(del_list) * cl_p[ind[batch][k]] * (1+0.0015))
        ratio = 0
        for k in range(len(topk_indices)):
            ratio += (y_true[batch][topk_indices[k]] * dict[str(topk_indices[k])][1])
        ratio_list.append(ratio)
    return topk_indices, ratio_list, dict

def calc_AR_SR(ratio_list):
    daily_ratio = [0]
    free = np.power(1 + 0.0246, 1 / 365) - 1
    for i in range(1, len(ratio_list)):
        daily_ratio.append((ratio_list[i] - ratio_list[i - 1]) / ratio_list[i - 1])
    SR = np.sqrt(252) * ((np.mean(np.array(daily_ratio)) - free) / np.std(np.array(daily_ratio) - free))
    AR = (ratio_list[-1] - ratio_list[0]) / ratio_list[0]
    print('AR:{}  SR:{}'.format(AR, SR))
    return AR, SR