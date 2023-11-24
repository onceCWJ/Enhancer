# Enhancer: Model Enhancement against Temporal-Relational Distribution Shifts for Stock Prediction
## Requirements
- Python=3.8.1
- Pytorch=1.12.1+cu116
- Numpy=1.23.5
- Pandas=1.5.3

## Data Preparation

#### Rank Task
Download NASDAQ, NYSE datasets from [https://disk.pku.edu.cn:443/link/FE8FB4665C8C7A117FB14E26F35035A5]. Put into the `Enhancer/data_{NASDAQ/NYSE}` folder.

#### Forecasting Task

Download CSI300,CSI500 datasets from [https://disk.pku.edu.cn:443/link/FE8FB4665C8C7A117FB14E26F35035A5]. Put into the `Enhancer/data/{CSI300,CSI500}` folder.

## Split dataset

Run the following commands to generate train/validation/test dataset at `data/{CSI300,CSI500}/{train,val,test}.npz`.

```
python generate_data.py --dataset CSI300 --train_rate (2/3) --val_rate (1/6)

python generate_data.py --dataset CSI500 --train_rate (2/3) --val_rate (1/6)

```

## Train Commands

### Rank Task

python train_rank.py --dataset NASDAQ 
python train_rank.py --dataset NYSE

### Forecast Task

python train_forecast.py --dataset_dir data/CSI300 --model_type GRU
python train_forecast.py --dataset_dir data/CSI500 --model_type GRU