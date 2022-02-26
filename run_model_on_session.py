import sys
from glob import glob
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils import data
import datetime
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt

from model import EMGModel
from train import *


class EMGTestDataset(data.Dataset):
    def __init__(self, session_paths, slices=None):
        sessions = [pd.read_csv(p).to_numpy(dtype=np.float32) for p in session_paths]
        slices = [(0., 1.) for _ in sessions] if slices is None else slices

        self.xs = [torch.tensor(sess[int(sli[0] * sess.shape[0]):int(sli[1] * sess.shape[0]), :8])
                   for sess, sli in zip(sessions, slices)]
        self.ys = [torch.tensor(sess[int(sli[0] * sess.shape[0]):int(sli[1] * sess.shape[0]), 8:])
                   for sess, sli in zip(sessions, slices)]

        # min-max normalize
        self.x_mins = torch.min(torch.cat(self.xs, dim=0), dim=0).values
        self.x_maxs = torch.max(torch.cat(self.xs, dim=0), dim=0).values
        self.y_mins = torch.min(torch.cat(self.ys, dim=0), dim=0).values
        self.y_maxs = torch.max(torch.cat(self.ys, dim=0), dim=0).values

        self.xs = [(x - self.x_mins) / (self.x_maxs - self.x_mins) for x in self.xs]
        self.ys = [(y - self.y_mins) / (self.y_maxs - self.y_mins) for y in self.ys]

        self.nsession_samples = [x.shape[0] // 1000 for x in self.xs]

    def __len__(self):
        return sum([x.shape[0] // 1000 for x in self.xs])

    def __getitem__(self, idx):

        session_i = 0
        while idx - self.nsession_samples[session_i] >= 0:
            idx -= self.nsession_samples[session_i] - 1
            session_i += 1

        x = self.xs[session_i][None, idx * 1000:idx * 1000 + 1000, :]  # added channel dim
        y = self.ys[session_i][None, idx * 1000:idx * 1000 + 1000, :]

        return x, y


if __name__ == '__main__':

    session_path = lambda i: f'data/Session {i}.csv'

    # train_sessions = [1]
    val_sessions = [6]  # 3
    test_sessions = []  # 4

    # train_slices = [(0., 0.7)]
    # val_slices = [(0.7, 1.0)]

    # train_dataset = EMGDataset([session_path(ts) for ts in train_sessions], train_slices)
    train_dataset = EMGDataset([session_path(i) for i in range(1, 7)
                                if i not in val_sessions + test_sessions])
    val_dataset = EMGTestDataset([session_path(vs) for vs in val_sessions])
    # test_dataset = EMGDataset([session_paths[test_session]])

    res_chans = [256] + [128, 256, 512] * 10
    res_kernel = [3, 5] * 15

    model_settings = {
        'dropout': 0.2,
        'res_block_params': [(256, 256, (3, 1)) for _ in range(35)],
        # [(res_chans[i - 1], res_chans[i], (res_kernel[i], 1)) for i in range(1, len(res_chans))],
        # [(256, 256, (5, 1)) for _ in range(10)],
        # [(256, 256, (5, 1)), (256, 512, (5, 1)), (512, 256, (5, 1)), (256, 256, (5, 1))],
        # [(256, 256, (3, 1)), (256, 128, (3, 1)), (128, 64, (3, 1)), (64, 128, (3, 1)), (128, 256, (3, 1))],
    }

    train_settings = {
        'lr': 0.001,  # TODO check
    }
    batch_size = 128
    epochs = 50

    print('batch', batch_size)
    print('model settings:', model_settings, file=sys.stderr)
    print('train settings:', train_settings, file=sys.stderr)

    loader_settings = {'batch_size': 1, 'num_workers': 2, 'prefetch_factor': 8}
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_settings)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_settings)
    # test_loader = torch.utils.data.DataLoader(test_dataset, **loader_settings)

    model_name = 'model_ver1_batch-256_20-45-32_dropout-0.2_res-35-256-(3, 1)'
    checkpoint_path = f'checkpoints/{model_name}'
    # 'model__batch-256_01-12-11_dropout-0.2_res-10-256-(5, 1)'

    print('MODEL:', model_name)

    model = EMGModel(**model_settings).cuda()
    # state = torch.load('models/model_gelu-bilinear-notbottle_batch-128_00-06-26_dropout-0.2_res-10-256-(5, 1).pt')
    # model.load_state_dict(state)

    checkpoint_path = list(glob(f'{checkpoint_path}/*.ckpt'))[0]
    model_checkpoint_path = f'models/{model_name}.pt'

    # task = TrialTask(model, train_dataset.y_mins, train_dataset.y_maxs, **train_settings)
    # task.load_from_checkpoint(list(glob(f'{checkpoint_path}/*.ckpt'))[0], model=model, y_mins=train_dataset.y_mins,
    #                           y_maxs=train_dataset.y_maxs, **train_settings).cuda()

    task = TrialTask.load_from_checkpoint(checkpoint_path,
                                          model=model, y_mins=train_dataset.y_mins,  # TODO try None !!!
                                          y_maxs=train_dataset.y_maxs, **train_settings).cuda()

    df_data_y, df_data_y_hat = [], []

    print('validation set', len(val_loader))

    for i, (x, y) in enumerate(val_loader):
        x = x.cuda()
        y = y.cuda()

        with torch.no_grad():
            # y_hat =  model(x)
            y_hat = task.predict_step((x, y), i)

        y = torch.squeeze(task._scale_back(y))
        y_hat = torch.squeeze(y_hat)

        for j in range(y.shape[0]):
            df_data_y.append(y[j, :].cpu().numpy())
            df_data_y_hat.append(y_hat[j, :].cpu().numpy())

        print(f'{i}.', end='')

    df_data_y = np.array(df_data_y)
    df_data_y_hat = np.array(df_data_y_hat)

    for i in range(10):
        plt.figure()
        plt.plot(df_data_y[3000:7000, i], label='y')  # [1000:2000, i]
        plt.plot(df_data_y_hat[3000:7000, i], label='y_hat')
        plt.legend()
        plt.savefig(f'results/{i}.png')
        plt.show()

    # print loss
    metrics = pd.read_csv(f'logs/{model_name}_logs.csv/version_0/metrics.csv')
    epoch = metrics['epoch'].loc[~pd.isna(metrics['loss'])]
    val_epoch = metrics['epoch'].loc[~pd.isna(metrics['val_loss'])]
    loss = metrics['loss'].loc[~pd.isna(metrics['loss'])]
    val_loss = metrics['val_loss'].loc[~pd.isna(metrics['val_loss'])]
    med_err = metrics['val_med_error'].loc[~pd.isna(metrics['val_loss'])]
    perc90_err = metrics['val_90%ile_error'].loc[~pd.isna(metrics['val_loss'])]

    plt.figure()
    plt.plot(epoch, loss, label='loss')
    plt.plot(val_epoch, val_loss, label='val_loss')
    plt.legend()

    plt.figure()
    plt.plot(med_err, label='med err')
    plt.plot(perc90_err, label='90%ile err')
    plt.legend()

    plt.show()

    act_df_data_y = {f'{i}_y': df_data_y[:, i] for i in range(10)}
    act_df_data_y_hat = {f'{i}_y^': df_data_y_hat[:, i] for i in range(10)}
    act_df = {f'y_{i//2}' if i % 2 == 0 else f'y^_{i//2}':
                  df_data_y[:, i//2] if i % 2 == 0 else df_data_y_hat[:, i//2] for i in range(20)}

    df = pd.DataFrame({**act_df}) # {**act_df_data_y, **act_df_data_y_hat})
    print(df)
    df.to_csv('results/result.csv', float_format='%.3f')

    exit()

    indices = np.array(sum([[i, i + 10] for i in range(10)], []))
    print(indices)

    # cols_y, cols_y_hat = [f'y_{i}' for i in range(10)], [f'y^_{i}' for i in range(10)]
    # columns = columns[::2] + columns[1::2]

    columns = [f'y{i // 2}' if i % 2 == 0 else f'y^{i // 2}' for i in range(20)]

    df = pd.DataFrame(df_data[:, indices], columns=columns)
    df.to_csv('result2.csv', float_format='%.3f')

    # print(df)
    print(df.shape)
