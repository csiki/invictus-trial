import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils import data
import datetime
from pytorch_lightning import loggers as pl_loggers

from model import EMGModel


class EMGDataset(data.Dataset):
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

        self.nsession_samples = [x.shape[0] - 1000 + 1 for x in self.xs]

    def __len__(self):
        return sum([x.shape[0] - 1000 for x in self.xs])

    def __getitem__(self, idx):

        session_i = 0
        while idx - self.nsession_samples[session_i] + 1 >= 0:
            idx -= self.nsession_samples[session_i] - 1
            session_i += 1

        x = self.xs[session_i][None, idx:idx + 1000, :]  # added channel dim
        y = self.ys[session_i][None, idx:idx + 1000, :]

        return x, y


class TrialTask(pl.LightningModule):

    def __init__(self, model, y_mins, y_maxs, **kwargs):
        super().__init__()
        self.model = model
        self.y_mins, self.y_maxs = y_mins.cuda(), y_maxs.cuda()

        self.lr = kwargs['lr']

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self._loss(y_hat, y)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, med_error, perc90_error = self._shared_eval_step(batch, batch_idx)
        metrics = {'val_loss': loss, 'val_med_error': med_error, 'val_90%ile_error': perc90_error}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, med_error, perc90_error = self._shared_eval_step(batch, batch_idx)
        metrics = {'test_loss': loss, 'test_med_error': med_error, 'test_90%ile_error': perc90_error}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self._loss(y_hat, y)

        degree_error = torch.abs(self._scale_back(y) - self._scale_back(y_hat))
        med_error = torch.median(degree_error)
        perc90_error = torch.quantile(degree_error, .9)

        return loss, med_error, perc90_error

    def _scale_back(self, x):
        return x * (self.y_maxs - self.y_mins) + self.y_mins  # reverse of (x - min) / (max - min)

    def _loss(self, y_hat, y):
        fingers = F.mse_loss(y_hat, y)
        vel = torch.diff(y_hat, dim=2)  # batch x channel x _t_ x joint
        smoothness = torch.mean(torch.pow(torch.diff(vel, dim=2), 2))
        return fingers + smoothness

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {'optimizer': opt,
                'lr_scheduler': {
                    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, verbose=True,
                                                                            threshold=1e-3, factor=.2),
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                }}


# TODO try layernorm, gelu


if __name__ == '__main__':

    session_path = lambda i: f'data/Session {i}.csv'

    #train_sessions = [1]
    val_sessions = [5]  #3
    test_sessions = []  #4

    #train_slices = [(0., 0.7)]
    #val_slices = [(0.7, 1.0)]

    #train_dataset = EMGDataset([session_path(ts) for ts in train_sessions], train_slices)
    train_dataset = EMGDataset([session_path(i) for i in range(1, 6)
                                if i not in val_sessions + test_sessions])
    val_dataset = EMGDataset([session_path(vs) for vs in val_sessions])
    # test_dataset = EMGDataset([session_paths[test_session]])

    model_settings = {
        'dropout': 0.3,
        'res_block_params': [(256, 256, (5, 1)) for _ in range(5)],
    }

    train_settings = {
        'lr': 0.001,
    }
    batch_size = 128
    epochs = 50

    loader_settings = {'batch_size': batch_size, 'num_workers': 2, 'prefetch_factor': 8}
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_settings)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_settings)
    # test_loader = torch.utils.data.DataLoader(test_dataset, **loader_settings)

    prefix = f''
    model_name = f'model_{prefix}_{datetime.datetime.now().strftime("%H-%M-%S")}_dropout-{model_settings["dropout"]}_' \
                 f'res-{len(model_settings["res_block_params"])}-{model_settings["res_block_params"][0][0]}-{model_settings["res_block_params"][0][2]}'

    model = EMGModel(**model_settings)
    task = TrialTask(model, train_dataset.y_mins, train_dataset.y_maxs, **train_settings)

    logger = pl_loggers.CSVLogger('logs/', f'{model_name}_logs.csv')  # pl_loggers.TensorBoardLogger("logs/")

    callbacks = [
        pl.callbacks.EarlyStopping('val_loss', patience=5, verbose=True, check_on_train_epoch_end=True, min_delta=1e-3),
        pl.callbacks.ModelCheckpoint(f'checkpoints/{model_name}',
                                     monitor='val_loss', verbose=True, save_last=False, save_top_k=1)
    ]

    trainer = pl.trainer.Trainer(gpus=1, callbacks=callbacks, max_epochs=epochs, enable_progress_bar=True,
                                 log_every_n_steps=50, resume_from_checkpoint=None, limit_train_batches=1.0,
                                 logger=logger)
    trainer.fit(task, train_loader, val_loader)