import sys
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
from model2 import EMGModel2


class EMGDataset(data.Dataset):
    def __init__(self, session_paths, slices=None, ver=1):
        sessions = [pd.read_csv(p).to_numpy(dtype=np.float32) for p in session_paths]
        slices = [(0., 1.) for _ in sessions] if slices is None else slices
        self.ver = ver

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

        if ver == 2:
            self.x_mins = self.x_mins[:, None]
            self.x_maxs = self.x_maxs[:, None]
            self.y_mins = self.y_mins[:, None]
            self.y_maxs = self.y_maxs[:, None]

        self.nsession_samples = [x.shape[0] - 1000 + 1 for x in self.xs]

    def __len__(self):
        return sum([x.shape[0] - 1000 for x in self.xs])

    def __getitem__(self, idx):

        session_i = 0
        while idx - self.nsession_samples[session_i] + 1 >= 0:
            idx -= self.nsession_samples[session_i] - 1
            session_i += 1

        if VER == 1:
            x = self.xs[session_i][None, idx:idx + 1000, :]  # added channel dim
            y = self.ys[session_i][None, idx:idx + 1000, :]
        else:
            x = self.xs[session_i][idx:idx + 1000, :]  # added channel dim
            y = self.ys[session_i][idx:idx + 1000, :]
            x = torch.permute(x, (1, 0))
            y = torch.permute(y, (1, 0))

        return x, y


class TrialTask(pl.LightningModule):

    def __init__(self, model, y_mins, y_maxs, ver=1, **kwargs):
        super().__init__()
        self.ver = ver
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
        _, y = batch
        loss, med_error, perc90_error, y_hat = self._shared_eval_step(batch, batch_idx)
        metrics = {'val_loss': loss, 'val_med_error': med_error, 'val_90%ile_error': perc90_error}
        self.log_dict(metrics, prog_bar=True)

        if batch_idx == 4:
            i = np.random.randint(0, 9, 2)
            b = np.random.randint(0, y.shape[0] - 1, 2)

            if self.ver == 1:
                plt.figure()
                plt.plot(y[b[0], 0, :, i[0]].cpu().numpy(), label='y')
                plt.plot(y_hat[b[0], 0, :, i[0]].cpu().numpy(), label='y_hat')

                plt.figure()
                plt.plot(y[b[1], 0, :, i[1]].cpu().numpy(), label='y')
                plt.plot(y_hat[b[1], 0, :, i[1]].cpu().numpy(), label='y_hat')
            else:
                plt.figure()
                plt.plot(y[b[0], i[0], :].cpu().numpy(), label='y')
                plt.plot(y_hat[b[0], i[0], :].cpu().numpy(), label='y_hat')

                plt.figure()
                plt.plot(y[b[1], i[1], :].cpu().numpy(), label='y')
                plt.plot(y_hat[b[1], i[1], :].cpu().numpy(), label='y_hat')

            plt.legend()
            plt.show()

        return metrics

    def test_step(self, batch, batch_idx):
        loss, med_error, perc90_error, _ = self._shared_eval_step(batch, batch_idx)
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

        return loss, med_error, perc90_error, y_hat

    def _scale_back(self, x):
        return x * (self.y_maxs - self.y_mins) + self.y_mins  # reverse of (x - min) / (max - min)

    def _loss(self, y_hat, y):
        fingers = F.mse_loss(y_hat, y, reduction='mean')
        vel = torch.diff(y_hat, dim=2)  # batch x channel x _t_ x joint
        smoothness = torch.mean(torch.pow(torch.diff(vel, dim=2), 2))
        self.log('fingers', fingers, prog_bar=True)
        self.log('smoothness', smoothness, prog_bar=True)

        loss = fingers + 25 * smoothness
        if self.ver == 2:
            loss += 0.1 * model.kld_loss

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return self._scale_back(y_hat)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {'optimizer': opt,
                'lr_scheduler': {
                    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, verbose=True,
                                                                            threshold=1e-3, factor=.1, mode='min'),
                    'monitor': 'val_med_error',
                    'interval': 'epoch',
                }}


# TODO try layernorm, gelu

# notes
# res t kernel: 5 > 3
# dropout: .1 > .5
# res bottleneck > no bottleneck
# nresblocks 3 ~ 1 ~ 7;


if __name__ == '__main__':

    VER = 1

    session_path = lambda i: f'data/Session {i}.csv'

    val_sessions = [6]
    train_dataset = EMGDataset([session_path(i) for i in range(1, 7)
                                if i not in val_sessions], ver=VER)
    val_dataset = EMGDataset([session_path(vs) for vs in val_sessions], ver=VER)

    # res_chans = [256] + [256, 512, 256, 256, 128, 256, 256, 512, 256, 256] * 3
    # res_kernel = [3, 5] * 15
    # res_chans = [256] + [256, 512, 256, 256, 256, 256, 256, 512, 256, 256] * 2
    # res_kernel = [3, 5] * 10

    model_settings = {
        'dropout': 0.4,
        'res_block_params': [(256, 256, (3, 1)) for _ in range(35)],
            # [(res_chans[i - 1], res_chans[i], (res_kernel[i - 1], 1)) for i in range(1, len(res_chans))],
            #[(256, 256, (5, 1)) for _ in range(10)],
            #[(256, 256, (5, 1)), (256, 512, (5, 1)), (512, 256, (5, 1)), (256, 256, (5, 1))],
            #[(256, 256, (3, 1)), (256, 128, (3, 1)), (128, 64, (3, 1)), (64, 128, (3, 1)), (128, 256, (3, 1))],
    }

    train_settings = {
        'lr': 0.0005,  # TODO check
    }
    batch_size = 256  # TODO
    epochs = 50

    print('batch', batch_size)
    print('model settings:', model_settings, file=sys.stderr)
    print('train settings:', train_settings, file=sys.stderr)

    loader_settings = {'batch_size': batch_size, 'num_workers': 2, 'prefetch_factor': 8}
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_settings)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_settings)
    # test_loader = torch.utils.data.DataLoader(test_dataset, **loader_settings)

    prefix = f'ver{VER}'
    model_name = f'model_{prefix}_batch-{batch_size}_{datetime.datetime.now().strftime("%H-%M-%S")}_dropout-{model_settings["dropout"]}_' \
                 f'res-{len(model_settings["res_block_params"])}-{model_settings["res_block_params"][0][0]}-{model_settings["res_block_params"][0][2]}'

    resume_from_checkpoint = 'checkpoints/model_gradclip_batch-256_16-07-22_dropout-0.2_res-35-256-(3, 1)/epoch=19-step=14879.ckpt'  # or None

    print('MODEL:', model_name)

    model = EMGModel2(**model_settings) if VER == 2 else EMGModel(**model_settings)
    task = TrialTask(model, train_dataset.y_mins, train_dataset.y_maxs, ver=VER, **train_settings)

    logger = pl_loggers.CSVLogger('logs/', f'{model_name}_logs.csv')  # pl_loggers.TensorBoardLogger("logs/")

    callbacks = [
        pl.callbacks.EarlyStopping('val_med_error', patience=5, verbose=True, check_on_train_epoch_end=True,
                                   min_delta=1e-3, mode='min'),
        pl.callbacks.ModelCheckpoint(f'checkpoints/{model_name}', mode='min', monitor='val_med_error',
                                     verbose=True, save_last=False, save_top_k=1)
    ]

    trainer = pl.trainer.Trainer(gpus=1, callbacks=callbacks, max_epochs=epochs, enable_progress_bar=True,
                                 log_every_n_steps=100, resume_from_checkpoint=resume_from_checkpoint,
                                 limit_train_batches=1.0, logger=logger, gradient_clip_val=2)
    trainer.fit(task, train_loader, val_loader)
    trainer.save_checkpoint(f'models/{model_name}.pt')
    print(model_name)
