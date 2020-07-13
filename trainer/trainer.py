import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop
from data_loader.data_one_domain import dataLoader
import copy


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader, valid_data_loader, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_loader = valid_data_loader

        # self.number_model = number_model.to(self.device)

        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            self.len_epoch = len_epoch

        self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(self.len_epoch))
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def similar_word(self, entity, word_embedding):
        # entity:(bs, 1500, 50)
        # word_embedding:(bs, 50, 1)

        word_embedding = torch.transpose(word_embedding, 1, 2)
        cos_dis = entity.bmm(word_embedding)
        cos_dis = cos_dis.squeeze(1)
        p = F.softmax(cos_dis, dim=1)
        value_out = torch.argmax(p, dim=1)
        return p, value_out


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """

        self.model.train()
        total_loss = 0
        total_pm25_loss = 0
        total_pm10_loss = 0
        total_pm25_metrics = 0
        total_pm10_metrics = 0
        length = len(self.data_loader)


        for batch_idx, (pm25, pm10, pm25_target, pm10_target) in enumerate(self.data_loader):
            pm25, pm10, pm25_target, pm10_target = pm25.to(self.device), pm10.to(self.device), pm25_target.to(self.device), pm10_target.to(self.device)

            self.optimizer.zero_grad()
            entity = self.model.number_embed.weight
            entity = entity.expand(pm25.shape[0], 1500, 50)

            pm25_predict_embed, pm10_predict_embed, pm25_target_embed, pm10_target_embed = self.model(
                pm25, pm10, pm25_target, pm10_target)
            pm25_predict_embed  = pm25_predict_embed.unsqueeze(1)
            pm10_predict_embed = pm10_predict_embed.unsqueeze(1)
            pm25_p, pm25_predict_value = self.similar_word(entity, pm25_predict_embed)
            pm10_p, pm10_predict_value = self.similar_word(entity, pm10_predict_embed)

            pm25_cross_entropy = self.CrossEntropyLoss(pm25_p, pm25_target)
            pm10_cross_entropy = self.CrossEntropyLoss(pm10_p, pm10_target)
            mse_loss = pm25_cross_entropy + pm10_cross_entropy

            pm25_predict_value, pm25_target, pm10_predict_value, pm10_target = pm25_predict_value.float(), \
                                                                   pm25_target.float(), \
                                                                   pm10_predict_value.float(), \
                                                                   pm10_target.float()
            # print(pm25_predict_value==pm25_target, pm10_predict_value==pm10_target)

            pm25_loss = self.loss(pm25_predict_value, pm25_target)
            pm10_loss = self.loss(pm10_predict_value, pm10_target)
            # mse_loss = pm25_loss + pm10_loss

            l2_reg = torch.tensor(0.0).to(self.device)
            if self.config['trainer']['l2_regularization']:
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, p=2)
                loss = mse_loss + self.config['trainer']['l2_lambda'] * l2_reg
            else:
                loss = mse_loss

            loss.backward()
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=self.config['trainer']['clip_max_norm'])
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss.item())

            pm25_metrics = self.metrics(pm25_predict_value, pm25_target)
            pm10_metrics = self.metrics(pm10_predict_value, pm10_target)

            total_pm25_metrics += pm25_metrics.item()
            total_pm10_metrics += pm10_metrics.item()
            total_pm25_loss += pm25_loss.item()
            total_pm10_loss += pm10_loss.item()
            total_loss += mse_loss.item()

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Loss_L: {:.6f}  Loss_R: {:.6f} l2_reg: {:.6f} pm25_smape: {:.6f}, pm10_smape: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    pm25_loss,
                    pm10_loss,
                    l2_reg.item(),
                    pm25_metrics.item(),
                    pm10_metrics.item()
                ))

        log = {
            'pm25_rmse loss: ': np.sqrt(total_pm25_loss / length),
            'pm10_rmse loss: ': np.sqrt(total_pm10_loss / length),
            'total_rmse loss': np.sqrt(total_loss / length),
            'pm25_smape: ':total_pm25_metrics / length,
            'pm10_smape: ': total_pm10_metrics / length,
        }

        if self.do_validation:
            print('do_validation')
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_log['val_rmse_loss'])

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        total_loss = 0
        total_pm25_loss = 0
        total_pm10_loss = 0
        total_pm25_metrics = 0
        total_pm10_metrics = 0
        length = len(self.valid_loader)


        with torch.no_grad():
            for batch_idx, (pm25, pm10, pm25_target, pm10_target) in enumerate(self.valid_loader):
                pm25, pm10, pm25_target, pm10_target = pm25.to(self.device), \
                                                       pm10.to(self.device), \
                                                       pm25_target.to(self.device), \
                                                       pm10_target.to(self.device)
                entity = self.model.number_embed.weight
                entity = entity.expand(pm25.shape[0], 1500, 50)
                pm25_predict_embed, pm10_predict_embed, pm25_target_embed, pm10_target_embed = self.model(
                    pm25, pm10, pm25_target, pm10_target)
                pm25_predict_embed = pm25_predict_embed.unsqueeze(1)
                pm10_predict_embed = pm10_predict_embed.unsqueeze(1)
                pm25_p, pm25_predict_value = self.similar_word(entity, pm25_predict_embed)
                pm10_p, pm10_predict_value = self.similar_word(entity, pm10_predict_embed)

                pm25_cross_entropy = self.CrossEntropyLoss(pm25_p, pm25_target)
                pm10_cross_entropy = self.CrossEntropyLoss(pm10_p, pm10_target)
                mse_loss = pm25_cross_entropy + pm10_cross_entropy

                pm25_predict_value, pm25_target, pm10_predict_value, pm10_target = pm25_predict_value.float(),\
                                                                       pm25_target.float(), \
                                                                       pm10_predict_value.float(), \
                                                                       pm10_target.float()


                pm25_loss = self.loss(pm25_predict_value, pm25_target)
                pm10_loss = self.loss(pm10_predict_value, pm10_target)
                # mse_loss = pm25_loss + pm10_loss

                self.writer.set_step((epoch - 1) * length + batch_idx, 'valid')
                self.writer.add_scalar('loss', mse_loss.item())

                pm25_metrics = self.metrics(pm25_predict_value, pm25_target)
                pm10_metrics = self.metrics(pm25_predict_value, pm10_target)

                total_pm25_metrics += pm25_metrics.item()
                total_pm10_metrics += pm10_metrics.item()
                total_pm25_loss += pm25_loss.item()
                total_pm10_loss += pm10_loss.item()
                total_loss += mse_loss.item()

        print('valid data length: ', length)
        return {
            'val_pm25_rmse_loss': np.sqrt(total_pm25_loss / length),
            'val_pm10_rmse_loss': np.sqrt(total_pm10_loss / length),
            'val pm25 smape loss': total_pm25_metrics / length,
            'val pm10 smape loss': total_pm10_metrics / length,
            'val_rmse_loss': np.sqrt((total_pm25_loss+total_pm10_loss) / length)
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'

        current = batch_idx
        total = self.len_epoch * len(self.data_loader)
        return base.format(current, total, 100.0 * current / total)
