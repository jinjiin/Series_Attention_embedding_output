import argparse
import collections
import torch
import data_loader.data_loaders as module_data
# import data_loader.data_loader_embed as module_data
import model.loss as module_loss

import model.metric as module_metric
import model.model as module_arch
# import model.model_future_mask sas module_arch
# from model.model import weights_init
from parse_config import ConfigParser
# from trainer import Trainer
from trainer.trainer_mse import Trainer


def main(config):
    logger = config.get_logger('train')
    data_loader = config.initialize('data_loader', module_data)
    valid_data_loader = config.initialize('valid_loader', module_data)

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    logger.info(model)
    # model.apply(weights_init)

    # number_embedding_model = resume_weight_transE()
    # print('number_embedding_model parameter')
    # for p in number_embedding_model.parameters():
    #     print(p.requires_grad)
    # print('model parameters')
    # for p in model.parameters():
    #     print(p.requires_grad)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = getattr(module_metric, config['metrics'])

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # print('training params ', trainable_params)
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)
    # optimizer.add_param_group({'params': number_embedding_model.parameters()})

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler
                      )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)
    main(config)
