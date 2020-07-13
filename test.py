import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
from data_loader.data_one_domain import dataLoader
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import model.number_embedding as number_embedding
from model.number_embedding import resume_weight_init
import numpy as np
from model.transE import resume_weight_transE


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    # data_loader = config.initialize('test_loader', module_data)
    aq_pos = ['dongsi_aq', 'tiantan_aq', 'guanyuan_aq', 'wanshouxigong_aq', 'aotizhongxin_aq', 'nongzhanguan_aq',
              'wanliu_aq', 'fengtaihuayuan_aq', 'qianmen_aq', 'yongdingmennei_aq', 'xizhimenbei_aq',
              'nansanhuan_aq',
              'dongsihuan_aq']
    test_loader = []
    for aq in aq_pos:
        test_loader.append(dataLoader(attribute_feature=["PM25", "PM10"],
                                       label_feature=["PM25", "PM10"],
                                       station=aq,
                                       mode="test",
                                       path="/lfs1/users/jbyu/QA_inference/data/single.csv",
                                       predict_time_len=1,
                                       encoder_time_len=24,
                                       batch_size=128,
                                       num_workeres=1,
                                       shuffle=False))

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metrics_fn = getattr(module_metric, config['metrics'])

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # number_embedding_model = number_embedding.MLP(entityNum=1500, embeddingDim=50)
    # number_embedding_model = resume_weight_init(number_embedding_model)
    number_embedding_model = resume_weight_transE()
    number_model = number_embedding_model.to(device)

    total_loss = 0.0
    total_L_loss = 0
    total_R_loss = 0
    total_metrics = 0
    length = 0
    with torch.no_grad():
        for j in range(len(aq_pos)):
            for batch_idx, (feature, target, _) in enumerate(test_loader[j]):
                length += 1
                feature, target = feature.to(device), target.to(device)

                feature = number_model.entityEmbedding(feature)
                feature_L, feature_R = torch.chunk(feature, 2, dim=2)
                feature_L, feature_R = feature_L.squeeze(2), feature_R.squeeze(2)

                predict_LR = model(feature_L, feature_R)
                L_out, R_out = torch.chunk(predict_LR, 2, dim=1)

                target = target.squeeze(1)
                target_L, target_R = torch.chunk(target, 2, dim=1)
                loss_L = loss_fn(L_out, target_L)
                loss_R = loss_fn(R_out, target_R)
                loss = loss_L + loss_R
                # loss = self.loss(predict_LR, target)

                metrics = metrics_fn(predict_LR, target)
                total_metrics += metrics.item()
                total_loss += loss.item()
                total_R_loss += loss_R.item()
                total_L_loss += loss_L.item()

    log = {'rmse loss': np.sqrt(total_loss / length),
           'L rmse loss': np.sqrt(total_L_loss / length),
           'R rmse loss': np.sqrt(total_R_loss / length),
           'smap loss': total_metrics / length,
           }

    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    config = ConfigParser(args)
    main(config)
