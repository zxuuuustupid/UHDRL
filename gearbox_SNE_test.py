# --------------
# --------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
# -------------------------------------
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import argparse
import random

from src.efficient_kan import KAN
from src import fullconnect
import CNNEncoder1
import vit
from src.fullconnect import FullyConnectedLayer
from tripletloss.tripletloss import TripletLoss

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=128)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=2)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=1)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=4)
parser.add_argument("-e", "--episode", type=int, default=2000)
parser.add_argument("-t", "--test_episode", type=int, default=10)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("-c", "--cpu", type=int, default=0)
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

train_result = './train_result/'

label_list = ['Health', 'anomaly']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())




def main():
    feature_encoder = CNNEncoder1.rsnet()
    fc = FullyConnectedLayer()
    feature_encoder.cuda(GPU)
    fc.cuda(GPU)
    if os.path.exists(
            str("./models/gearbox_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(
            str("./models/gearbox_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load gearbox feature encoder success")
    print("init data folders")

    # 优化器的定义
    # feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    # feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    fc_optim = torch.optim.Adam(fc.parameters(), lr=LEARNING_RATE)
    fc_scheduler = StepLR(fc_optim, step_size=100000, gamma=0.5)



    # TODO(1.19):add a api to get features

    # Step 3: build graph
    print("Training...")
    ticks = 1
    arch128_all=[]
    health128_all=[]
    anomaly128_all=[]
    last_accuracy = 0
    for episode in range(EPISODE):
        # print('episode', episode)
        # feature_encoder_scheduler.step(episode)
        fc_scheduler.step(episode)
        degrees = random.choice([0, 90, 180, 270])
        #########################################################
        # triplet_num=random.randint(1,9)
        triplet_num=1
        # health_character_folders_1 = [f'../CWT-1000/gearbox/train/health/WC{triplet_num}',
        #                               '../CWT-1000/gearbox/train/health']
        health_character_folders_1 = [f'../CWT3-1000/gearbox/train/health/WC{triplet_num}',
                                      '../CWT3-1000/gearbox/arch/health']
        # arch_character_folders_1 = [f'../CWT-1000/gearbox/train/health/WC{triplet_num}',
        #                             '../CWT-1000/gearbox/train/health']
        arch_character_folders_1 = [f'../CWT3-1000/gearbox/train/health/WC{triplet_num}',
                                    '../CWT3-1000/gearbox/arch/health']
        random_folder = random.choice([f.path for f in os.scandir('../CWT3-1000/gearbox/arch/anomaly') if f.is_dir()])
        anomaly_character_folders_1 = [random_folder,
                                       '../CWT3-1000/gearbox/arch/anomaly']
        task_health = tg.OmniglotTask(health_character_folders_1, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        batch_dataloader_health = tg.get_data_loader(task_health, num_per_class=BATCH_NUM_PER_CLASS, split="test",
                                                     shuffle=True, rotation=degrees)
        batch_dataloader_health = iter(batch_dataloader_health)
        batches_health, batch_labels_health = next(batch_dataloader_health)
        batch_features_health = feature_encoder(Variable(batches_health).cuda(GPU))
        task_arch = tg.OmniglotTask(arch_character_folders_1, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        batch_dataloader_arch = tg.get_data_loader(task_arch, num_per_class=BATCH_NUM_PER_CLASS, split="test",
                                                   shuffle=True,
                                                   rotation=degrees)
        batch_dataloader_arch = iter(batch_dataloader_arch)
        batches_arch, batch_labels_arch = next(batch_dataloader_arch)
        batch_features_arch = feature_encoder(Variable(batches_arch).cuda(GPU))

        task_anomaly = tg.OmniglotTask(anomaly_character_folders_1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
                                       BATCH_NUM_PER_CLASS)
        batch_dataloader_anomaly = tg.get_data_loader(task_anomaly, num_per_class=BATCH_NUM_PER_CLASS, split="test",
                                                      shuffle=True,
                                                      rotation=degrees)
        batch_dataloader_anomaly = iter(batch_dataloader_anomaly)
        batches_anomaly, batch_labels_anomaly = next(batch_dataloader_anomaly)
        batch_features_anomaly = feature_encoder(Variable(batches_anomaly).cuda(GPU))

        batch_features_arch = fc(batch_features_arch)
        batch_features_health = fc(batch_features_health)
        batch_features_anomaly = fc(batch_features_anomaly)
        '''change 0.1 to 1'''
        triloss = TripletLoss(margin=1000)

        loss_punish = triloss(batch_features_arch, batch_features_health, batch_features_anomaly)
        """Connect to line 112"""
        if episode>100:
            if loss_punish==0 and ticks < 24 and triplet_num==1:
                arch128 = [x_1.cpu().detach().numpy() for x_1 in batch_features_arch]
                health128=[x_2.cpu().detach().numpy() for x_2 in batch_features_health]
                anomaly128 = [x_3.cpu().detach().numpy() for x_3 in batch_features_anomaly]
                ticks=ticks+1
                arch128_all.extend(arch128)
                health128_all.extend(health128)
                anomaly128_all.extend(anomaly128)
                if ticks==24:
                    arch128_all = np.array(arch128_all)
                    health128_all = np.array(health128_all)
                    anomaly128_all = np.array(anomaly128_all)

                    # 保存数据到CSV文件
                    train_result = 'train_result/'  # 假设这是你的保存路径
                    np.savetxt(train_result +'output_features/'+ 'gearbox_arch.csv', arch128_all, fmt='%.8f', delimiter=',')
                    np.savetxt(train_result+'output_features/' + 'gearbox_health.csv', health128_all, fmt='%.8f', delimiter=',')
                    np.savetxt(train_result+'output_features/' + 'gearbox_anomaly.csv', anomaly128_all, fmt='%.8f', delimiter=',')
        #########################################################

        loss=loss_punish
        loss.backward()
        # torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(fc.parameters(), 0.5)
        # feature_encoder_optim.step()
        fc_optim.step()
    return 0


if __name__ == '__main__':
    loos_result_1 = main()
    loos_result_1_cpu = [x_1.cpu().detach().numpy() for x_1 in loos_result_1]
    # np.savetxt(train_result + 'gearbox_train_loss.csv', loos_result_1_cpu, fmt='%.8f', delimiter=',')
