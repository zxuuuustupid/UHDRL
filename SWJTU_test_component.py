# --------------
# --------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
# -------------------------------------
import gc

import pandas as pd
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
parser.add_argument("-t", "--test_episode", type=int, default=100)
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

test_result = './test_result/'

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
    loos_result = []
    loos_result_1 = []
    accuray_result_1_1 = []
    # Step 1: init data folders
    print("init data folders")
    print("init neural networks")
    feature_encoder = CNNEncoder1.rsnet()  # 特征提取
    relation_network = KAN([28 * 28, 128, 8])  # 定义关系网络
    relation_network_2 = KAN([8 * 512, 512, 32, 2])
    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)
    relation_network_2.cuda(GPU)

    if os.path.exists(
            str("./models1_SWJTU/feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(
            str("./models1_SWJTU/feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(
            str("./models1_SWJTU/relation_network_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(
            str("./models1_SWJTU/relation_network_" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load relation network success")
    if os.path.exists(
            str("./models1_SWJTU/relation_network_2" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        relation_network_2.load_state_dict(torch.load(
            str("./models1_SWJTU/relation_network_2" + str(CLASS_NUM) + "way_" + str(
                SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load relation network2 success")
        # Step 3: build graph
    accuracy_list = [[0] * 5 for _ in range(6)]
    for num_fault_type in ['7-Inner+Outer', '8-Inner+Roller']:
        for num_wc in range(1, 5 + 1):
            if num_wc==1 and num_fault_type =='7-Inner+Outer':
                continue
            total_acc=0
            for ten_epoches in range(1, 11):
                total_rewards = 0
                for i in range(TEST_EPISODE):
                    num_wc_mixed=random.randint(1,5)
                    degrees = random.choice([0, 90, 180, 270])
                    metatest_character_folders1 = [f'../CWT-XJT/test/health/WC{num_wc}',
                                                   f'../CWT-XJT/test/anomaly/{num_fault_type}/WC{num_wc}']
                    metatrain_character_folders1 = [f'../CWT-XJT/train/health/WC{num_wc_mixed}',
                                                    '../CWT-XJT/train/anomaly']
                    task = tg.OmniglotTask(metatest_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
                                           SAMPLE_NUM_PER_CLASS, )
                    task1 = tg.OmniglotTask(metatrain_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
                                            BATCH_NUM_PER_CLASS)
                    sample_dataloader = tg.get_data_loader(task1, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
                                                           shuffle=False, rotation=degrees)
                    test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test",
                                                         shuffle=True, rotation=degrees)
                    sample_dataloader = iter(sample_dataloader)
                    sample_images, sample_labels = next(sample_dataloader)
                    test_dataloader = iter(test_dataloader)
                    test_images, test_labels = next(test_dataloader)
                    sample_features = feature_encoder(Variable(sample_images).cuda(GPU))  # 5x64
                    test_features = feature_encoder(Variable(test_images).cuda(GPU))  # 20x64
                    sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1,
                                                                              1)
                    test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
                    test_features_ext = torch.transpose(test_features_ext, 0, 1)
                    relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1,
                                                                                                 FEATURE_DIM * 4,
                                                                                                 28 * 28)

                    relations1 = relation_network(relation_pairs)
                    relations1 = relations1.view(2, 8 * 512)
                    relations1 = relation_network_2(relations1)
                    relations = relations1.view(-1, CLASS_NUM)
                    # print(relations.shape)
                    bb = Variable(torch.zeros(CLASS_NUM)).cuda(GPU)

                    # print(relations)
                    for j in range(len(relations)):
                        if relations[j][0] > 0.9:
                            bb[j] = 0
                        else:
                            bb[j] = 1
                    predict_labels = bb.cpu()
                    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM)]
                    total_rewards += np.sum(rewards)
                accuracy = total_rewards / 1.0 / CLASS_NUM / TEST_EPISODE
                total_acc = total_acc + accuracy
            print(str(num_fault_type).ljust(2),
                  "WorkCondition:", str(num_wc).ljust(2),
                  "   Accuracy:", f"{total_acc / 10.0:.4f}".ljust(6),)
            accuracy_list[int(num_fault_type[0]) - 7][num_wc - 1] = total_acc / 10.0
    return accuracy_list


if __name__ == '__main__':
    acc_data= main()
    df_acc = pd.DataFrame(acc_data)
    # 构建目标文件路径
    # file_path_acc = os.path.join('test_result', 'SWJTU','mixed','acc.csv')
    # df_acc.to_csv(file_path_acc, index=False, header=False)
