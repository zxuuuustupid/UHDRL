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

    # 定义化网络
    # 关系网络和特征提取模块的加载话
    #    feature_encoder.apply(weights_init)
    #   relation_network.apply(weights_init)
    gearbox_feature_encoder = CNNEncoder1.rsnet()  # 特征提取
    gearbox_relation_network = KAN([28 * 28, 128, 8])  # 定义关系网络
    gearbox_relation_network_2 = KAN([8 * 512, 512, 32, 2])
    motor_feature_encoder = CNNEncoder1.rsnet()  # 特征提取
    motor_relation_network = KAN([28 * 28, 128, 8])  # 定义关系网络
    motor_relation_network_2 = KAN([8 * 512, 512, 32, 2])
    leftaxlebox_feature_encoder = CNNEncoder1.rsnet()  # 特征提取
    leftaxlebox_relation_network = KAN([28 * 28, 128, 8])  # 定义关系网络
    leftaxlebox_relation_network_2 = KAN([8 * 512, 512, 32, 2])
    gearbox_feature_encoder.cuda(GPU)
    gearbox_relation_network.cuda(GPU)
    gearbox_relation_network_2.cuda(GPU)
    leftaxlebox_feature_encoder.cuda(GPU)
    leftaxlebox_relation_network.cuda(GPU)
    leftaxlebox_relation_network_2.cuda(GPU)
    motor_feature_encoder.cuda(GPU)
    motor_relation_network.cuda(GPU)
    motor_relation_network_2.cuda(GPU)

    def load_model(network, file_prefix):
        file_path = f"./models/{file_prefix}_{CLASS_NUM}way_{SAMPLE_NUM_PER_CLASS}shot.pkl"
        if os.path.exists(file_path):
            network.load_state_dict(torch.load(file_path))
            print(f"load {file_prefix} success")

    # 定义模型及对应文件前缀
    models = [
        (gearbox_feature_encoder, "gearbox_feature_encoder"),
        (gearbox_relation_network, "gearbox_relation_network"),
        (gearbox_relation_network_2, "gearbox_relation_network_2"),
        (leftaxlebox_feature_encoder, "leftaxlebox_feature_encoder"),
        (leftaxlebox_relation_network, "leftaxlebox_relation_network"),
        (leftaxlebox_relation_network_2, "leftaxlebox_relation_network_2"),
        (motor_feature_encoder, "motor_feature_encoder"),
        (motor_relation_network, "motor_relation_network"),
        (motor_relation_network_2, "motor_relation_network_2"),
    ]

    # 加载模型
    for network, prefix in models:
        load_model(network, prefix)

    # Step 3: build graph
    accuracy_list = [[0] * 9 for _ in range(8)]
    recall_list = [[0] * 9 for _ in range(8)]
    std_list = [[0] * 9 for _ in range(8)]
    for num_fault_type in range(1, 8 + 1):
        #####################################################
        if num_fault_type in [6, 7]:
            continue
        ######################################################
        for num_wc in range(1, 9 + 1):
            total_acc = 0
            total_recall = 0
            acc_for_std_list = []
            for ten_epoches in range(1, 11):
                total_rewards = 0
                recall_rewards = 0
                recall_times = 0
                for i in range(TEST_EPISODE):
                    degrees = random.choice([0, 90, 180, 270])
                    # metatest_character_folders1 = [f'../CWT-1000/gearbox/train/health/WC{num_wc}',
                    #                                f'../CWT-1000/gearbox/test/G{num_fault_type}/anomaly/WC{num_wc}']
                    # metatrain_character_folders1 = [f'../CWT-1000/gearbox/train/health/WC{num_wc}',
                    #                                 '../CWT-1000/gearbox/train/anomaly']
                    metatest_character_folders1 = [f'../CWT3-1000/gearbox/train/health/WC{num_wc}',
                                                   f'../CWT3-1000/gearbox/test/G{num_fault_type}/anomaly/WC{num_wc}']
                    metatrain_character_folders1 = [f'../CWT3-1000/gearbox/train/health/WC{num_wc}',
                                                    '../CWT3-1000/gearbox/train/anomaly']
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
                    sample_features = gearbox_feature_encoder(Variable(sample_images).cuda(GPU))  # 5x64
                    test_features = gearbox_feature_encoder(Variable(test_images).cuda(GPU))  # 20x64
                    sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1,
                                                                              1)
                    test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
                    test_features_ext = torch.transpose(test_features_ext, 0, 1)
                    relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1,
                                                                                                 FEATURE_DIM * 4,
                                                                                                 28 * 28)

                    relations1 = gearbox_relation_network(relation_pairs)
                    relations1 = relations1.view(2, 8 * 512)
                    relations1 = gearbox_relation_network_2(relations1)
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
                    if test_labels[0] == 1:
                        recall_times = recall_times + 1
                        recall_reward = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM)]
                        recall_rewards += np.sum(recall_reward)
                accuracy = total_rewards / 1.0 / CLASS_NUM / TEST_EPISODE
                recall = recall_rewards / 1.0 / CLASS_NUM / recall_times

                total_acc = total_acc + accuracy
                total_recall = total_recall + recall
                acc_for_std_list.append(accuracy)
            print("FaultType:G", str(num_fault_type).ljust(2),
                  "WorkCondition:", str(num_wc).ljust(2),
                  "   Accuracy:", f"{total_acc / 10.0:.4f}".ljust(6),
                  "   Recall:", f"{total_recall / 10.0:.4f}".rjust(10))

            std_list[num_fault_type - 1][num_wc - 1] = np.std(acc_for_std_list)
            accuracy_list[num_fault_type - 1][num_wc - 1] = total_acc / 10.0
            recall_list[num_fault_type - 1][num_wc - 1] = total_recall / 10.0
    return std_list, accuracy_list, recall_list


if __name__ == '__main__':
    std_data, acc_data, recall_data = main()
    df_std = pd.DataFrame(std_data)
    df_acc = pd.DataFrame(acc_data)
    df_recall = pd.DataFrame(recall_data)
    # 构建目标文件路径
    file_path_std = os.path.join('test_result', 'gearbox', 'gearbox_std.csv')
    file_path_acc = os.path.join('test_result', 'gearbox', 'gearbox_acc.csv')
    file_path_recall = os.path.join('test_result', 'gearbox', 'gearbox_recall.csv')
    # 保存为 CSV 文件
    df_std.to_csv(file_path_std, index=False, header=False)
    df_acc.to_csv(file_path_acc, index=False, header=False)
    df_recall.to_csv(file_path_recall, index=False, header=False)
