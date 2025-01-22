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
from src import fullconnect
import CNNEncoder1
import vit
from src.fullconnect import FullyConnectedLayer, FullyConnectedLayer2
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
    loos_result = []
    loos_result_1 = []
    accuray_result_1_1 = []
    # Step 1: init data folders
    print("init data folders")
    print("init neural networks")

    # 定义化网络
    feature_encoder = CNNEncoder1.rsnet()  # 特征提取
    # relation_network = vit.ViT(image_size=28, patch_size=7, num_classes=2, dim=1024, depth=4, heads=8, mlp_dim=2048,
    #                            dropout=0.1, emb_dropout=0.1)  # 定义关系网络
    fc = FullyConnectedLayer()
    fc2 =FullyConnectedLayer2()
    # 关系网络和特征提取模块的加载话
    #    feature_encoder.apply(weights_init)
    #   relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    fc.cuda(GPU)
    fc2.cuda(GPU)
    # 优化器的定义
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    fc_optim = torch.optim.Adam(fc.parameters(), lr=LEARNING_RATE)
    fc_scheduler = StepLR(fc_optim, step_size=100000, gamma=0.5)
    fc2_optim = torch.optim.Adam(fc2.parameters(), lr=LEARNING_RATE)
    fc2_scheduler = StepLR(fc2_optim, step_size=100000, gamma=0.5)

    # Step 3: build graph
    print("Training...")
    last_accuracy = 0
    for episode in range(EPISODE):
        # print('episode', episode)
        feature_encoder_scheduler.step(episode)
        fc_scheduler.step(episode)
        fc2_scheduler.step(episode)
        degrees = random.choice([0, 90, 180, 270])
        #########################################################
        triplet_num = random.randint(1, 9)
        # health_character_folders_1 = [f'../CWT-1000/gearbox/train/health/WC{triplet_num}',
        #                               '../CWT-1000/gearbox/train/health']
        health_character_folders_1 = [f'../CWT3-1000/gearbox/train/health/WC{triplet_num}',
                                      '../CWT3-1000/gearbox/train/health']
        # arch_character_folders_1 = [f'../CWT-1000/gearbox/train/health/WC{triplet_num}',
        #                             '../CWT-1000/gearbox/train/health']
        arch_character_folders_1 = [f'../CWT3-1000/gearbox/train/health/WC{triplet_num}',
                                    '../CWT3-1000/gearbox/train/health']
        random_folder = random.choice([f.path for f in os.scandir('../CWT3-1000/gearbox/train/anomaly') if f.is_dir()])
        anomaly_character_folders_1 = [random_folder,
                                       '../CWT3-1000/gearbox/train/anomaly']
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

        triloss = TripletLoss(margin=0.1)
        loss_punish = triloss(batch_features_arch, batch_features_health, batch_features_anomaly)
        #########################################################

        ##第一个监测点  轴箱gearbox
        num_wc = random.randint(1, 9)
        metatrain_character_folders_1 = [f'../CWT3-1000/gearbox/train/health/WC{num_wc}',
                                         '../CWT3-1000/gearbox/train/anomaly']
        # metatrain_character_folders_1 = [f'../CWT-1000/gearbox/train/health/WC{num_wc}',
        #                                  '../CWT-1000/gearbox/train/anomaly']
        task_1 = tg.OmniglotTask(metatrain_character_folders_1, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        sample_dataloader_1 = tg.get_data_loader(task_1, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
                                                 shuffle=False, rotation=degrees)
        sample_dataloader_1 = iter(sample_dataloader_1)
        samples_1, sample_labels_1 = next(sample_dataloader_1)
        sample_features_1 = feature_encoder(Variable(samples_1).cuda(GPU))
        sample_features_1=sample_features_1.view(-1,2*128*28*28)
        sample_features_1 = fc2(sample_features_1)
        sample_features_1=sample_features_1.view(8,2)
        mse = nn.MSELoss().cuda(GPU)
        batch_labels_1 = sample_labels_1.long()
        one_hot_labels_1 = torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1,
                                                                                            batch_labels_1.view(-1, 1),
                                                                                            1).cuda(GPU)

        loss_1 = mse(sample_features_1, one_hot_labels_1)
        ########################################################################################
        feature_encoder.zero_grad()
        loss = loss_1 + loss_punish
        loss.backward()
        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(fc.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(fc2.parameters(), 0.5)
        feature_encoder_optim.step()
        fc_optim.step()
        fc2_optim.step()
        loos_result_1.append(loss_1)
        loos_result.append(loss)

        if (episode + 1) % 50 == 0:
            print("episode:", episode + 1, "loss", loss.item(), "TripletLoss", loss_punish)
            loos_result.append(loss)

        if (episode + 1) % 100 == 0:
            # test

            print("Testing")
            acc_list=[[0]*9 for _ in range(8) ]
            accuracy72 = 0
            for num_train_wc in range(1, 10):
                for num_train_fault_type in range(1, 9):
                    total_rewards_1_1 = 0
                    for i in range(TEST_EPISODE):

                        degrees = random.choice([0, 90, 180, 270])
                        metatest_character_folders1 = [f'../CWT3-1000/gearbox/test/health/WC{num_train_wc}',
                                                       f'../CWT3-1000/gearbox/test/G{num_train_fault_type}/anomaly/WC{num_train_wc}']
                        task = tg.OmniglotTask(metatest_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
                                               SAMPLE_NUM_PER_CLASS, )
                        test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
                                                             shuffle=True, rotation=degrees)
                        test_dataloader = iter(test_dataloader)
                        test_images, test_labels = next(test_dataloader)
                        test_features = feature_encoder(Variable(test_images).cuda(GPU))
                        test_features = test_features.view(-1,2*128*28*28)
                        relations=fc2(test_features)
                        relations=relations.view(8,2)
                        relations=relations.view(-1)
                        test_labels=test_labels.long()
                        test_labels = torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1,
                                                                                                            test_labels.view(
                                                                                                                -1, 1),
                                                                                                            1).cuda(GPU)
                        test_labels=test_labels.view(-1)
                        bb = Variable(torch.zeros(CLASS_NUM*8)).cuda(GPU)

                        # print(relations)
                        for j in range(len(relations)):
                            if relations[j]> 0.9:
                                bb[j] = 0
                            else:
                                bb[j] = 1
                        predict_labels = bb.cpu()
                        rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM*8)]
                        total_rewards_1_1 += np.sum(rewards)
                    test_accuracy = total_rewards_1_1 / 1.0 / (CLASS_NUM *8)/ TEST_EPISODE
                    accuray_result_1_1.append(test_accuracy)
                    acc_list[num_train_fault_type-1][num_train_wc-1]=test_accuracy
                    accuracy72 = accuracy72 + test_accuracy
            print(" test accuracy:", accuracy72 / 72.0)
            if episode == 99:
                acc_list=np.array(acc_list)
                np.savetxt(train_result + 'ablation/'+'gearbox_norelation.csv', acc_list, fmt='%.8f', delimiter=',')


            # if accuracy72 >= last_accuracy:
                # save networks
                # torch.save(feature_encoder.state_dict(),
                #            str("./models/gearbox_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                #                SAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                # print("save networks for episode:", episode)

                # last_accuracy = accuracy72

    return loos_result_1


if __name__ == '__main__':
    loos_result_1 = main()
    # loos_result_1_cpu = [x_1.cpu().detach().numpy() for x_1 in loos_result_1]
    # np.savetxt(train_result + 'gearbox_train_loss.csv', loos_result_1_cpu, fmt='%.8f', delimiter=',')
