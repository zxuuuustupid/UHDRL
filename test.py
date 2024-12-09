
# -------------------------------------
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

# from src.efficient_kan import KAN
import CNNEncoder1
import vit
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
    # init character folders for dataset construction
    # metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders(train,test,label_list)

    # Step 2: init neural networks
    print("init neural networks")

    # 定义化网络
    feature_encoder = CNNEncoder1.rsnet()  # 特征提取
    # relation_network = vit.ViT(image_size=28, patch_size=7, num_classes=2, dim=1024, depth=4, heads=8, mlp_dim=2048,
    #                            dropout=0.1, emb_dropout=0.1)  # 定义关系网络
    # relation_network = KAN([28 * 28, 128,8])
    # kan = KAN([8 * 512, 512, 128])
    # 关系网络和特征提取模块的加载话
    #    feature_encoder.apply(weights_init)
    #   relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    # relation_network.cuda(GPU)
    # kan.cuda(GPU)

    # 优化器的定义
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)

    # relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    # relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)
    #
    # if os.path.exists(
    #         str("./models/feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
    #     feature_encoder.load_state_dict(torch.load(
    #         str("./models/feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
    #     print("load feature encoder success")
    # if os.path.exists(str("./models/relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
    #     relation_network.load_state_dict(torch.load(str("./models/relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
    #     print("load relation network success")

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0

    for episode in range(EPISODE):
        # print('episode', episode)
        feature_encoder_scheduler.step(episode)
        # relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        degrees = random.choice([0, 90, 180, 270])

        ##第一个监测点  轴箱motor
        # metatrain_character_folders_1 = ['../CWT-1000/motor/train/health',
        #                                  '../CWT-1000/motor/train/anomaly']
        metatrain_character_folders_1 = ['../CWT-1000/motor/train/health0',
                                         '../CWT-1000/motor/train/health']
        # # label = 0 health

        # metatrain_character_folders_1 = '../CWT-1000/motor/train/health'
        task_1 = tg.OmniglotTask(metatrain_character_folders_1, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        batch_dataloader_1 = tg.get_data_loader(task_1, num_per_class=BATCH_NUM_PER_CLASS, split="test", shuffle=True,
                                                rotation=degrees)

        #add 1
        metatrain_character_folders_2 = ['../CWT-1000/motor/train/arch0',
                                         '../CWT-1000/motor/train/arch']
        task_2 = tg.OmniglotTask(metatrain_character_folders_2, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        batch_dataloader_2 = tg.get_data_loader(task_2, num_per_class=BATCH_NUM_PER_CLASS, split="test", shuffle=True,
                                                rotation=degrees)

        metatrain_character_folders_3 = ['../CWT-1000/motor/test/anomaly0',
                                         '../CWT-1000/motor/test/anomaly']
        task_3 = tg.OmniglotTask(metatrain_character_folders_3, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        batch_dataloader_3 = tg.get_data_loader(task_3, num_per_class=BATCH_NUM_PER_CLASS, split="test", shuffle=True,
                                                rotation=degrees)

        batch_dataloader_1 = iter(batch_dataloader_1)
        batches_1, batch_labels_1 = next(batch_dataloader_1)

        batch_dataloader_2 = iter(batch_dataloader_2)
        batches_2, batch_labels_2 = next(batch_dataloader_2)

        batch_dataloader_3 = iter(batch_dataloader_3)
        batches_3, batch_labels_3 = next(batch_dataloader_3)

        ## 特征提取
        # 5x64*5*5
        # print(batches_1.shape)
        batch_features_1 = feature_encoder(Variable(batches_1).cuda(GPU))
        batch_features_2 = feature_encoder(Variable(batches_2).cuda(GPU))
        batch_features_3 = feature_encoder(Variable(batches_3).cuda(GPU))# 20x64*5*5
        #######################################################################################


        triloss=TripletLoss(margin=0.5).cuda(GPU)
        # mse = nn.MSELoss().cuda(GPU)
        # 计算LOSS
        batch_labels_1 = batch_labels_1.long()
        # print(batches_1.shape)
        # print(batch_labels_1.view(-1,1))
        one_hot_labels_1 = torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1,
                                                                                            batch_labels_1.view(-1, 1),
                                                                                            1).cuda(GPU)
        # one_hot_labels_1.zero_()
        # one_hot_labels_1[:, 0] = 1
        # print('start')
        # print(relations_1,relations_2,relations_3)

        # print(one_hot_labels_1)
        # print(relations_1.shape)
        # print(batch_features_1.shape,batch_features_2.shape,batch_features_3.shape)
        # ap_dist = torch.norm(batch_features_1 - batch_features_2, 2, dim=1).view(-1)
        # an_dist = torch.norm(batch_features_1 - batch_features_3, 2, dim=1).view(-1)
        # print(batch_features_1.shape)
        batch_features_1=batch_features_1.view(8,-1)
        batch_features_2=batch_features_2.view(8, -1)
        batch_features_3=batch_features_3.view(8, -1)
        # print(batch_features_1.shape)
        loss_1 = triloss(batch_features_2, batch_features_1,batch_features_3)

        feature_encoder.zero_grad()

        loss = loss_1

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        # torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        # relation_network_optim.step()

        loos_result_1.append(loss_1)

        loos_result.append(loss)

        if (episode + 1) % 50 == 0:
            print("episode:", episode + 1, "loss", loss.item())
            loos_result.append(loss)

        if (episode + 1) % 100 == 0:
            # test

            print("Testing,motor",end='')
            total_rewards_1_1 = 0
            total_rewards_h=0
            total_rewards_f=0
            for i in range(TEST_EPISODE):
                degrees = random.choice([0, 90, 180, 270])
                metatest_character_folders1 = ['../CWT-1000/motor/test/health0',
                                               '../CWT-1000/motor/test/anomaly']

                task = tg.OmniglotTask(metatest_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
                                       SAMPLE_NUM_PER_CLASS, )

                test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test",
                                                     shuffle=True, rotation=degrees)
                test_dataloader = iter(test_dataloader)
                test_images1, test_labels1 = next(test_dataloader)
                test_features1 = feature_encoder(Variable(test_images1).cuda(GPU))  # 20x64


                metatest_character_folders1 = ['../CWT-1000/motor/test/health0',
                                               '../CWT-1000/motor/test/health']

                task = tg.OmniglotTask(metatest_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
                                       SAMPLE_NUM_PER_CLASS, )
                test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test",
                                                     shuffle=True, rotation=degrees)
                test_dataloader = iter(test_dataloader)
                test_images2, test_labels2 = next(test_dataloader)
                test_features2 = feature_encoder(Variable(test_images2).cuda(GPU))  # 20x64


                metatest_character_folders1 = ['../CWT-1000/motor/test/anomaly0',
                                               '../CWT-1000/motor/test/anomaly']
                # '../CWT-1000/motor/test/anomaly/anomalyTYPE13']
                # '../CWT-1000/motor/test/anomaly/anomalyTYPE14']

                task = tg.OmniglotTask(metatest_character_folders1, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
                                       SAMPLE_NUM_PER_CLASS, )
                test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test",
                                                     shuffle=True, rotation=degrees)
                test_dataloader = iter(test_dataloader)
                test_images3, test_labels3 = next(test_dataloader)
                test_features3 = feature_encoder(Variable(test_images3).cuda(GPU))  # 20x64

                bb = Variable(torch.zeros(CLASS_NUM)).cuda(GPU)

                # print(test_features1.shape)
                # 展平输入特征
                test_features1_flat = test_features1.view(test_features1.size(0), -1)  # 变为 [2, 128*28*28]
                test_features2_flat = test_features2.view(test_features2.size(0), -1)  # 变为 [2, 128*28*28]
                test_features3_flat = test_features3.view(test_features3.size(0), -1)  # 变为 [2, 128*28*28]

                # 计算欧氏距离
                an_dist = torch.norm(test_features1_flat - test_features3_flat, p=2, dim=1)  # 结果形状为 [2]
                ap_dist = torch.norm(test_features1_flat - test_features2_flat, p=2, dim=1)  # 结果形状为 [2]

                # print(an_dist,ap_dist,test_labels,end='')
                # print(an_dist, '', ap_dist)
                # print(ap_dist.shape," ",an_dist.shape)

                # if an_dist[0] > ap_dist[0] and an_dist[1] > ap_dist[1]:
                #     print("1", end="")
                # else:
                #     print("0", end="")

                # print(relations.shape)
                # print(ap_dist.shape)

                # print(relations.shape)
                for j in range(len(test_features1)):
                    # print(an_dist[j],ap_dist[j])
                    if an_dist[j] > ap_dist[j]:

                        bb[j]=0
                    else:
                        bb[j]=1

                print(bb,test_labels1)

                # for j in range(len(relations)):
                #     if relations[j][0] > 0.9:
                #         bb[j] = 1
                #     else:
                #         bb[j] = 0

                # _,predict_labels = torch.max(relations.data,1)
                predict_labels = bb.cpu()
                # print(predict_labels.shape)
                # # print(test_labels[1])
                # print(predict_labels[0], end='')
                # print(test_labels[0])
                rewards = [1 if predict_labels[j] == test_labels1[j] else 0 for j in range(CLASS_NUM)]
                total_rewards_1_1 += np.sum(rewards)
                # rewards_h=[1 if predict_labels[j] == test_labels1[j] and test_labels[j]==0 else 0 for j in range(CLASS_NUM)]
                # total_rewards_h +=np.sum(rewards_h)
                # rewards_f = [1 if predict_labels[j] == test_labels1[j] and test_labels[j] == 1 else 0 for j in
                #              range(CLASS_NUM)]
                # total_rewards_f += np.sum(rewards_f)
            test_accuracy_1_1 = total_rewards_1_1 / 1.0 / CLASS_NUM / TEST_EPISODE
            health_acc = total_rewards_h / 1.0 / TEST_EPISODE
            fault_acc = total_rewards_f/ 1.0 / TEST_EPISODE
            accuray_result_1_1.append(test_accuracy_1_1)
            print(" test accuracy:", test_accuracy_1_1)


    return loos_result_1


if __name__ == '__main__':
    loos_result_1 = main()
    loos_result_1_cpu = [x_1.cpu().detach().numpy() for x_1 in loos_result_1]

    # np.savetxt(train_result + 'motor2000_train_loss_1.csv', loos_result_1_cpu, fmt='%.8f', delimiter=',')

