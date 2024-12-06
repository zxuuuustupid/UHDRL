import torch
import random
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x


# 数据集构建
class OmniglotTask(object):
    # 这个类负责生成训练和测试任务
    def __init__(self, folder, num_classes, train_num, test_num):
        # 只接受一个文件夹
        self.folder = folder
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        # 获取文件夹中的所有图像路径
        all_images = [os.path.join(folder, x) for x in os.listdir(folder) if os.path.isfile(os.path.join(folder, x))]

        # 我们将所有图像标签设为同一类（标签为0）
        self.train_roots = random.sample(all_images, train_num)
        self.test_roots = random.sample(all_images, test_num)

        # 所有标签为统一的类
        self.train_labels = [0] * len(self.train_roots)
        self.test_labels = [0] * len(self.test_roots)

    def get_class(self, sample):
        return self.folder  # 所有样本都来自同一文件夹，类为0


class FewShotDataset(Dataset):
    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform  # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Omniglot(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)

        image = image.convert('RGB')
        image = image.resize((84, 84), resample=Image.LANCZOS)

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


class ClassBalancedSampler(Sampler):
    '''Samples 'num_inst' examples each from 'num_cl' pools of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # Return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train', shuffle=True, rotation=0):
    # NOTE: batch size here is # instances PER CLASS
    normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])

    dataset = Omniglot(task, split=split, transform=transforms.Compose([Rotate(rotation), transforms.ToTensor()]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)

    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    return loader
