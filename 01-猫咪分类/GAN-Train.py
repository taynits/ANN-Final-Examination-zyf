from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tensorboardX import SummaryWriter, writer
import matplotlib.pyplot as plt
import numpy as np
import logging
from models.GAN import GANClassifier, Discriminator, weights_init
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(filename='Logs/GAN_Classifier.log', level=logging.INFO)

# 如果有GPU就在GPU上运行
print(f'torch.cuda.is_available:{torch.cuda.is_available()}')
logging.info(f'torch.cuda.is_available:{torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 随机种子
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# ---------------准备数据集---------------------
class CatDataset(Dataset):
    def __init__(self, txt_file, img_dir, transform=None):
        self.data = []
        self.img_dir = img_dir
        self.transform = transform

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path, label = line.strip().split()
                self.data.append((img_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# 设置图像变换，图像增强
image_size = 64
transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

# 创建数据集
dataset = CatDataset(txt_file='./data/train_list.txt',
                     img_dir='./data', transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 显示训练集和验证集第一张图片
def show_image(image, label):
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title(f"Label: {label}")
    plt.show()


img1, label1 = next(iter(train_loader))
show_image(img1[0], label1[0])

img2, label2 = next(iter(val_loader))
show_image(img2[0], label2[0])

# ---------------参数设置---------------------
batch_size = 128
nc = 3
nz = 100  # 隐向量z的长度
ngf = 64  # 生成器的特征图大小
ndf = 64  # 判别器的特征图大小


# ----------------构造分类器----------------------
D = Discriminator().to(device)
D.apply(weights_init)
print("Staring Fineturn GAN...")
logging.info("Staring Fineturn GAN...")
D.load_state_dict(torch.load("results/discriminator.pth"))
D.eval()

feature_extractor = nn.Sequential(
    D.layer1,
    D.layer2,
    D.layer3,
    D.layer4
)
gan_classifier = GANClassifier(feature_extractor)
gan_classifier = gan_classifier.to(device)
print(gan_classifier)
logging.info(gan_classifier)

# ----------------训练与验证GAN分类器-----------------------
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
optimizer = torch.optim.SGD(gan_classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

epochs = 15
best_acc = 0.0
writer = SummaryWriter("Logs/logs_GANClassifier")
for i in range(epochs):
    gan_classifier.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 使用 GAN 分类器提取图像特征并进行分类
        outputs = gan_classifier(images)

        # 计算损失并更新分类器参数
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # 计算训练损失和准确率
    train_accuracy = correct_train / total_train
    avg_train_loss = running_loss / len(train_loader)
    writer.add_scalar('Training Accuracy', train_accuracy, i)
    writer.add_scalar('Training Loss', avg_train_loss, i)
    print('Epoch [{}] - Train - loss: {:.4f}  acc：{:.2f}%'.format(i + 1, avg_train_loss, train_accuracy * 100))
    logging.info('Epoch [{}] - Train - loss: {:.4f}  acc：{:.2f}%'.format(i + 1, avg_train_loss, train_accuracy * 100))

    # 测试验证
    gan_classifier.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = gan_classifier(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # 计算验证损失和准确率
    val_accuracy = correct_val / total_val
    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar('Validation Loss', avg_val_loss, i)
    writer.add_scalar('Validation Accuracy', val_accuracy, i)
    print('Epoch [{}] - Eval - loss: {:.4f}  acc：{:.2f}%'.format(i + 1, avg_val_loss, val_accuracy * 100))
    logging.info('Epoch [{}] - Eval - loss: {:.4f}  acc：{:.2f}%'.format(i + 1, avg_val_loss, val_accuracy * 100))
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(gan_classifier.state_dict(), "results/model_GAN_best.pth")
        print("Best GAN model saved.")
        logging.info("Best GAN model saved.")

    print("------------------------------------------------")

print("best model - accuracy: {:.2f}%".format(best_acc * 100))
logging.info("best model - accuracy: {:.2f}%".format(best_acc * 100))
writer.close()