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
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from models.GAN import Generator, Discriminator, weights_init
import logging
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(filename='Logs/GAN.log', level=logging.INFO)

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

# ---------------创建模型---------------------
G = Generator(nz=nz, ngf=ngf).to(device)
G.apply(weights_init)
print(G)
logging.info(G)

D = Discriminator(nc=nc, ndf=ndf).to(device)
D.apply(weights_init)
print(D)
logging.info(D)

# ------------------训练和验证GAN---------------------------
epochs = 300
lr = 0.0002
beta1 = 0.5
img_list = []
G_losses = []
D_losses = []
i = 0
writer = SummaryWriter("Logs/logs_GAN")

# 损失函数和优化器
loss_fn = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1.
fake_label = 0.
print("Starting Training GAN...")
logging.info("Starting Training GAN...")
for epoch in range(epochs):
    for images, labels in train_loader:

        # (1) 训练 D : maximize log(D(x)) + log(1 - D(G(z)))
        D.zero_grad()
        real_data = images.to(device)
        b_size = real_data.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = D(real_data).view(-1)
        lossD_real = loss_fn(output, label)
        lossD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = G(noise)
        label.fill_(fake_label)
        output = D(fake.detach()).view(-1)
        lossD_fake = loss_fn(output, label)
        lossD_fake.backward()
        D_G_z1 = output.mean().item()
        lossD = lossD_real + lossD_fake
        optimizerD.step()

        # (2) 训练 G: maximize log(D(G(z)))
        G.zero_grad()
        label.fill_(real_label)
        output = D(fake).view(-1)
        lossG = loss_fn(output, label)
        lossG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # 输出训练结果
        if i % 50 == 0:
            print('Epoch [%d/%d] - Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch + 1, epochs, lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
            logging.info('Epoch [%d/%d] - Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                         % (epoch + 1, epochs, lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
            writer.add_scalar('Loss_D', lossD.item(), epoch + 1)
            writer.add_scalar('Loss_G', lossG.item(), epoch + 1)
            writer.add_scalar('D(x)', D_x, epoch + 1)
            writer.add_scalar('D(G(z))_before', D_G_z1, epoch + 1)
            writer.add_scalar('D(G(z))_after', D_G_z2, epoch + 1)

        G_losses.append(lossG.item())
        D_losses.append(lossD.item())

        # 保存对固定扰动的生成效果
        if (i % 500 == 0) or ((epoch == epochs - 1) and (i == len(train_loader) - 1)):
            with torch.no_grad():
                fake = G(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        i += 1

real_batch = next(iter(train_loader))

# 显示原始图片与生成的假图片
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()

# 保存生成器和判别器模型权重
torch.save(G.state_dict(), "results/generator.pth")
print("Generator model saved.")
logging.info("Generator model saved.")
torch.save(D.state_dict(), "results/discriminator.pth")
print("Discriminator model saved.")
logging.info("Discriminator model saved.")

writer.close()