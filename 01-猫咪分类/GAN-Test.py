from __future__ import print_function
import os
import random
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from models.GAN import GANClassifier, Discriminator, weights_init
from torch.utils.data import Dataset, DataLoader
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
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

D = Discriminator().to(device)
D.apply(weights_init)
D.load_state_dict(torch.load("results/discriminator.pth"))
D.eval()

feature_extractor = nn.Sequential(
    D.layer1,
    D.layer2,
    D.layer3,
    D.layer4
)

model = GANClassifier(feature_extractor)
model = model.to(device)
model.load_state_dict(torch.load("results/model_GAN_best.pth"))
model.eval()
print(model)


# ----------------验证GAN分类器-----------------------
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

correct_val = 0
total_val = 0
val_loss = 0.0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()

# 计算验证损失和准确率
val_accuracy = correct_val / total_val
avg_val_loss = val_loss / len(val_loader)
print('best model - Eval - loss: {:.4f}  acc：{:.2f}%'.format(avg_val_loss, val_accuracy*100))
