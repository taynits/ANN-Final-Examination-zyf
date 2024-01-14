import os
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from models.CNN import ResNet, Bottleneck


# --------------准备数据集-----------------------
class CatDataset(Dataset):
    def __init__(self, txt_file, img_dir, transform=None):
        self.data = []
        self.img_dir = img_dir
        self.transform = transform

        # 读取file的每一行，【图片路径 标签】
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


# 设置图像大小变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
transform=transforms.Compose([transforms.Resize((224, 224)),
                               transforms.CenterCrop((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

# 创建数据集
dataset = CatDataset(txt_file='./data/train_list.txt',
                     img_dir='./data', transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 数据集长度
test_data_size = len(val_dataset)
print("Eval Dataset Size：{}".format(test_data_size))

# 创建 DataLoader
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 显示训练集和验证集第一张图片
def show_image(image, label):
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title(f"Label: {label}")
    plt.show()

# 加载模型与权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(Bottleneck, [3, 4, 6, 3])
print(model)
model = model.to(device)
model.load_state_dict(torch.load("results/model_CNN_best.pth"))
model.eval()

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


