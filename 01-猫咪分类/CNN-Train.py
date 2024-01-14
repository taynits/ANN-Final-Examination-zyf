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
import  logging
logging.basicConfig(filename='Logs/CNN.log', level=logging.INFO)

# 如果有GPU就在GPU上运行
print(f'torch.cuda.is_available:{torch.cuda.is_available()}')
logging.info(f'torch.cuda.is_available:{torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
train_data_size = len(train_dataset)
test_data_size = len(val_dataset)
print("Train Dataset Size：{}".format(train_data_size))
logging.info("Train Dataset Size：{}".format(train_data_size))
print("Eval Dataset Size：{}".format(test_data_size))
logging.info("Eval Dataset Size：{}".format(test_data_size))

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

# 打印图片大小
print(f"Train Dataset Picture Size: {img1.size()}")
logging.info(f"Train Dataset Picture Size: {img1.size()}")
print(f"Eval Dataset Picture Size: {img2.size()}")
logging.info(f"Eval Dataset Picture Size: {img2.size()}")

# 创建 ResNet-50 模型
model = ResNet(Bottleneck, [3, 4, 6, 3])

# 加载预训练的ResNet模型
model = torchvision.models.resnet50(pretrained=True)
inchannel = model.fc.in_features
model.fc = nn.Linear(inchannel, 12)
model = model.to(device)
print(model)
logging.info(model)

# 设置损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

best_acc = 0.0

# -----------------训练模型----------------------------
epochs = 10
writer = SummaryWriter("Logs/logs_CNN")

for i in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
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
    print('Epoch [{}] - Train - loss: {:.4f}  acc：{:.2f}%'.format(i+1, avg_train_loss, train_accuracy*100))
    logging.info('Epoch [{}] - Train - loss: {:.4f}  acc：{:.2f}%'.format(i+1, avg_train_loss, train_accuracy*100))

    # 测试验证
    model.eval()
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
    writer.add_scalar('Validation Loss', avg_val_loss, i)
    writer.add_scalar('Validation Accuracy', val_accuracy, i)
    print('Epoch [{}] - Eval - loss: {:.4f}  acc：{:.2f}%'.format(i + 1, avg_val_loss, val_accuracy*100))
    logging.info('Epoch [{}] - Eval - loss: {:.4f}  acc：{:.2f}%'.format(i + 1, avg_val_loss, val_accuracy*100))
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), "results/model_CNN_best.pth")
        print("Best CNN model saved.")
        logging.info("Best CNN model saved.")

    print("------------------------------------------------")

print("best model - accuracy: {:.2f}%".format(best_acc*100))
logging.info("best model - accuracy: {:.2f}%".format(best_acc*100))
writer.close()

