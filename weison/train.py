import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from torchvision import transforms
import model.Models as Models #加载数据集

from torch.utils.data import Dataset, DataLoader
import os

# 自定义数据集类
class SensorDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for label, gesture in enumerate(['punch', 'swipe']):
            gesture_path = os.path.join(self.data_path, gesture)
            for file in os.listdir(gesture_path):
                if file.endswith('.npy'):
                    file_path = os.path.join(gesture_path, file)
                    data = np.load(file_path)
                    self.data.append(data)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集
train_dataset = SensorDataset(data_path='path/to/train_data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

val_dataset = SensorDataset(data_path='path/to/val_data', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 定义ResNet 3D模型
class ResNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet3D, self).__init__()
        self.resnet = Models.Res3D(10, number_of_segments=8)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet3D(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    # 验证模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total}%')

print('Finished Training')