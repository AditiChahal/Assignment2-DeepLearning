import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import models
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data transformations and augmentations
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 dataset
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_data, val_data = random_split(train_set, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# VGG Models built from scratch
class VGG(nn.Module):
    def __init__(self, num_classes=10, version='VGG16'):
        super(VGG, self).__init__()
        if version == 'VGG16':
            layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        elif version == 'VGG19':
            layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

        self.features = self._make_layers(layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

vgg16 = VGG(num_classes=10, version='VGG16')
vgg19 = VGG(num_classes=10, version='VGG19')

# ResNet Models used in-built models
class ResNet(nn.Module):
    def __init__(self, model_name, num_classes=10):
        super(ResNet, self).__init__()
        if model_name == 'ResNet18':
            self.model = models.resnet18(pretrained=True)
        elif model_name == 'ResNet34':
            self.model = models.resnet34(pretrained=True)
        elif model_name == 'ResNet50':
            self.model = models.resnet50(pretrained=True)

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

resnet18 = ResNet('ResNet18')
resnet34 = ResNet('ResNet34')
resnet50 = ResNet('ResNet50')

# Define training function with learning rate scheduler and weight decay
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, optimizer_type='Adam'):
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader.dataset):.4f}, '
              f'Val Loss: {val_loss/len(val_loader.dataset):.4f}, Accuracy: {100 * correct/total:.2f}%')

# Evaluate the models
def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = test_loss / total
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return avg_loss, accuracy, conf_matrix

# Plot Confusion Matrix
def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# List of models to train and evaluate
models_to_run = [
    ('VGG16', vgg16, 'SGD', 0.01),
    ('VGG19', vgg19, 'SGD', 0.01),
    ('ResNet18', resnet18, 'Adam', 0.001),
    ('ResNet34', resnet34, 'Adam', 0.001),
    ('ResNet50', resnet50, 'Adam', 0.001)
]

set_seed(42)

for model_name, model, optimizer_type, learning_rate in models_to_run:
    print(f"Training {model_name}")
    train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=learning_rate, optimizer_type=optimizer_type)
    print(f"Evaluating {model_name}")
    test_loss, test_accuracy, conf_matrix = evaluate_model(model, test_loader)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print('Confusion Matrix:')
    print(conf_matrix)
    plot_confusion_matrix(conf_matrix, class_names)
