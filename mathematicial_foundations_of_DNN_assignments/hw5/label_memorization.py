import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
import random
import time
import matplotlib.pyplot as plt

# Make sure to use only 10% of the available MNIST data.
# Otherwise, experiment will take quite long (around 90 minutes).

# (Modified version of AlexNet)
class AlexNet(nn.Module):
    def __init__(self, num_class=10):
        super(AlexNet, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc_layer1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6400, 800),
            nn.ReLU(inplace=True),
            nn.Linear(800, 10)
        )

    def forward(self, x):
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = torch.flatten(output, 1)
        output = self.fc_layer1(output)
        return output

class RandomizedReducedMNIST(Dataset):
    def __init__(self):
        self.MNIST = train_dataset = datasets.MNIST(root='./mnist_data/',
                                                    train=True, 
                                                    transform=transforms.ToTensor(),
                                                    download=True)
        self.MNIST_reduced = Subset(train_dataset, torch.randint(60000, (6000,)))
        self.label = torch.randint(10, (6000,))

    def __len__(self):
        return len(self.MNIST_reduced)
    
    def __getitem__(self, idx):
        image = self.MNIST_reduced[idx][0]
        label = self.label[idx]
        return image, label




train_dataset = RandomizedReducedMNIST()
test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False, 
                              transform=transforms.ToTensor())

learning_rate = 0.1
batch_size = 64
epochs = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


train_accuracy = []
train_loss = []

tick = time.time()
for epoch in range(20):
    print(f"\nEpoch {epoch + 1} / {epochs}")
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = loss_function(output, labels)
        
        train_loss.append(loss)

        pred = output.argmax(dim=1)
        accuracy = pred.eq(labels.view_as(pred)).sum().item()
        train_accuracy.append(accuracy)

        loss.backward()

        optimizer.step()

tock = time.time()
print(f"Total training time: {tock - tick}")


plt.plot(train_accuracy)
plt.plot(train_loss)
plt.show()