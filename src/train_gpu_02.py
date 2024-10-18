

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.CIFAR10(root="../data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_size = len(train_data)
test_size = len(test_data)

# print("Train Data Size: {}".format(train_size))
# print("Test Data Size: {}".format(test_size))

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)


# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x



tudui = Tudui()
#可以不重新赋值
tudui = tudui.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)


learning_rate = 1e-3
#优化器
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)

total_train_step = 0

total_test_step = 0

epoch= 30

writer = SummaryWriter("../logs_gpu_train")
start_time = time.time()

for i in range(epoch):
    print("No. {} training step".format(i+1))

    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            end_time -= start_time
            print("--Training time for step {}: {}".format(total_train_step,end_time))
            print("Training step: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(),total_train_step)


    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("Overall Test Loss: {}".format(total_test_loss))
    print("Overall Test Accuracy: {}".format(total_accuracy/test_size))
    writer.add_scalar("test_loss", total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_size,total_test_step)
    total_test_step += 1

    torch.save(tudui,"tudui_{}.pth".format(i))
    #torch.save(tudui.state_dict(), "tudui_{}.pth".format(i))

    print("Model saved successfully.")

writer.close()