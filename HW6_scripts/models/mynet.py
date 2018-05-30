import torch.nn as nn
import math

class MyNet(nn.Module):

    def __init__(self, inplanes=3, num_classes=10, kernel_size=3, stride=1,padding=1):
        super().__init()
        out1=8
        out2=8
        out3=16
        out4=16
        out5=32
        out6=32

        self.conv1 = nn.Conv2d(inplanes,out1,kernel_size,stride,padding=1) #32x32 input
        self.conv2 = nn.Conv2d(out1,out2,kernel_size,stride,padding=1) #32x32 input
        self.conv3 = nn.Conv2d(out2,out3,kernel_size,stride,padding=1) #16x16 input
        self.conv4 = nn.Conv2d(out3,out4,kernel_size,stride,padding=1) #16x16 input
        self.conv5 = nn.Conv2d(out4,out5,kernel_size,stride,padding=1) #8x8 input
        self.conv5 = nn.Conv2d(out5,out6,kernel_size,stride,padding=1) #8x8 input
        self.fc1 = nn.Linear(4*4*512,100) #4x4 input
        self.fc2 = nn.Linear(100,num_classes) #4x4 input
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.maxpool(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def mynet(inplanes, num_classes):
    model = MyNet(num_classes)
    return model
