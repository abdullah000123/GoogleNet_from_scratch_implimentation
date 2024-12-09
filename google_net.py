import pandas as pd
import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as pt
import os 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels_path = 'data/cifar-10/trainLabels.csv'
train = 'data/cifar-10/train'
imgsz = 224  # Image size (height and width)
image_no = 50000
0000  # Maximum number of images to process
train_path = 'data/cifar-10/train/'  # Path to training images
count = 0
images = []

labels_data = pd.read_csv(labels_path)
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('\n\nairplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
encoded_labels = pd.get_dummies(labels_data['label'], columns=labels).astype(np.int8)
encoded_labels = np.array(encoded_labels)
print(labels_data.head())
labels = np.array(encoded_labels)


for i in range(1,image_no+1):
    if count >= image_no:  # Stop if the desired number of images is reached
        break
    img_path = os.path.join(train_path,  f"{i}.png")
    # Read the image in color
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    rs_img = cv.resize(img, (imgsz, imgsz))
    # Store the resized image in the array
    images.append(rs_img)
    count += 1
# Trim the `images` array to the actual number of processed images

print(f"Images loaded and resized successfully! Total images processed: {count}")
images = np.array(images)

batch_sz = 60
num_batches = len(images) // batch_sz
batch_img = []
batch_lab = []

for h in range(num_batches):
    start = h * batch_sz
    end = (h + 1) * batch_sz
    batch = images[start:end]
    batch_l = labels[start:end]
    batch_img.append(batch)
    batch_lab.append(batch_l)
batch_img = np.array(batch_img)
#batch_img = torch.tensor(batch_img).to('cuda')
batch_lab = np.array(batch_lab)
#batch_lab = torch.tensor(batch_lab).to('cuda')
num_batches

class Inception(nn.Module):
    def __init__(self, in_ch, c1, c3_r, c3, c5_r, c5, pool_c):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, c1, 1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, c3_r, 1),
            nn.ReLU(),
            nn.Conv2d(c3_r, c3, 3, padding=1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, c5_r, 1),
            nn.ReLU(),
            nn.Conv2d(c5_r, c5, 5, padding=2),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_c, 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        out = torch.cat([branch1, branch2, branch3, branch4], 1)
        return out

class auxl(nn.Module):
    def __init__(self, no_of_ch, no_c):
        super().__init__()
        self.avg =nn.AdaptiveAvgPool2d((4,4))
        self.conv1 = nn.Conv2d(no_of_ch, 128, 1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, no_c)

    def forward(self, x):
        x = self.avg(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        return x

class googlenet(nn.Module):
    def __init__(self, nc):
        super().__init__()
    
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, 3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception1a = Inception(192, 64, 96, 128, 16, 32, 32) 
        self.inception1b = Inception(256, 128, 128, 192, 32, 96, 64) 
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception2a = Inception(480, 192, 96, 208, 16, 48, 64) 
        self.auxl1 = auxl(512, nc)
        self.inception2b = Inception(512, 160, 112, 224, 24, 64, 64) 
        self.inception2c = Inception(512, 128, 128, 256, 24, 64, 64) 
        self.inception2d = Inception(512, 112, 144, 288, 32, 64, 64) 
        self.auxl2 = auxl(528, nc)
        self.inception2e = Inception(528, 256, 160, 320, 32, 128, 128) 
        self.pool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception3a = Inception(832, 256, 160, 320, 32, 128, 128) 
        self.inception3b = Inception(832, 384, 192, 384, 48, 128, 128) 
        self.avg1 = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc_final = nn.Linear(1024, nc)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.inception1a(x)
        x = self.inception1b(x)
        x = self.pool3(x)
        x = self.inception2a(x)
        aux1 = self.auxl1(x)
        x = self.inception2b(x)
        x = self.inception2c(x)
        x = self.inception2d(x)
        aux2 = self.auxl2(x)
        x = self.inception2e(x)
        x = self.pool4(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.avg1(x)
        x = x.reshape(x.size(0), -1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc_final(x)
        return x, aux1, aux2


model = googlenet(10).to('cuda')
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() ,lr = 0.001)
def train(epochs):
    for e in range(epochs):
        print('starting epoch')
        epoch_loss = 0
        for b in range(num_batches):
            optimizer.zero_grad()
            batch_im = torch.tensor(batch_img[b] ,dtype = torch.float32).permute(0 ,3 ,1 ,2).to('cuda')
            batch_l = torch.tensor(batch_lab[b] , dtype = torch.long).argmax(dim = 1).to('cuda')
            output ,aux1 ,aux2 = model(batch_im)
            loss1 = loss_fun(output ,batch_l)
            aux1_loss = loss_fun(aux1 ,batch_l)
            aux2_loss = loss_fun(aux2 ,batch_l)
            #print('Calculating loss for batch #######',b)
            loss = (1 - 0.3) * loss1 + (aux1_loss + aux2_loss) * 0.3
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #print(f"batch no {b} Loss {loss}")
        print(f"Epoch no {e + 1} Average loss {epoch_loss/num_batches:.4f}")
train(20)
