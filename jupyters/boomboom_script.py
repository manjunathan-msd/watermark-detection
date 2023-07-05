#Imports and Stuff
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)
from PIL import Image
import pandas as pd
import random
from tqdm import tqdm
import timm
import sys
sys.path.append('../')

#Sizes of Input Images
input_size = 256

#Model Building - ConvNext Tiny
from wmdetection.models.convnext import convnext_tiny, convnext_small

model_ft = convnext_tiny(pretrained=True, in_22k=True, num_classes=21841)

model_ft.head = nn.Sequential( 
    nn.Linear(in_features=768, out_features=512),
    nn.GELU(),
    nn.Linear(in_features=512, out_features=256),
    nn.GELU(),
    nn.Linear(in_features=256, out_features=1),
)

model_ft = model_ft.cuda()

#Preproc
class RandomRotation:
    def __init__(self, angles, p):
        self.p = p
        self.angles = angles

    def __call__(self, x):
        if random.random() < self.p:
            angle = random.choice(self.angles)
            return transforms.functional.rotate(x, angle)
        else:
            return x

preprocess = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        #transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        RandomRotation([90, -90], 0.2),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# df_train = pd.read_csv('../dataset/train_data_v1-1.csv')
# df_val = pd.read_csv('../dataset/val_data_v1-1.csv')


# df_train['path'] = df_train['path'].apply(lambda x: os.path.join('dataset', x))
# df_val['path'] = df_val['path'].apply(lambda x: os.path.join('dataset', x))

train_csv_path = sys.argv[1]
val_csv_path = sys.argv[2]

df_train = pd.read_csv(train_csv_path)
df_val = pd.read_csv(val_csv_path)

# Modify the image paths in the CSV files based on the input directory
input_directory = 'dataset'  # Modify this as per your directory structure
df_train['path'] = df_train['path'].apply(lambda x: os.path.join(input_directory, x))
df_val['path'] = df_val['path'].apply(lambda x: os.path.join(input_directory, x))



from io import BytesIO
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class WatermarkDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop = True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img = Image.open(self.df.loc[idx].path).convert('RGB')
        tensor = self.transform(img)
        return tensor, self.df.loc[idx].label

train_ds = WatermarkDataset(df_train, preprocess['train'])
val_ds = WatermarkDataset(df_val, preprocess['val'])

datasets = {
    'train': train_ds,
    'val': val_ds,
}

#Trainings and Forward Pass
from tqdm import tqdm
device = torch.device('cuda:0')

def train_model(model, dataloaders, criterion, optimizer, num_epochs=80):
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)

        #Plotting every 10 epochs
        if epoch % 10 == 0:
            # Plot and save the figure as PNG
            plt.plot([i.cpu().item() for i in train_acc_history])
            plt.plot([i.cpu().item() for i in val_acc_history])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['train', 'valid'], loc='upper left')
            plt.savefig(f'{epoch}.png')
            plt.close()
    
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model, train_acc_history, val_acc_history

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(params=model_ft.parameters(), lr=0.2e-5)

BATCH_SIZE = 64

dataloaders_dict = {
    x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=12) 
    for x in ['train', 'val']
}

import warnings
warnings.filterwarnings("ignore")

model_ft, train_acc_history, val_acc_history = train_model(
    model_ft, dataloaders_dict, criterion, optimizer, num_epochs=3
)

