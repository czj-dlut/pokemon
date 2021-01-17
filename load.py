import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
global train_dataset_loader,test_dataset
data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]),
}

train_dataset =torchvision.datasets.ImageFolder(root="C:/Users/Auror/Desktop/pokemon/data/train",transform=data_transform['train'])
train_dataset_loader =DataLoader(train_dataset,batch_size=8, shuffle=True,num_workers=4)    

val_dataset = torchvision.datasets.ImageFolder(root="C:/Users/Auror/Desktop/pokemon/data/val",transform=data_transform['val'])
val_dataset_loader = DataLoader(val_dataset,batch_size=8, shuffle=True,num_workers=4)   

test_dataset = torchvision.datasets.ImageFolder(root="C:/Users/Auror/Desktop/pokemon/data/test",transform=data_transform['val'])
test_dataset_loader = DataLoader(test_dataset,batch_size=25, shuffle=True,num_workers=4) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")