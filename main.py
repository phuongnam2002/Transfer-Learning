import copy
import os
from io import BytesIO
import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm
from TLDataset import *


DEVICE = "cuda"


def validate(model, data):

    total = 0
    correct = 0

    for (images, labels) in data:
        images = images.to(DEVICE)
        x = model(images) # xác suất rơi vào từng class [x, y]
        _, pred = torch.max(x, 1)
        total += x.size(0)
        pred = pred.to(DEVICE)
        labels = labels.to(DEVICE)
        correct += torch.sum(pred == labels)
        # print(labels): num_batch labels mỗi vòng for
        # print(x.size(0)): num_batch
    return correct*100/total


def train(model, num_epoch, lr, device):
    accuracies = []
    cnn = model.to(device)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)

    max_accuracy = 0

    for epoch in range(num_epoch):
        for i, (images, labels) in tqdm(enumerate(train_dl)):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
        accuracy = float(validate(cnn, val_dl))
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("saving best model with accuracy: ", accuracy)
        print("Epoch: ", epoch+1, "Accuracy: ", accuracy, "%")

    return best_model


if __name__ == '__main__':
    train_path = "./train"
    val_path = "./val"
    num_batch = 32
    epochs = 10
    lr = 1e-3
    transform = transforms.Compose([
        ToTensor(),
        Resize((500, 500))
    ])
    train_data = TLDataset(train_path, transform)
    val_data = TLDataset(val_path, transform)
    # t = next(iter(train_data))
    # print(t[1].shape)
    train_dl = DataLoader(train_data, batch_size=num_batch)
    val_dl = DataLoader(val_data, batch_size=num_batch)
    # t = next(iter(train_dl))
    # print(t[1].shape)
    # batch_size, channels, height, weight
    model = torchvision.models.resnet18(pretrained=True)
    # freeze weight
    for param in model.parameters():
        param.requires_grad = False
    # finetune FC cuối
    model.fc = nn.Sequential(*[
        nn.Linear(in_features=512, out_features=2),
        nn.Softmax(dim=1)
    ])
    model.to(DEVICE)
    # model ban đầu có 1000 class thì mk sẽ finetune lại còn 2 class
    model = train(model, epochs, lr, DEVICE)
    torch.save(model.state_dict(), "ResNet18_CatDog.pth")



