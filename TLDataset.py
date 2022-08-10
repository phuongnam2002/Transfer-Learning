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


class TLDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.images = os.listdir(path)  # lấy toàn bộ tên của file ảnh

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.path, self.images[item])
        label = self.images[item].split('.')[0]
        label = 0 if label == 'cat' else 1
        img = np.array(Image.open(img_path))
        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    pass
