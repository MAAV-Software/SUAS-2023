import os
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision
import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models

shapes = [
    "circle",
    "semicircle",
    "quartercircle",
    "star",
    "pentagon",
    "triangle",
    "rectangle",
    "cross",
]
colors_dict = {
    "white": (255, 255, 255), 
    "black": (0, 0, 0),       
    "red": (255, 0, 0),     
    "blue": (0, 0, 255),     
    "green": (0, 255, 0),     
    "purple": (127, 0, 255),   
    "brown": (102, 51, 0),  
    "orange": (255, 128, 0),   
}
symbols = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

class dataset_parser(Dataset):
    def __init__(self, root_dir, shapes, colors_dict, symbols, transform=None):
        self.root_dir = root_dir
        self.shapes = shapes
        self.colors_dict = colors_dict
        self.symbols = symbols
        self.transform = transform
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_name).convert("RGB")

        # image = image.filter(ImageFilter.GaussianBlur(radius = 15))

        # image.save("blurred.png")

        # Extract labels from the filename
        label_parts = os.path.splitext(self.file_list[idx])[0].split('_')
        shape_label = label_parts[1]
        color_label = label_parts[2]
        symbol_label = label_parts[3]

        shape_index = self.shapes.index(shape_label)
        color_index = list(self.colors_dict.keys()).index(color_label)
        symbol_index = self.symbols.index(symbol_label)

        label = torch.tensor([shape_index, color_index, symbol_index], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# Define your transformations 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

data_path = 'test_images'
dataset = dataset_parser(root_dir=data_path, shapes=shapes, colors_dict=colors_dict, symbols=symbols, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

torch.manual_seed(42)

# custom model using resnet
class resnet18_custom(nn.Module):
    def __init__(self, num_shapes, num_colors, num_symbols):
        super(resnet18_custom, self).__init__()

        self.features = models.resnet18(pretrained=True) 
        self.features.fc = nn.Identity()  

        self.fc_shape = nn.Linear(512, num_shapes)
        self.fc_color = nn.Linear(512, num_colors)
        self.fc_symbol = nn.Linear(512, num_symbols)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        shape_out = self.fc_shape(x)
        color_out = self.fc_color(x)
        symbol_out = self.fc_symbol(x)
        return shape_out, color_out, symbol_out

model = resnet18_custom(8, 8, 36)

checkpoint = torch.load('custom_model.pth')
model.load_state_dict(checkpoint)

model.eval()

for images, labels in dataloader:
    shape_pred, color_pred, symbol_pred = model(images)

    _, shape_predicted = torch.max(shape_pred, 1)
    _, color_predicted = torch.max(color_pred, 1)
    _, symbol_predicted = torch.max(symbol_pred, 1)

    print(labels)

    print(list(shapes)[shape_predicted])
    print(list(colors_dict)[color_predicted])
    print(list(symbols)[symbol_predicted])

