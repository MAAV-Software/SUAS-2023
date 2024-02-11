import os
from PIL import Image
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

# Define your transformations (you can modify this based on your needs)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Specify the path to your training images
training_data_path = '.data/training_images'

# Create an instance of the CustomDataset and DataLoader
dataset = dataset_parser(root_dir=training_data_path, shapes=shapes, colors_dict=colors_dict, symbols=symbols, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
#pdb.set_trace()


# Visualization: 
def imshow(img):
    img = img / 2 + 0.5    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 1st batch
for images, labels in dataloader:
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % shapes[labels[j][0]] for j in range(len(labels))))
    print(' '.join('%5s' % list(colors_dict.keys())[labels[j][1]] for j in range(len(labels))))
    print(' '.join('%5s' % symbols[labels[j][2]] for j in range(len(labels))))
    
    # Break to not go through all batches
    break






# Network Training: 

torch.manual_seed(42)

# custom model using resnet
class resnet18_custom(nn.Module):
    def __init__(self, num_shapes, num_colors, num_symbols):
        super(resnet18_custom, self).__init__()

        self.features = models.re`snet18(pretrained=True) 
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



num_shapes = len(shapes) # 8
num_colors = len(colors_dict) # 8
num_symbols = len(symbols) # 36

#model = CustomModel(num_shapes, num_colors, num_symbols)
model = resnet18_custom(num_shapes, num_colors, num_symbols).to('cuda')


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# split into training and validation 
train_percentage = 0.2
val_percentage = 0.1
total_size = len(dataset)
train_size = int(train_percentage * total_size)
val_size = int(val_percentage * total_size)
test_size = total_size - train_size - val_size

# split into train, val, tset 
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# for small testing: 
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)



num_epochs = 1
print("Starting training:")


for epoch in range(num_epochs):
    model.train()
    for images, labels in train_dataloader:
        optimizer.zero_grad()
        images, labels = images.to('cuda'), labels.to('cuda')
        shape_pred, color_pred, symbol_pred = model(images)
        loss_shape = criterion(shape_pred, labels[:, 0])
        loss_color = criterion(color_pred, labels[:, 1])
        loss_symbol = criterion(symbol_pred, labels[:, 2])
        total_loss = loss_shape + loss_color + loss_symbol
        total_loss.backward()
        optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for images, labels in val_dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            shape_pred, color_pred, symbol_pred = model(images)
            loss_shape = criterion(shape_pred, labels[:, 0])
            loss_color = criterion(color_pred, labels[:, 1])
            loss_symbol = criterion(symbol_pred, labels[:, 2])
            val_loss += (loss_shape + loss_color + loss_symbol).item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item():.4f}, Val Loss: {val_loss/len(val_dataloader):.4f}')

# Save the trained model (optional)
torch.save(model.state_dict(), 'custom_model.pth')






# Basic Testing: 
model.eval()
correct_shape, correct_color, correct_symbol = 0, 0, 0
total_samples = 0

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to('cuda'), labels.to('cuda')
        shape_pred, color_pred, symbol_pred = model(images)

        _, shape_predicted = torch.max(shape_pred, 1)
        _, color_predicted = torch.max(color_pred, 1)
        _, symbol_predicted = torch.max(symbol_pred, 1)

        correct_shape += (shape_predicted == labels[:, 0]).sum().item()
        correct_color += (color_predicted == labels[:, 1]).sum().item()
        correct_symbol += (symbol_predicted == labels[:, 2]).sum().item()

        total_samples += labels.size(0)

accuracy_shape = correct_shape / total_samples
accuracy_color = correct_color / total_samples
accuracy_symbol = correct_symbol / total_samples

print(f'Test Accuracy - Shape: {accuracy_shape:.4f}, Color: {accuracy_color:.4f}, Symbol: {accuracy_symbol:.4f}')