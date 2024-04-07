import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2

import matplotlib.pyplot as plt
import pdb

from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to the data
data_dir = ".data/full_drop_zone"
image_dir = os.path.join(data_dir, "images")
metadata_dir = os.path.join(data_dir, "metadata")
results_dir = os.path.join(data_dir, "results")

# Define hyperparameters
batch_size = 8
lr = 0.05
num_epochs = 4

# Define custom dataset class
class DropZoneDataset(Dataset):
    def __init__(self, image_dir, metadata_dir, transform=None):
        self.image_dir = image_dir
        self.metadata_dir = metadata_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        metadata_path = os.path.join(self.metadata_dir, image_name.replace(".png", ".txt"))

        # resize image 
        image = cv2.imread(image_path)
        #print(image.shape)
        h, w, dims = image.shape
        #pdb.set_trace()
        image = cv2.resize(image, (1000, 480))  # h, w

        # metadata
        with open(metadata_path, 'r') as f:
            data = f.readlines()
        drop_locations = []
        for line in data:
            coords = line.split("(")[1].split(")")[0].split(",")
            y, x = int(coords[0]), int(coords[1])
            drop_locations.append((x/10, y/10))

        # Padding to resolve error
        while len(drop_locations) < 5:
            drop_locations.append((0, 0))  

        sample = {'image': image, 'drop_locations': drop_locations}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, drop_locations = sample['image'], sample['drop_locations']
        image = image.transpose((2, 0, 1))
        return {'image': torch.tensor(image, dtype=torch.float).to(device),
                'drop_locations': torch.tensor(drop_locations, dtype=torch.float).to(device)}


class DropZoneModel(nn.Module):
    def __init__(self):
        super(DropZoneModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 10)  # 5*2 # Output layer gives x, y coords

    def forward(self, x):
        #print("this is x input: ", x.shape)
        x = self.resnet(x)
        x = self.fc(x)
        x = x.view(-1, 5, 2)
        #print("this is x output: ", x.shape)
        return x.to(device)



def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['image'], data['drop_locations']
            optimizer.zero_grad()

            outputs = model(inputs)
            #outputs = model(inputs.flip(-1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")



def find_drop_locations(model, test_loader):
    model.eval()
    predicted_locations_list = []
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, _ = data['image'], data['drop_locations']
            outputs = model(inputs)
            predicted_locations = outputs.cpu().numpy()
            #print("predictions: ", outputs)

            predicted_locations_list.append(predicted_locations)
    return predicted_locations_list



def highlight_drop_locations(model, test_dataset, predicted_locations_list):
    # for i in range(len(test_dataset)):
    #     image, _ = test_dataset[i]['image'], test_dataset[i]['drop_locations']
    #     predicted_locations = predicted_locations_list[i]
    #     #image = np.transpose(image, (1, 2, 0))
        
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     for loc in predicted_locations:
    #         cv2.rectangle(image, (int(loc[0])-5, int(loc[1])-5), (int(loc[0])+5, int(loc[1])+5), (255, 0, 0), 2)
    #     cv2.imwrite(os.path.join(results_dir, f"result_{i}.png"), image)
        
    
    # image = cv2.imread(".data/full_drop_zone/images/config_0.png")
    # image = cv2.resize(image, (224, 224))
    # image = image.transpose((2, 0, 1))
    image = test_dataset[0]
    inputs, gt_lbl = image['image'], image['drop_locations']
    inputs = inputs.unsqueeze(0)
    pred_location = model(inputs)
    square_half_len = 25
    pred_location = pred_location.squeeze()
    #pdb.set_trace()
    
    image = inputs.squeeze().cpu().numpy().transpose(1,2,0) #change this as well
    image = np.ascontiguousarray(image)
    
    #pred_location_np = pred_location.detach().cpu().numpy()
    #pdb.set_trace()
    for loc in pred_location:
        st = (int(loc[1]) - square_half_len, int(loc[0]) - square_half_len)
        ed = (int(loc[1]) + square_half_len, int(loc[0]) + square_half_len)
        cv2.rectangle(image, st, ed, (0, 0, 230), 10)

    print("predicted location: ", pred_location)
        
    # os.makedirs(results_dir, exist_ok=True)
    # output_path = os.path.join(results_dir, 'results.jpg')
    # plt.imsave(output_path, image)
    
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Drop Locations")
    plt.show()




def main():
    dataset = DropZoneDataset(image_dir=image_dir, metadata_dir=metadata_dir, transform=ToTensor())
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Define the model, loss function, and optimizer
    model = DropZoneModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs)

    # Find drop locations
    predicted_locations_list = find_drop_locations(model, test_loader)

    # Highlight drop locations on images and save
    highlight_drop_locations(model, test_dataset, predicted_locations_list)
    print("Evaluation complete. Results saved in 'data/full_drop_zone/results'")

if __name__ == "__main__":
    main()
