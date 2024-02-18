# Airdrop location detector
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch 
import os
from PIL import Image
import pdb 



#TODO: change this class to take in all of the images in drop zone folder assign them to 'image' variable. 
# Also take all of the ground truths' from the metadata folder and create a label variable for each image containg a list of 5 tuples for x,y pixel coordinates
# labels: [(x1, y1), (x2, y2), ... (x5, y5)]

### example config txt file ###
#test_rectangle_green_W_orange.png placed at (294,3167)
#test_triangle_brown_W_white.png placed at (5226,881)
#test_triangle_blue_N_red.png placed at (7766,790)
#test_semicircle_purple_T_white.png placed at (2717,3669)
#test_pentagon_blue_G_brown.png placed at (1318,2226)

class dataset_parser(Dataset):
    def __init__(self, root_dir, transform=None):
        # should be drop zone folder directory
        self.root_dir = root_dir
        
        # the paths to the images
        self.images_list = os.listdir(root_dir + "/images")
        self.labels_list = os.listdir(root_dir + "/metadata")
        #self.image = self.images_list[0]

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir + "/images", self.images_list[idx])
        image = Image.open(img_name).convert("RGB")

        # Extract labels from the filename
        label_path = os.path.join(self.root_dir + "/images", self.labels_list[idx])
        file = open(label_path, 'r')
        lines = file.readlines()
        file.close()
        curr_label = torch.empty()
        for line in lines:
            line = (line.partition("(")[2]).partition(")")[0]
            curr_label.append(line.partition(",")[0], line.parition(",")[2])

        if self.transform:
            image = self.transform(image)

        return image, curr_label
    
transform = transforms.Compose([
    transforms.ToTensor()
])
    
    
    
class resnet18_custom(nn.Module):
    def __init__(self, out_shape):
        super(resnet18_custom, self).__init__()

        self.features = models.resnet18(pretrained=True) 
        self.features.fc = nn.Identity()  
        
        self.fc_shape = nn.Linear(512, out_shape)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.fc_shape(x) # output is 2,5 tensor
        return out
    
    





def train(image_path, num_epochs):
    
    # prepare data 
    dataset = dataset_parser(root_dir=image_path, transform=transform)
    pdb.set_trace()
    train_perc = 0.7
    train_size = int(train_perc * len(dataset))
    val_size = int((1 - train_perc - 0.15) * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset,[train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

    
    out_shape = torch.tensor((2,5))
    model = resnet18_custom(out_shape).to('cuda')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_dataloader:
            optimizer.zero_grad()
            images, labels = images.to('cuda'), labels.to('cuda')
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images, labels in val_dataloader:
                images, labels = images.to('cuda'), labels.to('cuda')
                preds = model(images)
                val_los2 += criterion(preds, labels)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_dataloader):.4f}')



    return model

def find_drop_locations(model, image):
    drop_locations = model(image)    
    
    return drop_locations




def highlight_drop_locations(image_path, drop_locations):
    image = cv2.imread(image_path)

    for location in drop_locations:
        square_half_len = 35
        st = (location[0] - square_half_len, location[1] - square_half_len)
        ed = (location[0] + square_half_len, location[1] + square_half_len)
        cv2.rectangle(image, st, ed, (220, 220, 220), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Drop Locations")
    plt.show()

if __name__ == "__main__":
    #file_name = "exmple-airdrop-config-5-smaller.png"
    file_name = ".data/full_drop_zone"

    image_path = file_name

    # Find drop locations
    num_epochs = 1
    model = train(image_path, num_epochs)
    drop_locations = find_drop_locations(image_path)
    print("Drop Locations:")
    for i, location in enumerate(drop_locations):
        print(f"Drop {i+1}: ({location[0]}, {location[1]})")

    # Overlay/highlight drop locations 
    highlight_drop_locations(image_path, drop_locations)



