import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import ruleset

class AirDropSimulatedDataset(Dataset):
    def __init__(self, img_dir=".data/", transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image, return_tensors="pt")
        
        # Extract label from filename
        label = self.images[idx].split('_') # Adjust according to your naming convention
        _, shape_label, color_label, letter_label, letter_color_label = label

        # Convert labels to tensors if they aren't already
        color_label = torch.tensor(ruleset.color_labels.index(color_label), dtype=torch.long)
        shape_label = torch.tensor(ruleset.shapes.index(shape_label), dtype=torch.long)
        letter_label = torch.tensor(ruleset.symbols.index(letter_label), dtype=torch.long)
        letter_color_label = torch.tensor(ruleset.color_labels.index(letter_color_label.split(".")[0]), dtype=torch.long)

        return image, (color_label, shape_label, letter_label, letter_color_label)
