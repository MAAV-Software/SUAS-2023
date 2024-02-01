import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Ruleset for the competition
shapes = [
    "Circle",
    "Semi_Circle",
    "Quarter_Circle",
    "Star",
    "Pentagon",
    "Triangle",
    "Rectangle",
    "Cross",
]
colors_dict = {
    "White": (255, 255, 255),  # White
    "Black": (0, 0, 0),        # Black
    "Red": (255, 0, 0),      # Red
    "Blue": (0, 0, 255),      # Blue
    "Green": (0, 255, 0),      # Green
    "Purple": (127, 0, 255),    # Purple
    "Brown": (102, 51, 0),     # Brown
    "Orange": (255, 128, 0),    # Orange
}
symbols_dict = {
    "sym_A": "A", "sym_B": "B", "sym_C": "C", "sym_D": "D", "sym_E": "E", "sym_F": "F", "sym_G": "G", "sym_H": "H", "sym_I": "I", "sym_J": "J", "sym_K": "K", "sym_L": "L", "sym_M": "M",
    "sym_N": "N", "sym_O": "O", "sym_P": "P", "sym_Q": "Q", "sym_R": "R", "sym_S": "S", "sym_T": "T", "sym_U": "U", "sym_V": "V", "sym_W": "W", "sym_X": "X", "sym_Y": "Y", "sym_Z": "Z",
    "sym_0": "0", "sym_1": "1", "sym_2": "2", "sym_3": "3", "sym_4": "4", "sym_5": "5", "sym_6": "6", "sym_7": "7", "sym_8": "8", "sym_9": "9"
}
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.data_dir, img_name)

        # Extract information from the image name
        parts = img_name.split('_')
        shape = parts[1]
        shape_color = colors_dict[parts[2].capitalize()]
        character = symbols_dict[str(parts[3])]

        character_color = colors_dict[parts[4].split('.')[0]]

        # Load and transform the image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Return image and labels
        return img, {
            "shape": shape,
            "shape_color": shape_color,
            "character": character,
            "character_color": character_color
        }

# Data transformations
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.229, 0.224, 0.225])
])

# Model definition
class CustomModel(nn.Module):
    def __init__(self, num_shapes, num_colors, num_symbols):
        super(CustomModel, self).__init__()
        # Load pre-trained ResNet and modify the final fully connected layer
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_shapes + num_colors + num_symbols)

    def forward(self, x):
        return self.resnet(x)

# Model parameters
num_shapes = len(shapes)
num_colors = len(colors_dict)
num_symbols = len(symbols_dict)

# Create an instance of the model
model = CustomModel(num_shapes, num_colors, num_symbols)

# Define your data directories
training_data_dir = ".data/training_images"

# Create an instance of the dataset
custom_dataset = CustomDataset(training_data_dir, transform=data_transforms)

# Split dataset into training and validation sets
train_size = int(0.8 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # Modify the labels to match the model's output
        labels_combined = torch.cat([
            torch.tensor(shapes.index(labels['shape']), device=device),
            torch.tensor(colors_dict[labels['shape_color']], device=device),
            torch.tensor(symbols_dict[labels['character']], device=device)
        ])
        loss = criterion(outputs, labels_combined)
        loss.backward()
        optimizer.step()

    # Validation (similar adjustments as training loop)
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels_combined = torch.cat([
                torch.tensor(shapes.index(labels['shape']), device=device),
                torch.tensor(colors_dict[labels['shape_color']], device=device),
                torch.tensor(symbols_dict[labels['character']], device=device)
            ])
            loss = criterion(outputs, labels_combined)

    scheduler.step()
    print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'path/to/fine_tuned_model.pth')
