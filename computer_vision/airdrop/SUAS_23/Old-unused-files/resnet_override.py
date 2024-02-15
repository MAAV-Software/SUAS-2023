import torch.nn as nn
import torchvision.models as models

class MultiLabelResNet(nn.Module):
    def __init__(self, base_model, num_colors, num_shapes, num_letters):
        super(MultiLabelResNet, self).__init__()
        self.base_model = base_model

        # Remove the last fully connected layer
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])

        # Define new fully connected layers for each attribute
        self.color_classifier = nn.Linear(base_model.fc.in_features, num_colors)
        self.shape_classifier = nn.Linear(base_model.fc.in_features, num_shapes)
        self.letter_classifier = nn.Linear(base_model.fc.in_features, num_letters)
        self.letter_color_classifier = nn.Linear(base_model.fc.in_features, num_colors)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        # Classify each attribute
        color = self.color_classifier(x)
        shape = self.shape_classifier(x)
        letter = self.letter_classifier(x)
        letter_color = self.letter_color_classifier()
        
        return color, shape, letter, letter_color
