import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from agn_src.VariableGroupNorm import VariableGroupNorm


import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class SimpleNN_GN(nn.Module):
    def __init__(self, num_channels):
        super(SimpleNN_GN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=5)
        self.gn = nn.GroupNorm(num_groups=2, num_channels=num_channels)
        self.fc = nn.Linear(num_channels * 12 * 12, 10)  # Adjust for your dataset and architecture

    def forward(self, x):
        x = F.relu(self.gn(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class SimpleNN_VGN(nn.Module):
    def __init__(self, num_channels, group_sizes):
        super(SimpleNN_VGN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=5)
        self.vgn = VariableGroupNorm(num_channels=num_channels, group_sizes=group_sizes)
        self.fc = nn.Linear(num_channels * 12 * 12, 10)  # Adjust for your dataset and architecture

    def forward(self, x):
        x = F.relu(self.vgn(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def calculate_param_differences(initial_params, updated_params):
    differences = {}
    for (name1, param1), (name2, param2) in zip(initial_params.items(), updated_params.items()):
        differences[name1] = (param2 - param1).abs().mean().item()  # Mean absolute difference
    return differences

# Function to copy weights and biases from one model to another
def copy_model_parameters(source_model, target_model):
    for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
        target_param.data.copy_(source_param.data)

set_seed(0)
# Dataset and DataLoader setup
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model, optimizer, and loss function
num_channels = 8
group_sizes = torch.tensor([4, 4])  # Example group sizes, adjust as needed

model_gn = SimpleNN_GN(num_channels=num_channels)
model_vgn = SimpleNN_VGN(num_channels=num_channels, group_sizes=group_sizes)

# Copy parameters from GN model to VGN model
copy_model_parameters(model_gn, model_vgn)

optimizer_gn = optim.Adam(model_gn.parameters())
optimizer_vgn = optim.Adam(model_vgn.parameters())

loss_function_gn = nn.CrossEntropyLoss()
loss_function_vgn = nn.CrossEntropyLoss()

initial_params_gn = {name: param.clone() for name, param in model_gn.named_parameters() if
                     'weight' in name or 'bias' in name}
initial_params_vgn = {name: param.clone() for name, param in model_vgn.named_parameters() if
                      'weight' in name or 'bias' in name}

# Training loop for a few epochs
num_epochs = 3  # Example, adjust as needed

for epoch in range(num_epochs):
    # Start the epoch for both models
    model_gn.train()
    model_vgn.train()

    for data, target in train_loader:
        # Train model_gn
        optimizer_gn.zero_grad()
        output_gn = model_gn(data)
        loss_gn = loss_function_gn(output_gn, target)
        loss_gn.backward()
        optimizer_gn.step()

        # Train model_vgn with the same data
        optimizer_vgn.zero_grad()
        output_vgn = model_vgn(data)
        loss_vgn = loss_function_vgn(output_vgn, target)
        loss_vgn.backward()
        optimizer_vgn.step()

    # After each epoch, capture and compare parameters for both models
    updated_params_gn = {name: param.clone() for name, param in model_gn.named_parameters() if 'weight' in name or 'bias' in name}
    updated_params_vgn = {name: param.clone() for name, param in model_vgn.named_parameters() if 'weight' in name or 'bias' in name}

    # Calculate differences for GN
    param_differences_gn = calculate_param_differences(initial_params_gn, updated_params_gn)
    print(f'Epoch {epoch}, Model GN: Loss {loss_gn.item()}')
    for name, diff in param_differences_gn.items():
        print(f'GN Parameter: {name}, Change: {diff}')

    # Calculate differences for VGN
    param_differences_vgn = calculate_param_differences(initial_params_vgn, updated_params_vgn)
    print(f'Epoch {epoch}, Model VGN: Loss {loss_vgn.item()}')
    for name, diff in param_differences_vgn.items():
        print(f'VGN Parameter: {name}, Change: {diff}')


# Add your evaluation logic here (e.g., compare loss, accuracy)
