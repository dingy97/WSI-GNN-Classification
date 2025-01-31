import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torchvision import models

class PatchFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )
        
    def forward(self, x):
        return self.feature_extractor(x).squeeze()

class WSIGraphSAGE(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=2):
        super().__init__()
        self.patch_extractor = PatchFeatureExtractor()
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_classes)
        )
        
    def forward(self, x, edge_index, batch):
        # Extract patch features
        x = self.patch_extractor(x)
        
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        return self.fc(x)
