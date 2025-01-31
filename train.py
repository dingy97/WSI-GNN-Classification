import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
from utils.data_loader import WSIDataset
from models.gnn_model import WSIGraphSAGE
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def build_graph(patch_coords, patch_size):
    # Build adjacency matrix based on spatial proximity
    edge_index = []
    for i, (x1, y1) in enumerate(patch_coords):
        for j, (x2, y2) in enumerate(patch_coords):
            if i != j and abs(x1 - x2) <= patch_size and abs(y1 - y2) <= patch_size:
                edge_index.append([i, j])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def train():
    # Hyperparameters
    batch_size = 4
    lr = 1e-4
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare datasets
    train_dataset = WSIDataset('F:/食管癌阳性')
    val_dataset = WSIDataset('F:/食管癌阴性')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = WSIGraphSAGE().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        for patches, labels in train_loader:
            # Build graph for each WSI
            patch_coords = [p['coords'] for p in patches]
            edge_index = build_graph(patch_coords, train_dataset.patch_size)
            
            # Prepare batch
            batch = Batch.from_data_list([
                Data(x=patch, edge_index=edge_index) 
                for patch in patches
            ]).to(device)
            
            # Forward pass
            outputs = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(outputs, labels.to(device))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        
        # Print epoch stats
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Acc: {val_metrics["acc"]:.4f}, Val F1: {val_metrics["f1"]:.4f}')
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Update learning rate
        scheduler.step(val_metrics['f1'])

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for patches, labels in loader:
            # Build graph
            patch_coords = [p['coords'] for p in patches]
            edge_index = build_graph(patch_coords, loader.dataset.patch_size)
            
            # Prepare batch
            batch = Batch.from_data_list([
                Data(x=patch, edge_index=edge_index) 
                for patch in patches
            ]).to(device)
            
            # Forward pass
            outputs = model(batch.x, batch.edge_index, batch.batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    return {
        'acc': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }

if __name__ == '__main__':
    train()
