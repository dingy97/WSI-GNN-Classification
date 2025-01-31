import torch
from torch_geometric.data import Data, Batch
from models.gnn_model import WSIGraphSAGE
from utils.data_loader import WSIDataset
import openslide
import numpy as np

class WSIPredictor:
    def __init__(self, model_path, patch_size=256):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = WSIGraphSAGE().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.patch_size = patch_size
        
    def predict(self, slide_path):
        # Load slide
        slide = openslide.OpenSlide(slide_path)
        level = 0
        dimensions = slide.level_dimensions[level]
        
        # Extract patches
        patches = []
        coords = []
        for y in range(0, dimensions[1], self.patch_size):
            for x in range(0, dimensions[0], self.patch_size):
                patch = slide.read_region((x, y), level, 
                                       (self.patch_size, self.patch_size))
                patch = patch.convert('RGB')
                patches.append(patch)
                coords.append((x, y))
        
        # Build graph
        edge_index = self._build_graph(coords)
        
        # Prepare batch
        batch = Batch.from_data_list([
            Data(x=patch, edge_index=edge_index) 
            for patch in patches
        ]).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().numpy()
            
        # Aggregate predictions
        positive_ratio = np.mean(pred)
        final_pred = 1 if positive_ratio > 0.5 else 0
        confidence = max(positive_ratio, 1 - positive_ratio)
        
        return {
            'prediction': '阳性' if final_pred == 1 else '阴性',
            'confidence': float(confidence),
            'positive_ratio': float(positive_ratio)
        }
    
    def _build_graph(self, coords):
        edge_index = []
        for i, (x1, y1) in enumerate(coords):
            for j, (x2, y2) in enumerate(coords):
                if i != j and abs(x1 - x2) <= self.patch_size and abs(y1 - y2) <= self.patch_size:
                    edge_index.append([i, j])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

if __name__ == '__main__':
    predictor = WSIPredictor('best_model.pth')
    result = predictor.predict('path_to_your_slide.ndpi')
    print(f"预测结果: {result['prediction']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"阳性比例: {result['positive_ratio']:.2f}")
