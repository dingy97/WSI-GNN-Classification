import os
import openslide
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

class WSIDataset(Dataset):
    def __init__(self, data_dir, patch_size=256, level=0):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.level = level
        self.slide_files = self._get_slide_files()
        self.patches = self._preprocess_slides()
        
    def _get_slide_files(self):
        return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                if f.endswith('.ndpi')]
    
    def _preprocess_slides(self):
        patches = []
        for slide_path in tqdm(self.slide_files):
            slide = openslide.OpenSlide(slide_path)
            dimensions = slide.level_dimensions[self.level]
            
            # Generate patch coordinates
            for y in range(0, dimensions[1], self.patch_size):
                for x in range(0, dimensions[0], self.patch_size):
                    patches.append({
                        'slide': slide,
                        'coords': (x, y),
                        'label': 1 if '阳性' in slide_path else 0
                    })
        return patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        slide = patch_info['slide']
        x, y = patch_info['coords']
        
        # Read patch from slide
        patch = slide.read_region((x, y), self.level, 
                                 (self.patch_size, self.patch_size))
        patch = patch.convert('RGB')
        
        # Apply transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(patch), patch_info['label']
