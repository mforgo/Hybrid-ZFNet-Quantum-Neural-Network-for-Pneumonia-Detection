import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 1. Define the ZFNet/AlexNet feature extractor
class ZFNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Use AlexNet as a ZFNet proxy (very similar architecture)
        alexnet = models.alexnet(pretrained=True)
        # Freeze convolutional layers
        for param in alexnet.features.parameters():
            param.requires_grad = False
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        # Use all but the last classification layer (output: 4096-dim)
        self.classifier = nn.Sequential(*list(alexnet.classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 2. Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. Feature extraction for a single image
def extract_features(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(input_tensor)
    return features.cpu().numpy().squeeze()

# 4. Batch processing and saving features
def process_and_save_features(image_dir, output_dir, model, device):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        features = extract_features(img_path, model, device)
        np.save(os.path.join(output_dir, img_file + '.npy'), features)
        print(f"Saved: {img_file}, Feature shape: {features.shape}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ZFNetFeatureExtractor().to(device)
    model.eval()

    # Example: process NORMAL class from train split
    image_dir = './data/chest_xray/train/NORMAL'
    output_dir = './data/features/NORMAL'
    process_and_save_features(image_dir, output_dir, model, device)

    # Verify feature dimensions
    sample_feature = np.load(os.path.join(output_dir, os.listdir(output_dir)[0]))
    print(f"Sample feature vector shape: {sample_feature.shape}")  # Should be (4096,)
