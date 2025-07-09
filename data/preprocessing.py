import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd

# 1. Define the ZFNet/AlexNet feature extractor (fixed deprecated parameter)
class ZFNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed: Use weights parameter instead of pretrained
        alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
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
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(input_tensor)
        return features.cpu().numpy().squeeze()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# 4. Process entire dataset
def process_complete_dataset(data_root, output_root, model, device):
    """Process all splits and classes in the dataset"""
    
    splits = ['train', 'test', 'val']
    classes = ['NORMAL', 'PNEUMONIA']
    
    metadata_records = []
    
    for split in splits:
        for class_name in classes:
            input_dir = os.path.join(data_root, split, class_name)
            output_dir = os.path.join(output_root, split, class_name)
            
            if not os.path.exists(input_dir):
                print(f"Warning: {input_dir} does not exist, skipping...")
                continue
                
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get all image files
            image_files = [f for f in os.listdir(input_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Processing {split}/{class_name}: {len(image_files)} images")
            
            # Process each image
            for img_file in tqdm(image_files, desc=f"{split}/{class_name}"):
                img_path = os.path.join(input_dir, img_file)
                output_path = os.path.join(output_dir, img_file.replace('.jpeg', '.npy').replace('.jpg', '.npy').replace('.png', '.npy'))
                
                # Extract features
                features = extract_features(img_path, model, device)
                
                if features is not None:
                    # Save features
                    np.save(output_path, features)
                    
                    # Record metadata
                    metadata_records.append({
                        'image_path': img_path,
                        'feature_path': output_path,
                        'split': split,
                        'class': class_name,
                        'label': 0 if class_name == 'NORMAL' else 1,
                        'feature_shape': features.shape
                    })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(os.path.join(output_root, 'metadata.csv'), index=False)
    
    return metadata_df

# 5. Verification function
def verify_extraction(output_root):
    """Verify the feature extraction results"""
    metadata_path = os.path.join(output_root, 'metadata.csv')
    
    if not os.path.exists(metadata_path):
        print("Metadata file not found. Run extraction first.")
        return
    
    metadata_df = pd.read_csv(metadata_path)
    
    print(f"\n=== EXTRACTION VERIFICATION ===")
    print(f"Total samples processed: {len(metadata_df)}")
    print(f"Samples by split:")
    print(metadata_df['split'].value_counts())
    print(f"Samples by class:")
    print(metadata_df['class'].value_counts())
    
    # Check a few random samples
    sample_idx = 0
    sample_feature_path = metadata_df.iloc[sample_idx]['feature_path']
    
    if os.path.exists(sample_feature_path):
        sample_features = np.load(sample_feature_path)
        print(f"\nSample feature shape: {sample_features.shape}")
        print(f"Feature dtype: {sample_features.dtype}")
        print(f"Feature range: [{sample_features.min():.4f}, {sample_features.max():.4f}]")
    else:
        print(f"Warning: Sample feature file not found: {sample_feature_path}")

if __name__ == '__main__':
    # Configuration
    DATA_ROOT = "./data/chest_xray"  # Your dataset path
    OUTPUT_ROOT = "./data/features"  # Where to save features
    
    # Check if data directory exists
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data directory {DATA_ROOT} not found!")
        print("Please ensure you've downloaded and extracted the chest X-ray dataset.")
        exit(1)
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ZFNetFeatureExtractor().to(device)
    model.eval()
    
    # Process entire dataset
    print("Starting feature extraction for complete dataset...")
    metadata_df = process_complete_dataset(DATA_ROOT, OUTPUT_ROOT, model, device)
    
    # Verify results
    verify_extraction(OUTPUT_ROOT)
    
    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Features saved to: {OUTPUT_ROOT}")
    print(f"Metadata saved to: {os.path.join(OUTPUT_ROOT, 'metadata.csv')}")
