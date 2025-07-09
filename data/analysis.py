import os

# Define dataset paths
paths = {
    "normalTest": "./chest_xray/test/NORMAL/",
    "normalTrain": "./chest_xray/train/NORMAL/",
    "normalVal": "./chest_xray/val/NORMAL/",
    "pneumoniaTest": "./chest_xray/test/PNEUMONIA/",
    "pneumoniaTrain": "./chest_xray/train/PNEUMONIA/",
    "pneumoniaVal": "./chest_xray/val/PNEUMONIA/"
}

# Count files in each directory
counts = {key: len(os.listdir(path)) for key, path in paths.items()}

# Calculate totals
total_samples = sum(counts.values())
normal_total = counts["normalTest"] + counts["normalTrain"] + counts["normalVal"]
pneumonia_total = counts["pneumoniaTest"] + counts["pneumoniaTrain"] + counts["pneumoniaVal"]

# Calculate percentages
normal_percentage = (normal_total / total_samples) * 100 if total_samples > 0 else 0
pneumonia_percentage = (pneumonia_total / total_samples) * 100 if total_samples > 0 else 0

# Print detailed analytics
print(f"Total samples: {total_samples}")
print(f"Normal samples: {normal_total} ({normal_percentage:.2f}%)")
print(f"Pneumonia samples: {pneumonia_total} ({pneumonia_percentage:.2f}%)")
print("\nBreakdown by dataset split:")
for category in ["normal", "pneumonia"]:
    for split in ["Test", "Train", "Val"]:
        key = f"{category}{split}"
        print(f"{key}: {counts[key]}")
