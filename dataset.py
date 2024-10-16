import pandas as pd
import torch
from PIL import Image
import numpy as np
import os

class CraterBoulderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, transforms=None):
        self.root_dir = root_dir
        self.csv_file = pd.read_csv(csv_file)
        self.transforms = transforms
        
        # Mapping class strings to integers
        self.class_mapping = {
            'crater': 1,
            'boulder': 2,
            'Boulder': 2
        }

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        current_row = self.csv_file.iloc[idx]
        image_filename = current_row['filename']
        
        # Load the image
        img_name = os.path.join(self.root_dir, 'images', image_filename)
        print(f"Loading image: {img_name}")
        image = Image.open(img_name).convert("RGB")
        
        # Initialize lists to hold boxes, labels, and masks
        boxes = []
        labels = []
        masks = []
        
        # Get all annotations for the current image
        current_image_rows = self.csv_file[self.csv_file['filename'] == image_filename]
        
        for _, row in current_image_rows.iterrows():
            # Read bounding box coordinates
            xmin = float(row['xmin'])  # xmin
            ymin = float(row['ymin'])  # ymin
            xmax = float(row['xmax'])  # xmax
            ymax = float(row['ymax'])  # ymax
            
            # Check for valid bounding box dimensions (xmax > xmin, ymax > ymin)
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                print(f"Invalid bounding box found: {[xmin, ymin, xmax, ymax]} for image {image_filename}")
                continue  # Skip invalid bounding boxes

            # Map the class name to an integer label
            class_name = row['class'].lower()
            class_label = self.class_mapping.get(class_name)
            if class_label is None:
                raise ValueError(f"Class '{class_name}' is not in the mapping.")
            labels.append(class_label)

            # Use the full mask path from the CSV directly
            mask_path = row['mask']
            print(f"Loading mask: {mask_path}")
            
            # Load the .npy mask file
            try:
                mask = np.load(mask_path)
                masks.append(torch.as_tensor(mask, dtype=torch.uint8))
            except FileNotFoundError:
                print(f"Mask file not found: {mask_path}")
                masks.append(torch.zeros((640, 640), dtype=torch.uint8))  # Provide a default mask if the file is not found

        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Concatenate masks into a single tensor, or use a zero tensor if no masks found
        masks = torch.stack(masks) if masks else torch.zeros((0, 640, 640), dtype=torch.uint8)

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks  # Return concatenated masks
        }

        # Print number of boxes and masks found
        print(f"Bounding boxes found: {len(boxes)}")
        print(f"Number of masks loaded: {masks.shape[0]}")

        # Apply transformations if any
        if self.transforms:
            image = self.transforms(image)

        return image, target