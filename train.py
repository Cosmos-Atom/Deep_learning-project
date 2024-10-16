import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from dataset import CraterBoulderDataset
from utils import collate_fn
from torchvision.transforms import functional as F
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torch.nn as nn

def get_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Add normalization
    ])
loss_function = nn.CrossEntropyLoss()


# def load_data(root_dir, csv_file, batch_size=2, shuffle=True):
    # dataset = CraterBoulderDataset(root_dir=root_dir, csv_file=csv_file, transforms=get_transform())
    # return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=4)

def load_data(root_dir, csv_file, batch_size=2, shuffle=True, limit_data=None):
    dataset = CraterBoulderDataset(root_dir=root_dir, csv_file=csv_file, transforms=get_transform())
    
    # Limit dataset to only the first `limit_data` samples if specified
    if limit_data:
        dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), limit_data)))
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=4)

# Use only 2 images for training and validation datasets
train_loader = load_data(root_dir='Dataset/train', csv_file='Dataset/train/annotations_train_with_masks.csv', limit_data=3)
valid_loader = load_data(root_dir='Dataset/valid', csv_file='Dataset/valid/annotations_valid_with_masks.csv', shuffle=False, limit_data=2)




def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)  # Use weights parameter
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes=3)
model.to(device)

optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)

def train_one_epoch(model, optimizer, dataloader, device):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / len(dataloader)

def validate(model, valid_loader, device):
    model.train()  # Temporarily set the model to train mode to compute loss
    total_loss = 0
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for images, targets in valid_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Explicitly compute loss in training mode
            loss_dict = model(images, targets)
            
            # Calculate the total loss for the batch
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            print(f"Batch loss: {losses.item()}")  # Print the batch loss for debugging

    model.eval()  # Return the model to evaluation mode after validation
    # Return the average validation loss
    return total_loss / len(valid_loader) if len(valid_loader) > 0 else 0




def train_model(model, train_loader, valid_loader, num_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}')
        val_loss = validate(model, valid_loader, device)
        print(f'Validation Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_mask_rcnn_model_epoch_{epoch + 1}.pth')
            print(f"Best model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
    print("Training completed.")

if __name__ == "__main__":
    train_model(model, train_loader, valid_loader, num_epochs=10)
