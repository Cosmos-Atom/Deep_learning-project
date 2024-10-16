import os
import torch
import numpy as np
import cv2
import torchvision
from torchvision import transforms
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from utils import collate_fn
import matplotlib.pyplot as plt


# Load the trained model
def load_model(model_path, num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=False)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

# Function to perform predictions and overlay masks and boxes
def predict_and_overlay(image_path, model, device):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    # Make predictions
    with torch.no_grad():
        predictions = model(image_tensor)

    # Debugging information
    masks = predictions[0]['masks']
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']

    # Print shapes of the predictions
    print("Masks shape:", masks.shape)
    print("Scores:", scores)
    print("Boxes:", boxes)

    # Threshold for detection
    threshold = 0.1  # Reduced threshold for testing
    masks = masks > threshold  # Apply threshold to masks
    masks = masks.squeeze(1).cpu().numpy()  # Remove the channel dimension

    # Load image as a numpy array
    image_np = np.array(image)
    image_np = image_np.astype(np.uint8)  # Ensure the image is of type uint8

    # Overlay masks in red
    red_color = np.array([255, 0, 0], dtype=np.uint8)  # Define red color
    red_image = np.full_like(image_np, red_color)  # Create a full red image of the same size as the original image

    for i in range(masks.shape[0]):
        if scores[i] > threshold:  # Only consider masks above the threshold
            mask = masks[i].astype(np.uint8)  # Get the mask for the current object

            # Create a 3-channel mask by stacking the 2D mask along the third dimension
            mask_3d = np.stack([mask] * 3, axis=-1)

            # Use broadcasting to apply red color only where the mask is true
            image_np = np.where(mask_3d == 1, cv2.addWeighted(image_np, 0.5, red_image, 0.5, 0), image_np)

            # Draw bounding box around the object
            box = boxes[i].cpu().numpy().astype(int)
            cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)  # Red box

    return image_np

  

# Save the output image
def save_output_image(output_image, output_path):
    output_image_pil = Image.fromarray(output_image)
    output_image_pil.save(output_path)

def main():
    # Path configurations
    image_path = 'Dataset\\test\\000009_jpg.rf.55e826085f24bc0d90d4031737d1ec43.jpg'  # Update this path
    model_path = 'best_mask_rcnn_model_epoch_2.pth'  # Update this path (replace X with the epoch number)
    output_path = 'output_image_with_masks_4.jpg'  # Output path for the saved image

    num_classes = 3  

    # Load the model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(model_path, num_classes)
    model.to(device)

    # Perform prediction and overlay
    output_image = predict_and_overlay(image_path, model, device)

    # Save the output image
    save_output_image(output_image, output_path)
    print(f"Output image saved to {output_path}")

if __name__ == "__main__":
    main()
