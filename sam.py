import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from matplotlib import pyplot as plt
import cv2
import supervision as sv

CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# Give the path of your image
image_path = './KOA_Nassau_2697x1517.jpg'
# Read the image from the path
image = cv2.imread(image_path)
# Convert to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Generate segmentation mask
output_mask = mask_generator.generate(image_rgb)


mask_annotator = sv.MaskAnnotator(color_map="index")
detections = sv.Detections.from_sam(output_mask)
annotated_image = mask_annotator.annotate(image_rgb, detections)