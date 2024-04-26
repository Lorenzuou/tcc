
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib

from matplotlib import pyplot as plt
import cv2
import supervision as sv
import os

CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

sam_masks_dir = './sam_masks/'
os.makedirs(sam_masks_dir, exist_ok=True)
real_images_folder = './pad/train/data/'



def get_second_largest_area(result_dict):

    return result_dict

    sorted_result = sorted(result_dict, key=(lambda x: x['area']),      reverse=True)
    return sorted_result[1]


def get_most_probable_area(result_dict, shape):
    #sort the result dict by area
    sorted_result = sorted(result_dict, key=(lambda x: x['area']),      reverse=True)
    # # return the one that has the largest area and does not touch the edges of the image
    # for val in sorted_result:
    #     bbox = val['bbox']
    #     print(bbox)
    #     print(shape)

    #     if bbox[0] > 0 and bbox[1] > 0 and bbox[2] < shape[1] -10 and bbox[3] < shape[0] -10: 
    #         return val
        
    return sorted_result[0]

#for all folders in real_images_folder
for folder in os.listdir(real_images_folder):
    print(folder)
    #make a dir for the masks
    dir_to_save = os.path.join(sam_masks_dir, folder)
    os.makedirs(dir_to_save, exist_ok=True)
    for image_name in os.listdir(os.path.join(real_images_folder, folder)):
        if (image_name[:-4] + "_segmentation.png")  in os.listdir(dir_to_save):
            print("Already segmented")
            continue
        # Read the image from the path
        image = cv2.imread(os.path.join(real_images_folder, folder, image_name))

        if image.shape[0] > 1920 or image.shape[1] > 1080:
            if image.shape[0] > 1920:
                image = cv2.resize(image, (1920, int(1920 * image.shape[1] / image.shape[0])))
            elif image.shape[1] > 1080:
                image = cv2.resize(image, (int(1080 * image.shape[0] / image.shape[1]), 1080))

            
        # Convert to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Generate segmentation mask
        output_mask = mask_generator.generate(image)
        #get second largest area
        second_largest_area = get_most_probable_area(output_mask, image.shape)
        mask = second_largest_area['segmentation']
        mask = np.logical_not(mask)
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask = mask * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        #save the mask
        
        cv2.imwrite(dir_to_save + "/" + image_name.replace('.jpg', '_segmentation.png'), mask)
        
    

