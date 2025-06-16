### Write observed maps from a given directory to data (image, mask) format for LAMA
import os 
import cv2 
from PIL import Image
import numpy as np
from tqdm import tqdm

input_dir = '../global_obs/'
output_dir = '../cherie_data/'

os.makedirs(output_dir, exist_ok=True)
# Read in the observed maps, and write image and masks

# Get the list of observed maps
observed_maps = sorted(os.listdir(input_dir))

for observed_map_name in tqdm(observed_maps):
    # import pdb; pdb.set_trace()
    # Read in the observed map, crops 
    print(f'Processing {observed_map_name}')
    observed_map_full_path = os.path.join(input_dir, observed_map_name)
    observed_map = np.array(Image.open(observed_map_full_path))[650:-650, 650:-650].astype(int)

    # Write the observed map to image and mask
    mask = np.zeros((observed_map.shape[0], observed_map.shape[1]))
    mask[observed_map[:,:,0] == 128] = 255
    
    # Save the image and mask
    observed_map_name_no_ext = os.path.splitext(observed_map_name)[0]
    observed_map_image_path = os.path.join(output_dir, observed_map_name_no_ext + '.png')
    observed_map_mask_path = os.path.join(output_dir, observed_map_name_no_ext + '_mask001.png')
    
    cv2.imwrite(observed_map_image_path, observed_map)
    cv2.imwrite(observed_map_mask_path, mask)
    
    print(f'Writing to {observed_map_image_path} and {observed_map_mask_path}')