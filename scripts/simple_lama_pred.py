"""
Simple LAMA Prediction Script
Processes experiment folders and generates LAMA predictions for observed maps.

Usage:
    python simple_lama_pred.py --model_path <path> --input_folder <path> [options]
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import yaml
from torch.utils.data._utils.collate import default_collate
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.measure import block_reduce
import argparse

from lama_pred_utils import load_lama_model, visualize_prediction, get_lama_transform, convert_obsimg_to_model_input


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate LAMA predictions for experiment folders')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to LAMA model (e.g., pretrained_models/weights/big_lama)')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Root folder containing experiment subdirectories')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--num_frames_to_skip', type=int, default=50,
                        help='Process every Nth frame')
    parser.add_argument('--out_size', type=int, nargs=2, default=[512, 512],
                        help='Output size (width height)')
    parser.add_argument('--crop_size', type=int, default=500,
                        help='Crop size for visualization')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Configuration
    modelalltrain_path = args.model_path
    input_experiment_root_folder = args.input_folder
    device = args.device
    num_frames_to_skip = args.num_frames_to_skip
    out_size = tuple(args.out_size)
    crop = args.crop_size
    
    input_exp_names = sorted(os.listdir(input_experiment_root_folder))
    print(f"Processing {len(input_exp_names)} experiment folders")
    
    # Load LAMA model
    transform_variant = 'default_map_eval'
    model_alltrain = load_lama_model(modelalltrain_path, device=device)
    lama_map_transform = get_lama_transform(transform_variant, out_size)
    
    input_experiment_folders = [os.path.join(input_experiment_root_folder, exp_name) for exp_name in input_exp_names]
    for input_exp_i, input_experiment_folder in enumerate(input_experiment_folders):
        print(f"Processing {input_exp_i+1}/{len(input_experiment_folders)}: {input_experiment_folder}")
        assert os.path.exists(input_experiment_folder), "Experiment folder does not exist"
        
        odom_path = os.path.join(input_experiment_folder, 'odom.npy')
        odom = np.load(odom_path)
        input_obsimg_folder_path = os.path.join(input_experiment_folder, 'global_obs')
        gt_path = os.path.join(input_experiment_folder, 'gt_map.png')
        output_folder = os.path.join(input_experiment_folder, 'global_pred')
        os.makedirs(output_folder, exist_ok=True)

        for obsimg_name in tqdm(sorted(os.listdir(input_obsimg_folder_path))[::num_frames_to_skip]):
            input_obsimg_path = os.path.join(input_obsimg_folder_path, obsimg_name)
            frame_num = int(obsimg_name.split('.')[0])
            
            # Load and transform observed image
            obs_img_threechan = cv2.imread(input_obsimg_path)
            obs_img_onechan = cv2.cvtColor(obs_img_threechan, cv2.COLOR_BGR2RGB)[:,:,0]

            # Convert observed image to model input
            input_lama_batch, lama_mask = convert_obsimg_to_model_input(
                np.stack([obs_img_onechan, obs_img_onechan, obs_img_onechan], axis=2), 
                lama_map_transform, device
            )

            # Get prediction from model trained on all data
            lama_pred_alltrain = model_alltrain(input_lama_batch)
            lama_pred_alltrain_viz = visualize_prediction(lama_pred_alltrain, lama_mask)

            # Visualization
            plt_row = 2
            plt_col = 2
            plt.figure(figsize=(10, 10))
            
            plt.subplot(plt_row, plt_col, 1)
            plt.imshow(obs_img_onechan[crop:-crop, crop:-crop], cmap='gray')
            plt.scatter(odom[:frame_num, 1]-crop, odom[:frame_num, 0]-crop, c='r', s=1)
            plt.scatter(odom[frame_num, 1]-crop, odom[frame_num, 0]-crop, c='r', s=10, marker='x')
            plt.title('Observed Image')
            
            plt.subplot(plt_row, plt_col, 2)
            plt.imshow(cv2.imread(gt_path)[crop:-crop, crop:-crop])
            plt.title('Ground Truth')
            
            plt.subplot(plt_row, plt_col, 3)
            plt.imshow(lama_pred_alltrain_viz[crop:-crop, crop:-crop])
            plt.scatter(odom[:frame_num, 1]-crop, odom[:frame_num, 0]-crop, c='r', s=1)
            plt.scatter(odom[frame_num, 1]-crop, odom[frame_num, 0]-crop, c='r', s=10, marker='x')
            plt.title('LAMA Prediction')
            plt.tight_layout()
            
            # Save outputs
            lama_pred_viz_path = os.path.join(output_folder, f'lama_pred_viz_{obsimg_name.split(".")[0]}.png')
            plt.savefig(lama_pred_viz_path)
            plt.close()
            
            pred_path = os.path.join(output_folder, f'{obsimg_name.split(".")[0]}_pred.npy')
            np.save(pred_path, lama_pred_alltrain_viz)

