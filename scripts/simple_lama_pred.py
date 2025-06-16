import os
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
import yaml
from torch.utils.data._utils.collate import default_collate
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
from skimage.measure import block_reduce
import argparse

from lama_pred_utils import load_lama_model, visualize_prediction, get_lama_transform, convert_obsimg_to_model_input

# Configuration and paths
modelalltrain_path = '/home/seungchan/MapEx/pretrained_models/weights/big_lama' #make sure to customize
input_experiment_root_folder = '/home/seungchan/MapEx/experiments/20250110_test' #make sure to customize
input_exp_names = sorted(os.listdir(input_experiment_root_folder))
num_frames_to_skip = 50

print("Processing # exp folder:", len(input_exp_names))


input_experiment_folders = [os.path.join(input_experiment_root_folder, exp_name) for exp_name in input_exp_names]
for input_exp_i, input_experiment_folder in enumerate(input_experiment_folders):
    print(f"Processing {input_exp_i+1}/{len(input_experiment_folders)}: {input_experiment_folder}")
    transform_variant = 'default_map_eval'
    device = 'cuda'
    out_size = (512, 512)
    assert os.path.exists(input_experiment_folder), "Experiment folder does not exist"
    odom_path = os.path.join(input_experiment_folder, 'odom.npy')
    odom = np.load(odom_path)
    input_obsimg_folder_path = os.path.join(input_experiment_folder, 'global_obs')
    gt_path = os.path.join(input_experiment_folder, 'gt_map.png')
    output_folder = os.path.join(input_experiment_folder, 'global_pred')
    os.makedirs(output_folder, exist_ok=True)

    # Load LAMA models
    model_list = []
    model_alltrain = load_lama_model(modelalltrain_path, device=device)
    num_models = 0

    # Get map transform function
    lama_map_transform = get_lama_transform(transform_variant, out_size)


    for obsimg_name in tqdm(sorted(os.listdir(input_obsimg_folder_path))[::num_frames_to_skip]):
        input_obsimg_path = os.path.join(input_obsimg_folder_path, obsimg_name)
        frame_num = int(obsimg_name.split('.')[0])
        # Load and transform observed image
        obs_img_threechan = cv2.imread(input_obsimg_path)
        obs_img_onechan = cv2.cvtColor(obs_img_threechan, cv2.COLOR_BGR2RGB)[:,:,0]

        # Convert observed image to model input
        input_lama_batch, lama_mask = convert_obsimg_to_model_input(np.stack([obs_img_onechan, obs_img_onechan, obs_img_onechan], axis=2), lama_map_transform, device)

        # Get prediction from model trained on all data
        lama_pred_alltrain = model_alltrain(input_lama_batch)
        lama_pred_alltrain_viz = visualize_prediction(lama_pred_alltrain, lama_mask)
            

        plt_row = 2# 4 + num_pred_lama 
        plt_col = 2
        plt.figure(figsize=(10, 10))
        plt.subplot(plt_row, plt_col, 1)
        plt.imshow(obs_img_onechan[500:-500,500:-500], cmap='gray')
        plt.scatter(odom[:frame_num, 1]-500, odom[:frame_num, 0]-500, c='r', s=1)
        plt.scatter(odom[frame_num, 1]-500, odom[frame_num, 0]-500, c='r', s=10, marker='x')
        plt.title('Observed Image')
        plt.subplot(plt_row, plt_col, 2)
        plt.imshow(cv2.imread(gt_path)[500:-500,500:-500])
        plt.title('Ground Truth')
        plt.subplot(plt_row, plt_col, 3)
        plt.imshow(lama_pred_alltrain_viz[500:-500,500:-500])
        plt.scatter(odom[:frame_num, 1]-500, odom[:frame_num, 0]-500, c='r', s=1)
        plt.scatter(odom[frame_num, 1]-500, odom[frame_num, 0]-500, c='r', s=10, marker='x')
        plt.title('All Train Prediction')
        plt.tight_layout()
        
        lama_pred_viz_path = os.path.join(output_folder, f'lama_pred_viz_{obsimg_name.split(".")[0]}.png')
        plt.savefig(lama_pred_viz_path)
        plt.close()
        
        pred_path = os.path.join(output_folder, f'{obsimg_name.split(".")[0]}_pred.npy')
        # cv2.imwrite(pred_path, lama_pred_clean_viz)
        np.save(pred_path, lama_pred_alltrain_viz)

