import os
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
import yaml
from torch.utils.data._utils.collate import default_collate

import sys
sys.path.append('../lama')
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.training.data.datasets import get_transforms
from saicinpainting.evaluation.utils import move_to_device

def get_lama_transform(transform_variant, out_size):
    map_transform = get_transforms(transform_variant, out_size)
    return map_transform

def convert_obsimg_to_model_input(obs_img, map_transform, device):
    # Transform the observed image, gets masks, move to device
    transformed = map_transform(image=obs_img, obs_img=obs_img)
    obs_img = np.transpose(transformed['obs_img'], (2, 0, 1))
    mask = ((obs_img[0] > 0.49) & (obs_img[0] < 0.51)).astype(np.float32)[None, ...]
    input_batch = default_collate([{'image': obs_img, 'mask': mask}])
    input_batch = move_to_device(input_batch, device)
    return input_batch, mask

# Visualization
def load_lama_model(model_path, checkpoint_name='best.ckpt', device='cuda'):
    train_config_path = os.path.join(model_path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    
    checkpoint_path = os.path.join(model_path, 'models', checkpoint_name)
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'
    
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=device).to(device)
    model.freeze()
    return model


def visualize_prediction(batch_pred, mask):
    cur_gt = batch_pred['image'][0].permute(1, 2, 0).cpu().numpy()
    cur_res = np.clip(batch_pred['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy() * 255, 0, 255).astype('uint8')
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    cur_gt = np.clip(cur_gt * 255, 0, 255).astype('uint8')
    cur_gt = cv2.cvtColor(cur_gt, cv2.COLOR_RGB2BGR)

    gt_masked = cur_gt.copy()
    gt_masked[mask[0] > 0] = 122
    disp_output = np.vstack([cur_gt, gt_masked, cur_res])
    cv2.imwrite('lama_pred.png', disp_output)
    return cur_res