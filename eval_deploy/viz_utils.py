
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math
import torch
from PIL import Image
from torchvision.utils import make_grid
import random


import sys
sys.path.append('../')


# three label classification (0:void, 1:occupied, 2:free)
label_conversion_40_3 = {-1:0, 0:0, 1:1, 2:2, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:2, 17:2,
                    18:1, 19:1, 20:1, 21:1, 22:1, 23:1, 24:1, 25:1, 26:1, 27:1, 28:1, 29:1, 30:1, 31:1, 32:1,
                    33:1, 34:1, 35:1, 36:1, 37:1, 38:1, 39:1, 40:1}
# color_mapping_3 = {
#     0:(128,128,128), # grey, void
#     1:(0,0,0), # black, occupied
#     2:(255,255,255), # white, free
# }
color_mapping_3 = {
    0:(255,255,255), # white, void
    1:(0,0,255), # blue, occupied
    2:(0,255,0), # green, free
}

def get_colorized_map(map):
    colorized= colorize_grid(map, color_mapping=3)
    colorized_grid = make_grid(colorized[:,0,:,:,:])  # TODO: check 0
    return colorized_grid


def colorize_grid(grid, color_mapping, sample_limit=8): # to pass into tensorboardX video
    # Limit the number of samples
    grid = grid[:sample_limit]

    if grid.dim() == 3: # if gt
        grid = grid.unsqueeze(1)
    grid = grid.unsqueeze(1)
    grid = grid.detach().cpu().numpy()
    grid_img = np.zeros((grid.shape[0], grid.shape[1], grid.shape[3], grid.shape[4], 3),  dtype=np.uint8)
    if grid.shape[2] > 1:
        grid_prob_max = np.amax(grid, axis=2)
        inds = np.asarray(grid_prob_max<=0.33).nonzero()
        grid[inds[0], inds[1], 0, inds[2], inds[3]] = 1
        grid = np.argmax(grid, axis=2)
    else:
        grid = grid.squeeze(2)

    color_mapping = color_mapping_3
    for label in color_mapping.keys():
        grid_img[ grid==label ] = color_mapping[label]
    
    return torch.tensor(grid_img.transpose(0, 1, 4, 2, 3), dtype=torch.uint8)

def display_sample(rgb_obs, depth_obs, savepath=None):
    depth_obs = depth_obs / np.amax(depth_obs) # normalize for visualization
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")
    
    arr = [rgb_img, depth_img]
    
    n=len(arr)
    plt.figure(figsize=(12 ,8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, n, i+1)
        ax.axis('off')
        plt.imshow(data)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=50) # 100
    plt.close()



def save_trajectory(sg, gt_map_initial, goal_pos, stg_pos_list, pose_coords_list, pose_coords_noisy_list, exploration, filename):
    color_occ_grid = colorize_grid(sg.occ_grid.unsqueeze(1), color_mapping=3)
    im_occ_grid = color_occ_grid[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    color_proj_grid = colorize_grid(sg.proj_grid.unsqueeze(1), color_mapping=3)
    im_proj_grid = color_proj_grid[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    color_gt_map_initial = colorize_grid(gt_map_initial.unsqueeze(0).unsqueeze(0), color_mapping=3)
    im_gt_grid = color_gt_map_initial[0,0,:,:,:].permute(1,2,0).cpu().numpy()
    
    init_pose_coords = pose_coords_list[0]

    arr = [ im_occ_grid,
            im_proj_grid, 
            im_gt_grid
            ]
    n=len(arr)
    plt.figure(figsize=(20 ,15))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i+1)
        ax.axis('off')
        plt.imshow(data)
        plt.scatter(init_pose_coords[0,0,0], init_pose_coords[0,0,1], color="green", s=30)
        if not exploration:
            plt.scatter(goal_pos[0,0,0], goal_pos[0,0,1], color="magenta", s=30)
        if i==0:
            # show the short-term goals
            for k in range(len(stg_pos_list)):
                stg_pos = stg_pos_list[k].cpu().numpy()
                plt.scatter(stg_pos[0,0,0], stg_pos[0,0,1], color="black", s=5)
            # overlay the trajectory            
            for j in range(len(pose_coords_list)-1):
                p1 = pose_coords_list[j].squeeze(0).squeeze(0)
                p2 = pose_coords_list[j+1].squeeze(0).squeeze(0)
                plt.plot([ p1[0], p2[0] ], [ p1[1], p2[1] ], color="red")
            
            # add the noisy poses if exist
            if len(pose_coords_noisy_list)!=0:
                for j in range(len(pose_coords_noisy_list)-1):
                    p1 = pose_coords_noisy_list[j].squeeze(0).squeeze(0)
                    p2 = pose_coords_noisy_list[j+1].squeeze(0).squeeze(0)
                    plt.plot([ p1[0], p2[0] ], [ p1[1], p2[1] ], color="green")                

    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()
    


def generate_random_color():
    r, g, b = random.random(), random.random(), random.random()
    return (r, g, b)