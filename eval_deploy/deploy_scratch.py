# Third party imports
import torch
import os 
from torchvision.transforms import ToTensor, Resize, Compose
import PIL
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import matplotlib.patches as patches
from torchmetrics.classification import JaccardIndex


# Custom imports
import sys
sys.path.append('../')
# from gen_building_utils import * 
from options.deploy_options import *
from models.predictors import get_predictor_from_options
from eval_deploy import deploy_utils as dutils
from eval_deploy import viz_utils as vutils
from eval_deploy import glocal_utils as glocal
from data_factory import gen_building_utils 
from data_factory import simple_mask_utils as smu

## Problem Inputs 
ensemble_folder_name = 'sc_map_pred_ensemble1' # Path to ensemble folder, in weights folder



# Initialize options
cmd_line_args_dict = ['--name', 'deploy']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ['L2M_root_path'] = '/home/cherie/research/hmap/map_prediction_toolbox/' # change for different computer
options = DeployOptions().parse_args(cmd_line_args_dict)

# Initialize model ensembles # TODO: make into function
ensemble_dir = os.path.join(options.root_path, 'weights', ensemble_folder_name)
assert os.path.exists(ensemble_dir), "Ensemble dir does not exist"
ensemble_exp = os.listdir(ensemble_dir)
ensemble_exp.sort()
models_dict = {}
ensemble_size = len(ensemble_exp)
for n in range(ensemble_size):
    models_dict[n] = {'predictor_model': get_predictor_from_options(options)}
    models_dict[n] = {k:v.to(device) for k,v in models_dict[n].items()}
    models_dict[n]['predictor_model'] = torch.nn.DataParallel(models_dict[n]['predictor_model'])
    checkpoint_dir = ensemble_dir + "/" + ensemble_exp[n]
    latest_checkpoint = dutils.get_latest_model(save_dir=checkpoint_dir)
    models_dict[n] = dutils.load_model(models=models_dict[n], checkpoint_file=latest_checkpoint)
    models_dict[n]["predictor_model"].eval()
    print("Model {}: loading checkpoint {}".format(n, latest_checkpoint))




map_configs = {
    'min_start_end_dist': 20,
    'collect_interval_m': 5,
    'laser_range': 50,
    'num_laser': 500,
    'num_rand_traj_per_map':1,
    'percent_test': 0.2,
    "local_map_size": 30,
} # TODO: move to top of file


# Generate a global ground-truth occupancy grid 
building_occ_map = gen_building_utils.make_building_occ_map()
building_occ_map = smu.convert_012_labels_to_maskutils_labels(building_occ_map)

# Get list of local masks given a random trajectory
# TODO: set it to take in a trajectory instead 
map, mask_list, local_mask_list, local_gt_list, pose_list = smu.make_masklist_with_rand_traj_in_map(building_occ_map, map_configs, show_viz=False)

# Make everything 5 times bigge
multiplier = 5
map = dutils.make_map_larger(map, multiplier)
mask_list = [dutils.make_map_larger(map, multiplier) for map in mask_list]
local_mask_list = [dutils.make_map_larger(map, multiplier) for map in local_mask_list]
local_gt_list = [dutils.make_map_larger(map, multiplier) for map in local_gt_list]
pose_list = pose_list * multiplier


print("Map shape: ", map.shape)
print("Mask list shape: ", mask_list[0].shape)

import numpy as np


## For each point in trajectory, we want to mock the prediction process
###  - get local observed map
###  - get local predicted map
###  - get indices of local predicted map in global map
###  - update global map with local map
###  - visualize global predicted map 

# Intialize transform 
local_transform = Compose([
            Resize((256, 256), interpolation= PIL.Image.NEAREST),
            ToTensor()
        ])

global_transform = Compose([
            ToTensor(),
        ]) # do not do any resizing

# Initialize metric function 
iou_metric = JaccardIndex(task="multiclass", num_classes=3, reduction="none")


# Initialize metric lists for visualization 
iou_metric_obs_list = []
iou_metric_pred_list = []
iou_metric_pred_multi_list = []

# Initialize global maps for accumulation
## Global map predicted multi-patches 
global_map_pred_multi = np.ones_like(map) * 0.5 # init as void
global_pred_multi_onehot_tensor = dutils.convert_maputils_labelmaps_to_model_input_format(global_map_pred_multi, global_transform)

## Global map predicted
global_map_pred = np.ones_like(map) * 0.5 # init as void
global_pred_onehot_tensor = dutils.convert_maputils_labelmaps_to_model_input_format(global_map_pred, global_transform)
## Global map observed
global_map_obs = np.ones_like(map) * 0.5 # init as void
global_obs_onehot_tensor = dutils.convert_maputils_labelmaps_to_model_input_format(global_map_obs, global_transform)


for i in range(len(mask_list)):
    # Get local observed map
    local_observed_map = 1-local_mask_list[i]
    global_observed_map = 1-mask_list[i]

    # Get local predicted map    
    local_input_onehot_tensor = dutils.convert_maputils_labelmaps_to_model_input_format(local_observed_map, local_transform)
    local_pred_mean_onehot, local_pred_ensemble_onehot = dutils.run_map_predictor(local_input_onehot_tensor, device, models_dict)

    # Get local GT map (for visualization)
    local_gt_map = 1-local_gt_list[i]
    local_gt_onehot_tensor = dutils.convert_maputils_labelmaps_to_model_input_format(local_gt_map, local_transform)

    # Get global GT map (for visualization)
    global_map_gt = map 
    global_gt_onehot_tensor = dutils.convert_maputils_labelmaps_to_model_input_format(global_map_gt, global_transform)

    # Accumulate global observed map
    # # Get indices of local observed map in global map, given pose_list
    local_map_center_in_global = pose_list[i]
    # print("local_map_center_in_global: ", local_map_center_in_global)
    # print("global_map_obs.shape: ", global_map_obs.shape)
    # print("global_obs_onehot_tensor.shape: ", global_obs_onehot_tensor.shape)
    # print("local_pred_mean_onehot.shape: ", local_pred_mean_onehot.shape)
    
    # # Update global observed map with newest global observed map 
    global_obs_onehot_tensor = dutils.convert_maputils_labelmaps_to_model_input_format(global_observed_map, global_transform)

    # # Update global predicted map with local predicted map
    global_pred_onehot_tensor = dutils.update_global_map_with_local(global_pred_onehot_tensor, local_pred_mean_onehot, 
                                 robot_pose=local_map_center_in_global, local_map_size=map_configs['local_map_size']*multiplier)

    # # Update global predicted multi map with multi-patch inference
    stride_m = 10
    global_pred_multi_onehot_tensor = glocal.calculate_global_pred_with_glocal_inference(global_obs_onehot_tensor,
                                                    global_pred_multi_onehot_tensor, 
                                                    map_configs, multiplier, 
                                                    stride_m, models_dict, 
                                                    device, show_viz=False)
    
    # # Calculate metrics
    iou_metric_obs = dutils.get_freq_iou(iou_metric, global_obs_onehot_tensor, global_gt_onehot_tensor)
    iou_metric_obs_list.append(iou_metric_obs)

    iou_metric_pred = dutils.get_freq_iou(iou_metric, global_pred_onehot_tensor, global_gt_onehot_tensor)
    # print("iou_metric_pred: ", iou_metric_pred)
    iou_metric_pred_list.append(iou_metric_pred)

    iou_metric_pred_multi = dutils.get_freq_iou(iou_metric, global_pred_multi_onehot_tensor, global_gt_onehot_tensor)
    # print("iou_metric_pred_multi: ", iou_metric_pred_multi)
    iou_metric_pred_multi_list.append(iou_metric_pred_multi)

    # Converting one hot tensors to colorized and then visualize 
    plt_row = 3
    plt_col = 3
    plt.figure(figsize=(15,10))
    plt.subplot(plt_row,plt_col,1)
    # plt.imshow(local_observed_map)
    colorized_input = vutils.get_colorized_map(local_input_onehot_tensor)
    colorized_input_permuted = colorized_input.permute(1,2,0).squeeze().detach().cpu().numpy()
    plt.imshow(colorized_input_permuted)
    plt.scatter(colorized_input_permuted.shape[1]//2, colorized_input_permuted.shape[0]//2,c='r', s=10) # robot pose
    plt.title("Local map input")

    plt.subplot(plt_row,plt_col,2)
    colorized_mean = vutils.get_colorized_map(local_pred_mean_onehot)
    plt.imshow(colorized_mean.permute(1,2,0).squeeze().detach().cpu().numpy())
    plt.scatter(colorized_input_permuted.shape[1]//2, colorized_input_permuted.shape[0]//2,c='r', s=10) # robot pose
    plt.title("Local map predicted")

    plt.subplot(plt_row,plt_col,3)
    colorized_gt = vutils.get_colorized_map(local_gt_onehot_tensor)
    plt.imshow(colorized_gt.permute(1,2,0).squeeze().detach().cpu().numpy())
    plt.scatter(colorized_input_permuted.shape[1]//2, colorized_input_permuted.shape[0]//2,c='r', s=10) # robot pose
    plt.title("Local map GT")

    plt.subplot(plt_row,plt_col,4)
    colorized_obs_global = vutils.get_colorized_map(global_obs_onehot_tensor)
    plt.imshow(colorized_obs_global.permute(1,2,0).squeeze().detach().cpu().numpy())
    plt.title("Global map observed")

    plt.subplot(plt_row,plt_col,5)
    colorized_pred_global = vutils.get_colorized_map(global_pred_onehot_tensor)
    plt.imshow(colorized_pred_global.permute(1,2,0).squeeze().detach().cpu().numpy())
    plt.title("Global map predicted")

    plt.subplot(plt_row,plt_col,6)
    colorized_pred_global_mult = vutils.get_colorized_map(global_pred_multi_onehot_tensor)
    plt.imshow(colorized_pred_global_mult.permute(1,2,0).squeeze().detach().cpu().numpy())
    plt.title("Global map predicted_mult")


    plt.subplot(plt_row,plt_col,7)
    colorized_gt_global = vutils.get_colorized_map(global_gt_onehot_tensor)
    plt.imshow(colorized_gt_global.permute(1,2,0).squeeze().detach().cpu().numpy()) 
    x_pose = pose_list[i,0]
    y_pose = pose_list[i,1]
    
    plt.scatter(y_pose, x_pose,c='r',marker='x', s=100) # robot pose 
    plt.plot(pose_list[:i+1,1], pose_list[:i+1,0],'r.') # robot past traj 
    local_map_size_multiplied = map_configs['local_map_size'] * multiplier
    rect = patches.Rectangle((y_pose - (local_map_size_multiplied/2), x_pose - (local_map_size_multiplied/2)), local_map_size_multiplied, local_map_size_multiplied, 
                             linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    # make a rectangle
    plt.title("Global map gt")


    ## Metrics 
    plt.subplot(plt_row,plt_col,8)
    plt.plot(iou_metric_obs_list, label="observed")
    plt.plot(iou_metric_pred_list, label="predicted")
    plt.plot(iou_metric_pred_multi_list, label="predicted_multi")
    plt.title("Freq-weighted IOU metric")
    plt.xlim(0, len(mask_list))
    plt.ylim(0, 1)
    plt.legend()
    
    plt.show()
    del colorized_input, colorized_mean, colorized_gt, colorized_obs_global, colorized_pred_global, colorized_gt_global
    torch.cuda.empty_cache()

