### Script to mock deployment of map propagation.
# Ideally as similar as possible to a ros node. 

### General structure
##  Make a map 
##  Get list of observations given random trajectory 
## For loop 
###   Get global observed map
###   Predict global map with global observed map
###   Visualize global map

# Third party imports
import torch
import os 
from torchvision.transforms import ToTensor, Resize, Compose
import PIL
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import matplotlib.patches as patches
from torchmetrics.classification import JaccardIndex
import numpy as np
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_cdt

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

## Main variables
ensemble_folder_name = 'sc_map_pred_ensemble1' # Path to ensemble folder, in weights folder
map_configs = {
    'min_start_end_dist': 50,
    'collect_interval_m': 5,
    'laser_range': 50,
    'num_laser': 500,
    'num_rand_traj_per_map':1,
    'percent_test': 0.2,
    "local_map_size": 30, # TODO: add multiplier 
}

# Initialize deploy variables given main variables

# Initialize options
cmd_line_args_dict = ['--name', 'deploy']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ['L2M_root_path'] = '/home/seungchan/Documents/map_prediction_toolbox/' # change for different computer
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

# Make a map 
# Generate a global ground-truth occupancy grid 
building_occ_map = gen_building_utils.make_building_occ_map()
building_occ_map = smu.convert_012_labels_to_maskutils_labels(building_occ_map)

# Randomize if transpose 

if np.random.rand() > 0.5:
    building_occ_map = building_occ_map.T
# 

plt.imshow(building_occ_map)
plt.title("Global map")
plt.show()

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

## For loop over trajectory 
local_transform = Compose([
            Resize((256, 256), interpolation= PIL.Image.NEAREST),
            ToTensor()
        ])

global_transform = Compose([
            ToTensor(),
        ]) # do not do any resizing

# Initialize global maps for accumulation
init_value = 2 # init as free space

## Global map observed
global_map_obs = np.ones_like(map) * init_value 
global_obs_onehot_tensor = dutils.convert_maputils_labelmaps_to_model_input_format(global_map_obs, global_transform)

## Global map predicted propagated
global_map_pred_propagated = np.ones_like(map) * init_value
global_pred_propagated_onehot_tensor = dutils.convert_maputils_labelmaps_to_model_input_format(global_map_pred_propagated, global_transform)


## Collect predictions over trajectory
for i in tqdm(range(len(mask_list))):
    global_observed_map = 1-mask_list[i]
    # print("Global observed map shape: ", global_observed_map.shape)

    # # Update global observed map with newest global observed map 
    global_obs_onehot_tensor = dutils.convert_maputils_labelmaps_to_model_input_format(global_observed_map, global_transform)
    # print("global_obs_onehot_tensor.shape: ", global_obs_onehot_tensor.shape)

    # # Predict global map with global observed map
    stride_m = 10 # TODO: move up to config, should have seperate prediction config     global_pred_propagate_onehot_tensor_rollouts = glocal.calculate_rollouts_global_map_prediction(models_dict, device, global_obs_onehot_tensor, map_configs, multiplier, stride_m, known_threshold, show_viz=False)
    known_threshold = 0.1 

    global_pred_propagate_onehot_tensor_rollouts = glocal.calculate_rollouts_global_map_prediction(models_dict, device, global_obs_onehot_tensor, map_configs, multiplier, stride_m, known_threshold, show_viz=False)
    print("global_pred_propagate_onehot_tensor_rollouts.shape: ", global_pred_propagate_onehot_tensor_rollouts.shape)
    global_pred_propagate_onehot_tensor_mean = torch.mean(global_pred_propagate_onehot_tensor_rollouts, dim=0).unsqueeze(0)

    # Get the variance 
    var_class = 1
    variance_global_pred_propagate_onehot_tensor = torch.var(global_pred_propagate_onehot_tensor_rollouts[:,var_class,:,:], dim=0)

    
    # Visualize 
    plt.figure(figsize=(20,12), facecolor=(1,1,1))
    plt.subplot(1,3,1)
    colorized_obs_global = vutils.get_colorized_map(global_obs_onehot_tensor)
    plt.imshow(colorized_obs_global.permute(1,2,0).squeeze().detach().cpu().numpy())
    plt.title("No Map Prediction")

    plt.subplot(1,3,2)
    colorized_pred_global = vutils.get_colorized_map(global_pred_propagate_onehot_tensor_mean)
    plt.imshow(colorized_pred_global.permute(1,2,0).squeeze().detach().cpu().numpy())
    plt.title("Propagated Map Prediction")

    plt.subplot(1,3,3)
    plt.imshow(variance_global_pred_propagate_onehot_tensor.detach().cpu().numpy())
    plt.title("Variance of Map Prediction")
    plt.show()

