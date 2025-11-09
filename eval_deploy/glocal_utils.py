
# third party import 
from torchvision.transforms import ToTensor, Resize, Compose
import PIL
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np 
from scipy.ndimage import distance_transform_cdt
from tqdm import tqdm
import time 
import torch.nn.functional as F

# custom import 
import sys 
sys.path.append('../')
from eval_deploy import deploy_utils as dutils
from eval_deploy import viz_utils as vutils


def get_rollouts_local_pred_and_gt(global_obs_onehot_tensor, global_gt_onehot_tensor, max_propagation_layer, map_configs, multiplier, stride_m, known_threshold, show_viz=False):
    """
    Get local prediction and ground truth patch for each rollout.
    """
    patch_grid_dt, patch_grid_inds, patch_centers = get_patch_grid_dt(global_obs_onehot_tensor, map_configs, multiplier, stride_m, known_threshold, random_shift=True)
    
    # print("patch_grid_dt.shape: {}".format(patch_grid_dt.shape))
    # plt.subplot(1,3,1)
    # plt.imshow(global_gt_onehot_tensor.squeeze().permute(1,2,0).detach().cpu().numpy())
    # plt.subplot(1,3,2)
    # plt.imshow(global_obs_onehot_tensor.squeeze().permute(1,2,0).detach().cpu().numpy())
    # plt.subplot(1,3,3)
    # plt.imshow(patch_grid_dt)
    # plt.show()
    local_obs_maputils_list, local_gt_maputils_list = get_local_gt_and_pred(global_obs_onehot_tensor, global_gt_onehot_tensor, max_propagation_layer, patch_grid_dt, patch_grid_inds, patch_centers, multiplier, map_configs)
    return local_obs_maputils_list, local_gt_maputils_list

def calculate_rollouts_global_map_prediction(models_dict, device, global_obs_onehot_tensor, max_propagation_layer, map_configs, multiplier, stride_m, known_threshold, show_viz=False):
    """ 
    Calculate multiple global map prediction by rolling out local map prediction, layer by layer. 

    Args:
        models_dict: dictionary of models
        global_obs_onehot_tensor:  current observed global map (1 x C x H x W) tensor
        map_configs: dictionary of map configurations
        multiplier: multiplier for local map size
        stride_m: stride between patches in meters
        known_threshold: threshold for known space to see if a patch is known or not as a 0th layer of onion
        show_viz: whether to show visualization or not
        max_propagation_layer: maximum number of layers to propagate outwards from known space

    Returns 
        global_pred_onehot_tensor_list: (num_rollouts x C x H x W) tensor of global map prediction. Stacked along 0th dimension for each rollout
    """
    # Get patch distance transform from known space
    show_patch_viz = False
    
    patch_grid_dt, patch_grid_inds, patch_centers = get_patch_grid_dt(global_obs_onehot_tensor, map_configs, multiplier, stride_m, known_threshold, show_viz=show_patch_viz)
    # TODO: experiment with different ways to vary model being used. 
    # Currently rolling out model by model. Resulting in (number of models) maps
    global_pred_onehot_tensor_rollouts = []
    for model_num_to_use in range(len(models_dict.keys())):

        global_pred_onehot_tensor = calculate_global_pred_with_local_map_propagation(global_obs_onehot_tensor, max_propagation_layer, patch_grid_dt, patch_grid_inds, patch_centers, multiplier, map_configs, device, models_dict, 
                                                                                    model_num_to_use=model_num_to_use, show_viz=show_viz)
        global_pred_onehot_tensor_rollouts.append(global_pred_onehot_tensor)

 


    # Make list into tensor
    global_pred_onehot_tensor_rollouts = torch.vstack(global_pred_onehot_tensor_rollouts) # (num_rollouts x C x H x W)
    # print("global_pred_onehot_tensor_list.shape: {}".format(global_pred_onehot_tensor_rollouts.shape))
    return global_pred_onehot_tensor_rollouts

def get_patch_grid_dt(global_obs_map_onehot, map_configs, multiplier, stride_m, known_threshold, random_shift=False, show_viz=False):
    # print("Getting patch grid distance transform")
    # Input: global_obs_onehot_tensor
    local_map_size_multiplied = map_configs['local_map_size'] * multiplier
    patch_shape = (local_map_size_multiplied, local_map_size_multiplied)
    stride = (stride_m * multiplier, stride_m * multiplier)
    # print(global_obs_map_onehot.shape)
    assert global_obs_map_onehot.shape[2] > patch_shape[0], "Patch shape is larger than global map height"
    assert global_obs_map_onehot.shape[3] > patch_shape[1], "Patch shape is larger than global map width"
    
    patch_centers, patch_grid = dutils.get_patch_centers(global_obs_map_onehot.shape[2:], patch_shape, stride, random_shift=random_shift)
    local_patches = dutils.get_patches_given_centers(patch_centers, global_obs_map_onehot, map_configs["local_map_size"]*multiplier)
    # print("Number of patches: {}".format(patch_centers.shape[0]))   

    # to keep track of what has been predicted 
    patch_grid_inds_xv, patch_grid_inds_yv = np.meshgrid(range(patch_grid.shape[0]), range(patch_grid.shape[1]))
    patch_grid_inds = np.column_stack((patch_grid_inds_xv.ravel(), patch_grid_inds_yv.ravel()))


    # Get 0th layer of patches (known % is above threshold)
    num_patch_known = 0
    for patch_i in range(patch_centers.shape[0]):
        # Key variables for this patch
        patch_center = patch_centers[patch_i]
        local_patch = local_patches[patch_i].unsqueeze(0)
        patch_grid_ind = patch_grid_inds[patch_i] # to keep track of what has been predicted 

        # Get percentage of known pixels (n1 or 2)
        # # Get number of pixels in each class seperately
        num_pixels_in_classes = torch.sum(local_patch, dim=(2,3))[0]
        # print(num_pixels_in_classes)
        percentage_known = (num_pixels_in_classes[1] + num_pixels_in_classes[2]) / torch.sum(num_pixels_in_classes)

        if percentage_known > known_threshold:
            num_patch_known += 1
            patch_grid[patch_grid_ind[0], patch_grid_ind[1]] = 1
    # print("Number of patches known: {}, out of {} total patches".format(num_patch_known, patch_centers.shape[0]))
    # Get distance transform of patch_grid to get "onion layers"
    distance_transform_patch_grid = distance_transform_cdt(1-patch_grid, metric='chessboard') # TODO: check why it is 1-patch_grid
    
    show_patch_viz = False 
    if show_patch_viz:
        plt_row = 1
        plt_col = 3 
        # Overlay patch rectangles on global observed
        plt.subplot(plt_row, plt_col, 1)
        obs_colorized = vutils.get_colorized_map(global_obs_map_onehot)
        plt.imshow(obs_colorized.squeeze().permute(1,2,0).detach().cpu().numpy())
        for patch_i in tqdm(range(patch_centers.shape[0])):
            patch_grid_ind = patch_grid_inds[patch_i] # to keep track of what has been predicted 
            layer_associated = distance_transform_patch_grid[patch_grid_ind[0], patch_grid_ind[1]]
            if layer_associated <= 1: # 0 or 1st layer

                plt.subplot(plt_row, plt_col, 1)
                patch_grid_ind = patch_grid_inds[patch_i] # to keep track of what has been predicted 
                patch_center = patch_centers[patch_i]
                if layer_associated == 0:
                    color = 'r'
                    linewidth = 3
                else:
                    color = 'k'
                    linewidth = 1
                plt.scatter(patch_center[1], patch_center[0], c=color, marker='x', s=10) # patch centers
                rect = patches.Rectangle((patch_center[1] - (local_map_size_multiplied/2), patch_center[0] - (local_map_size_multiplied/2)), local_map_size_multiplied, local_map_size_multiplied, 
                                        linewidth=linewidth, edgecolor=color, facecolor='none',alpha=0.5)
                plt.gca().add_patch(rect)

        # Overlay last patch rectangle in a different color to show size 
        plt.subplot(plt_row, plt_col, 2)
        plt.imshow(obs_colorized.squeeze().permute(1,2,0).detach().cpu().numpy())
        plt.scatter(patch_center[1], patch_center[0], c='r', marker='x', s=20) # patch centers
        rect = patches.Rectangle((patch_center[1] - (local_map_size_multiplied/2), patch_center[0] - (local_map_size_multiplied/2)), local_map_size_multiplied, local_map_size_multiplied, 
                                linewidth=1, edgecolor='r', facecolor='none',alpha=0.5)
        plt.gca().add_patch(rect)

        # plt.show()
        print("saving patch grid figure")
        plt.savefig("patch_grid.png")
        plt.show()
    
    if show_viz:
        # Visualize (each layer)
        plt_row = 1 
        plt_col = 5

        for layer in range(np.max(distance_transform_patch_grid)):
            print("visualizing layer: {}".format(layer))
            plot_global_obs_with_patch_num = 5
            plot_global_obs = 1
            plot_patch_grid = 2
            plot_distance_transform = 3
            plot_current_layer = 4

            plt.figure(figsize=(20, 5))
            plt.subplot(plt_row, plt_col, plot_global_obs_with_patch_num)
            colorized_obs_global = vutils.get_colorized_map(global_obs_map_onehot)
            plt.imshow(colorized_obs_global.permute(1,2,0).squeeze().detach().cpu().numpy()) 

            plt.title("Current patches in layer {}".format(layer))

            plt.subplot(plt_row, plt_col, plot_global_obs)
            colorized_obs_global = vutils.get_colorized_map(global_obs_map_onehot)
            plt.imshow(colorized_obs_global.permute(1,2,0).squeeze().detach().cpu().numpy()) 
            plt.title("Global obs map")

            plt.subplot(plt_row, plt_col, plot_patch_grid)
            plt.imshow(patch_grid)
            plt.title("Patch grid \n (1: >{}% known)".format(round(known_threshold*100, 2)))

            for patch_i in range(patch_centers.shape[0]):
                plt.subplot(plt_row, plt_col, plot_global_obs_with_patch_num)
                patch_grid_ind = patch_grid_inds[patch_i] # to keep track of what has been predicted 
                patch_center = patch_centers[patch_i]
                if distance_transform_patch_grid[patch_grid_ind[0], patch_grid_ind[1]] == layer:
                    plt.scatter(patch_center[1], patch_center[0], c='r', marker='x', s=100) # patch centers
                    rect = patches.Rectangle((patch_center[1] - (local_map_size_multiplied/2), patch_center[0] - (local_map_size_multiplied/2)), local_map_size_multiplied, local_map_size_multiplied, 
                                            linewidth=1, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rect)
                else: 
                    plt.scatter(patch_center[1], patch_center[0], c='k', marker='x', s=100) # patch centers




            plt.subplot(plt_row, plt_col, plot_distance_transform)
            plt.imshow(distance_transform_patch_grid)
            plt.title("Distance transform \n of patch grid")
            plt.colorbar()

            plt.subplot(plt_row, plt_col, plot_current_layer)
            plt.imshow(distance_transform_patch_grid == layer)
            plt.title("Current patch layer")
            plt.show()
    # import pdb; pdb.set_trace()

    return distance_transform_patch_grid, patch_grid_inds, patch_centers

def calculate_global_pred_with_local_map_propagation(global_obs_onehot_tensor, max_propagation_layers, patch_grid_dt, patch_grid_inds, patch_centers, multiplier, map_configs, device, models_dict, model_num_to_use, show_viz=False):
    start_func_time = time.time()
    cum_patch_pred_time = 0
    cum_patch_pred_count = 0
    cum_clone_time = 0
    cum_update_time = 0
    
    # Start with prediction as global obs map 
    clone_start_time = time.time()
    global_pred_multi_onehot_tensor = global_obs_onehot_tensor.clone()
    cum_clone_time += time.time() - clone_start_time
    local_map_size_multiplied = map_configs['local_map_size'] * multiplier
    # For a given layer 
    for layer_ind in range(np.min([np.max(patch_grid_dt) + 1, max_propagation_layers])):
    # for layer_ind in range(1):
        before_layer_update_global_pred = global_pred_multi_onehot_tensor.clone()
        # get the patch centers associated with the layer
        patch_centers_for_layer = [] # TODO: vectorize
        for patch_i in range(patch_centers.shape[0]):
            patch_center = patch_centers[patch_i]
            patch_grid_ind = patch_grid_inds[patch_i]
            if patch_grid_dt[patch_grid_ind[0], patch_grid_ind[1]] == layer_ind:
                patch_centers_for_layer.append(patch_center)
        patch_centers_for_layer = np.array(patch_centers_for_layer)
        # Collect patches in a batch
        clone_start_time = time.time()
        before_layer_update_global_pred = global_pred_multi_onehot_tensor.clone()
        cum_clone_time += time.time() - clone_start_time
        local_patches = dutils.get_patches_given_centers(patch_centers_for_layer, before_layer_update_global_pred, local_map_size_multiplied)
        local_patches_resized = Resize((64, 64), interpolation=PIL.Image.NEAREST)(local_patches)  # Resize patch #! TODO: Parametrize
        # loop through each patch, predict then update global map 

        # Predict for the entire batch
        pred_start_time = time.time()
        batch_pred_onehot = dutils.run_map_predictor_single_model(local_patches_resized, device, models_dict, model_num_to_use)
        cum_patch_pred_time += time.time() - pred_start_time

        # Iterate through batch predictions to update the global map
        for i, local_pred_onehot in enumerate(batch_pred_onehot):
            patch_center = patch_centers_for_layer[i]
            global_pred_multi_onehot_tensor = dutils.update_global_map_with_local(global_pred_multi_onehot_tensor, local_pred_onehot.unsqueeze(0), patch_center, local_map_size=map_configs['local_map_size']*multiplier)

        # for patch_i in range(len(patch_centers_for_layer)):
        #     patch_center = patch_centers_for_layer[patch_i]
        #     patch_center_expanded = np.expand_dims(patch_center,axis=0)
        #     # Get patch from current predicted map
        #     local_patch = dutils.get_patches_given_centers(patch_center_expanded, before_layer_update_global_pred, map_configs["local_map_size"]*multiplier)
        #     # Resize 
        #     local_patch_resized = Resize((64, 64), interpolation=PIL.Image.NEAREST)(local_patch) # TODO: add 256 to configs

        #     # Predict
        #     pred_start_time = time.time()
        #     local_pred_onehot = dutils.run_map_predictor_single_model(local_patch_resized, device, models_dict, model_num_to_use)

        #     cum_patch_pred_time += time.time() - pred_start_time
        #     cum_patch_pred_count += 1
        #     # Update global predicted map with local prediction 
        #     clone_start_time = time.time()
        #     if show_viz:
        #         prev_global_pred_multi_onehot_tensor = global_pred_multi_onehot_tensor.clone() #for visualization
        #     cum_clone_time += time.time() - clone_start_time
            
        #     update_start_time = time.time()
        #     global_pred_multi_onehot_tensor = dutils.update_global_map_with_local(global_pred_multi_onehot_tensor, local_pred_onehot, patch_center, local_map_size=map_configs['local_map_size']*multiplier)
        #     cum_update_time += time.time() - update_start_time
        #     # plt.imshow(local_patch_resized[0].permute(1,2,0).detach().cpu().numpy())
        #     # plt.show()

    # print("GOT SOMETHING")
    # print("Cumulative patch prediction time: {}".format(cum_patch_pred_time))
    # print("Cumulative clone time: {}".format(cum_clone_time))
    # print("Cumulative update time: {}".format(cum_update_time))
    # print("! Total time for glocal propagation: {}".format(time.time() - start_func_time))
    # print("# of patches predicted: {}".format(cum_patch_pred_count))
    # print("Average patch prediction time: {}".format(cum_patch_pred_time/cum_patch_pred_count))
    return global_pred_multi_onehot_tensor


def get_local_gt_and_pred(global_obs_onehot_tensor, global_gt_one_hot_tensor, max_propagation_layers, patch_grid_dt, patch_grid_inds, patch_centers, multiplier, map_configs):
    local_obs_maputils_list = []
    local_gt_maputils_list = []

    # print("patch_grid_dt.shape: {}".format(patch_grid_dt.shape))
    layer_ind_list = range(np.min([np.max(patch_grid_dt) + 1, max_propagation_layers]))
    # For a given layer 
    for layer_ind in layer_ind_list:
        # import pdb; pdb.set_trace()

        # get the patch centers associated with the layer
        patch_centers_for_layer = [] # TODO: vectorize
        for patch_i in range(patch_centers.shape[0]):
            patch_center = patch_centers[patch_i]
            patch_grid_ind = patch_grid_inds[patch_i]
            if patch_grid_dt[patch_grid_ind[0], patch_grid_ind[1]] == layer_ind:
                patch_centers_for_layer.append(patch_center)
        patch_centers_for_layer = np.array(patch_centers_for_layer)

        # loop through each patch, predict then update global map 
        for patch_i in range(len(patch_centers_for_layer)):
            patch_center = patch_centers_for_layer[patch_i]
            patch_center_expanded = np.expand_dims(patch_center,axis=0)
            # Get patch from current predicted map
            local_obs_patch = dutils.get_patches_given_centers(patch_center_expanded, global_obs_onehot_tensor, map_configs["local_map_size"]*multiplier)
            # Resize 
            local_obs_patch_resized = Resize((64, 64), interpolation=PIL.Image.NEAREST)(local_obs_patch) # TODO: add 256 to configs

            # Get patch from current GT map 
            local_gt_patch = dutils.get_patches_given_centers(patch_center_expanded, global_gt_one_hot_tensor, map_configs["local_map_size"]*multiplier)
            # Resize
            local_gt_patch_resized = Resize((64, 64), interpolation=PIL.Image.NEAREST)(local_gt_patch) # TODO: add 256 to configs

            local_obs_maputils_list.append(dutils.convert_model_input_format_to_maputils_labelmaps(local_obs_patch_resized))
            local_gt_maputils_list.append(dutils.convert_model_input_format_to_maputils_labelmaps(local_gt_patch_resized))
            # print(local_obs_maputils.shape)
            # plt.subplot(1,2,1)
            # plt.imshow(local_obs_maputils)
            # plt.colorbar()
            # plt.subplot(1,2,2)
            # plt.imshow(local_gt_maputils)
            # plt.show()
    return local_obs_maputils_list, local_gt_maputils_list


def calculate_global_pred_with_glocal_inference(global_obs_map_onehot, global_pred_onehot_tensor, map_configs, multiplier, stride_m, models_dict, device, show_viz=False):
    """
    Calculate the global prediction using glocal inference (multiple local patches)

    Args:
        global_obs_map_onehot (tensor): (1,num_class,height,width). Most recently (1,3,500,600)
        global_pred_map_onehot (tensor): (1,num_class,height,width)
        map_configs (dict): map configurations
        multiplier (int): multiplier for global map size (meter to pixels)
        stride_m (int): how far apart the local patch queries are in meters
        models_dict (dict): dictionary of models
        device (str): device to run the model on
        show_viz (bool): whether to show visualization

    Output:
        global_pred_onehot_tensor (tensor): (1,num_class,height,width)
    """
    # Initialize global prediction map with current one 
    # TODO: right now we are doing direct replacement, but we want pred as input to do bayesian fusing later
    global_pred_onehot_tensor = global_pred_onehot_tensor.clone()

    # Get patch centers
    local_map_size_multiplied = map_configs['local_map_size'] * multiplier
    patch_shape = [local_map_size_multiplied, local_map_size_multiplied]
    stride = [stride_m * multiplier, stride_m * multiplier]
    patch_centers, _ = dutils.get_patch_centers(global_obs_map_onehot.shape[2:], patch_shape, stride)

    # Get local patches from global observed map
    local_patches = dutils.get_patches_given_centers(patch_centers, global_obs_map_onehot, map_configs["local_map_size"]*multiplier)

    # Loop through each patch: resize, predict, then update global prediction map
    for i in range(local_patches.shape[0]):
        local_patch = local_patches[i].unsqueeze(0)
        patch_center = patch_centers[i]

        # Resize local patch to model input 
        local_patch_resized = Resize((256, 256), interpolation=PIL.Image.NEAREST)(local_patch) # TODO: add 256 to configs

        # print("cuda mem before local pred: ", torch.cuda.memory_allocated() / 1e6)
        # Predict given local patch #TODO: ideally can parallelize with batch 
        local_pred_mean_onehot, _ = dutils.run_map_predictor(local_patch_resized, device, models_dict)
        
        # print("cuda mem after local pred: ", torch.cuda.memory_allocated() / 1e6)
        # Update global predicted map with local prediction 
        global_pred_onehot_tensor = dutils.update_global_map_with_local(global_pred_onehot_tensor, local_pred_mean_onehot, patch_center, local_map_size=map_configs['local_map_size']*multiplier)

        # print("cuda mem after global pred: ", torch.cuda.memory_allocated() / 1e6)
        # Optional: Show viz
        if show_viz:
            plt_row = 1
            plt_col = 4
            plt.figure(figsize=(10, 2))

            # (Plot 1) visualize global map and patch rectangle
            plt.subplot(plt_row, plt_col, 1)
            colorized_obs_global = vutils.get_colorized_map(global_obs_map_onehot)
            plt.imshow(colorized_obs_global.permute(1,2,0).squeeze().detach().cpu().numpy()) 
            plt.scatter(patch_center[1], patch_center[0], c='b', marker='x', s=100) # patch centers
            rect = patches.Rectangle((patch_center[1] - (local_map_size_multiplied/2), patch_center[0] - (local_map_size_multiplied/2)), local_map_size_multiplied, local_map_size_multiplied, 
                                    linewidth=3, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

            # (Plot 2) visualize local patch that we put into the model
            plt.subplot(plt_row, plt_col, 2)
            colorized_obs_local = vutils.get_colorized_map(local_patch)
            plt.imshow(colorized_obs_local.permute(1,2,0).squeeze().detach().cpu().numpy())
            plt.title("Local patch" + str(patch_center))

            # (Plot 3) visualize local prediction
            plt.subplot(plt_row, plt_col, 3)
            colorized_pred_local = vutils.get_colorized_map(local_pred_mean_onehot)
            plt.imshow(colorized_pred_local.permute(1,2,0).squeeze().detach().cpu().numpy())
            plt.title("Predicted local map")

            # (Plot 4) visualize global prediction
            plt.subplot(plt_row, plt_col, 4)

            colorized_pred_global = vutils.get_colorized_map(global_pred_onehot_tensor)
            plt.imshow(colorized_pred_global.permute(1,2,0).squeeze().detach().cpu().numpy())
            plt.title("Predicted global map")


    return global_pred_onehot_tensor
    # TODO: show viz