# Adapted from UPEN

import numpy as np
import torch
import os
from sklearn.metrics import confusion_matrix
from torchvision.transforms import ToTensor, Resize, Compose
from matplotlib import pyplot as plt
import cv2 
import PIL
import time 
# Custom imports
import sys 
sys.path.append('../')
from eval_deploy import viz_utils as vutils


def get_latest_model(save_dir):
    checkpoint_list = []
    for dirpath, _, filenames in os.walk(save_dir):
        for filename in filenames:
            if filename.endswith('.pt'):
                checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
    checkpoint_list = sorted(checkpoint_list)
    latest_checkpoint =  None if (len(checkpoint_list) is 0) else checkpoint_list[-1]
    return latest_checkpoint

def load_model(models, checkpoint_file):
    # Load the latest checkpoint
    checkpoint = torch.load(checkpoint_file)
    new_state_dict = {'module.' + k: v for k, v in checkpoint['model_state_dict'].items()}
    models['predictor_model'].load_state_dict(new_state_dict, strict=False)
    return models

def run_map_predictor_single_model(step_ego_grid_crops, device, models_dict, model_num_to_use):
    with torch.no_grad():
        input_batch = {'step_ego_grid_crops_spatial': step_ego_grid_crops}
        input_batch = {k: v.to(device) for k, v in input_batch.items()}
        model_pred = models_dict[model_num_to_use]['predictor_model'](input_batch)
        
    return model_pred['pred_maps_spatial']


def get_freq_iou(iou_metric, pred_onehot_tensor, target_onehot_tensor):
    """
    iou_metric: function that takes in two tensors and returns a scalar,
    make sure reduction='none'
    JaccardIndex(task="multiclass", num_classes=3, reduction="none")
    # TODO add documentation on expected size
    """
    iou_score_per_class = get_iou_per_class(iou_metric, pred_onehot_tensor, target_onehot_tensor)
    # calculate number of pixels per class in ground truth
    pixel_count_per_class = [torch.sum(target_onehot_tensor[:,i,:,:]).item()for i in range(target_onehot_tensor.shape[1])]
    freq_weighted_iou = sum([iou_score_per_class[i] * pixel_count_per_class[i] for i in range(len(iou_score_per_class))])/sum(pixel_count_per_class)

    return freq_weighted_iou


def get_mean_iou(iou_metric, pred_onehot_tensor, target_onehot_tensor, mode='both', obs_onehot_tensor=None):
    """
    iou_metric: function that takes in two tensors and returns a scalar,
    make sure reduction='none'
    JaccardIndex(task="multiclass", num_classes=3, reduction="none")
    # TODO add documentation on expected size
    """

    iou_score_per_class = get_iou_per_class(iou_metric, pred_onehot_tensor, target_onehot_tensor, mode, obs_onehot_tensor)
    mean_iou = torch.mean(iou_score_per_class)

    return mean_iou

def get_mean_iou_allmodes(iou_metric, pred_onehot_tensor, target_onehot_tensor, obs_onehot_tensor):
    mean_iou_both = get_mean_iou(iou_metric, pred_onehot_tensor, target_onehot_tensor, mode='both', obs_onehot_tensor=obs_onehot_tensor)
    mean_iou_known = get_mean_iou(iou_metric, pred_onehot_tensor, target_onehot_tensor, mode='known', obs_onehot_tensor=obs_onehot_tensor)
    mean_iou_occluded = get_mean_iou(iou_metric, pred_onehot_tensor, target_onehot_tensor, mode='occluded', obs_onehot_tensor=obs_onehot_tensor)

    return mean_iou_both, mean_iou_known, mean_iou_occluded


def get_known_stats(iou_metric, obs_onehot_tensor, pred_onehot_tensor, target_onehot_tensor):
    """
    iou_metric: function that takes in two tensors and returns a scalar,
    make sure reduction='none'
    don't use torchmetrics, since later we want to implement fuzzy 

    obs_onehot_tensor: (B x Classes x H x W)
    pred_onehot_tensor: (B x Classes x H x W)
    target_onehot_tensor: (B x Classes x H x W)
    """


    obs_onehot_tensor_argmax = torch.argmax(obs_onehot_tensor, dim=1)
    pred_tensor_argmax = torch.argmax(pred_onehot_tensor, dim=1)
    target_tensor_argmax = torch.argmax(target_onehot_tensor, dim=1)
    assert pred_tensor_argmax.shape == target_tensor_argmax.shape, "pred and target must have same shape"
    assert pred_tensor_argmax.shape[0] == 1, "batch size must be 1"  # TODO: need to handle more than one image in batch 

    # For pixels where obs is not 0 (void), what is the IOU

    # Get the pixels where obs is not 0
    obs_not_void_mask = obs_onehot_tensor_argmax != 0
    obs_not_void_indices = obs_not_void_mask.nonzero()
    # get mask into index 
    
    # print("obs_not_void_indices: ", obs_not_void_indices.shape)
    # print("pred_onehot_tensor.shape: ", pred_onehot_tensor.shape)   
    # Get the IOU for those pixels
    obs_not_void_iou_per_class = get_iou_per_class(iou_metric, pred_onehot_tensor[obs_not_void_indices[:,0],:,obs_not_void_indices[:,1], obs_not_void_indices[:,2]],
                                                    target_onehot_tensor[obs_not_void_indices[:,0],:,obs_not_void_indices[:,1], obs_not_void_indices[:,2]])

    print("obs_not_void_iou_per_class: ", obs_not_void_iou_per_class)
    mean_iou = torch.mean(obs_not_void_iou_per_class)
    return mean_iou

def get_mean_iou_cherie_impl(num_class, pred_onehot_tensor, target_onehot_tensor):
    """
    iou_metric: function that takes in two tensors and returns a scalar,
    make sure reduction='none'
    don't use torchmetrics, since later we want to implement fuzzy 

    pred_onehot_tensor: (B x Classes x H x W)
    target_onehot_tensor: (B x Classes x H x W)
    """

    # Get IOU per class (argmax)
    iou_list = []
    pred_tensor_argmax = torch.argmax(pred_onehot_tensor, dim=1)
    target_tensor_argmax = torch.argmax(target_onehot_tensor, dim=1)
    assert pred_tensor_argmax.shape == target_tensor_argmax.shape, "pred and target must have same shape"
    assert pred_tensor_argmax.shape[0] == 1, "batch size must be 1"  # TODO: need to handle more than one image in batch 
    for class_idx in range(num_class):

        # Get the intersection (with some spatial threshold)
        
        intersection_class_i = 0
        for i in range(pred_tensor_argmax.shape[1]):
            for j in range(pred_tensor_argmax.shape[2]):
                # print(target_tensor_argmax.shape)
                if target_tensor_argmax[0, i,j] == class_idx:# TODO: need to handle more than one image in batch 
                    # Get a patch around the i,j pixel in pred_tensor_argmax
                    neighborhood_radius = 30
                    patch_min_i = max(0, i-neighborhood_radius)
                    patch_max_i = min(pred_tensor_argmax.shape[1], i+neighborhood_radius)
                    patch_min_j = max(0, j-neighborhood_radius)
                    patch_max_j = min(pred_tensor_argmax.shape[2], j+neighborhood_radius)

                    pred_patch = pred_tensor_argmax[:,patch_min_i:patch_max_i, patch_min_j:patch_max_j]# this is the patch for comparison to a GT pixel to give some leeway
                    # print(pred_patch.shape)
                    # print(patch_min_i, patch_max_i, patch_min_j, patch_max_j)
                    # If the GT pixel is in the patch, then it is considered a match
                    if target_tensor_argmax[0, i,j] in pred_patch: # TODO: need to handle more than one image in batch 
                        intersection_class_i += 1
        # intersection = torch.sum((pred_tensor_argmax == class_idx) & (target_tensor_argmax == class_idx))
        # Get the union (TODO: think mathematically if there should be fuzzy room here too)
        union = torch.sum((pred_tensor_argmax == class_idx) | (target_tensor_argmax == class_idx))
        # Get the IOU
        iou = float(intersection_class_i) / union.float()
        iou_list.append(iou)
    print("iou_list (cherie)", iou_list)

    # Get the mean IOU
    mean_iou = torch.mean(torch.stack(iou_list))
    return mean_iou

def get_iou_per_class(iou_metric, pred_onehot_tensor, target_onehot_tensor, mode='both', obs_onehot_tensor=None):
    """
    iou_metric: function that takes in two tensors and returns a scalar,
    make sure reduction='none'
    JaccardIndex(task="multiclass", num_classes=3, reduction="none")
    Mode: Regions where metric is calculated. Both: occluded and known, Occluded: occluded, Known: known
    # TODO add documentation on expected size
    """
    target_tensor_argmax = torch.argmax(target_onehot_tensor, dim=1) # TODO: maybe move outside since it's repeated for each map

    # Compute metrics (IOU, currently observed)
    # For anywhere not currently observed, set target to 0 (unknown, ignore_index), so that it doesn't count towards the IOU
    if mode is "known":
        assert obs_onehot_tensor is not None, "obs_onehot_tensor must be provided if mode is known"
        unk_class = 0
        unk_mask = obs_onehot_tensor[:,unk_class,:,:]==1 # 1 if unknown, 0 if observed
        target_tensor_argmax[unk_mask] = 0 # set unknown to 0 so that it doesn't count towards the IOU
    
    # Compute metrics (IOU, currently occluded, but is not unknown in ground truth)
    # For anywhere that is currently observed, set target to 0 (unknown, ignore_index), so that it doesn't count towards the IOU
    elif mode is "occluded":
        assert obs_onehot_tensor is not None, "obs_onehot_tensor must be provided if mode is occluded"
        unk_class = 0
        known_mask = obs_onehot_tensor[:,unk_class,:,:]==0 # 1 if observed, 0 if unknown
        target_tensor_argmax[known_mask] = 0 # set known to 0 so that it doesn't count towards the IOU

    iou_score_per_class = iou_metric(pred_onehot_tensor, target_tensor_argmax)
    return iou_score_per_class

def change_onehot_unobserved_to_free(onehot_tensor):
    """
    Change all instances of unobserved class to free class.
    Change from class 0 to class 2

    Args:
        onehot_tensor (torch.tensor): (B x C x H x W)
    
    Returns:
        onehot_tensor (torch.tensor): (B x C x H x W)
    """
    onehot_tensor = onehot_tensor.clone()
    unobserved_class_idx = 0
    free_class_idx = 2

    onehot_tensor[:,free_class_idx,:,:] = onehot_tensor[:,unobserved_class_idx,:,:] + onehot_tensor[:,free_class_idx,:,:]

    onehot_tensor[:, unobserved_class_idx, :, :] = 0

    return onehot_tensor

def change_onehot_unobserved_to_nearest(onehot_tensor):
    """
    Change all instances of unobserved class to value given nearest known.

    Args:
        onehot_tensor (torch.tensor): (B x C x H x W)
    
    Returns:
        onehot_tensor (torch.tensor): (B x C x H x W)
    """
    assert onehot_tensor.shape[0] == 1, "assumes batch size is 1 for now!"
    onehot_tensor = onehot_tensor.clone()
    # Convert tensor to rgb channel to use cv2 inpaint functions
    onehot_3chan = onehot_tensor[0].permute(1,2,0) #! TODO assumes batch size is 1 for now
    unobserved_class_idx = 0
    unknown_mask = (onehot_3chan[:,:,unobserved_class_idx].numpy()==1).astype(np.uint8)
    inpainted = cv2.inpaint((onehot_3chan.numpy()*255).astype(np.uint8), unknown_mask, 3, cv2.INPAINT_NS)
    inpainted_argmax = np.argmax(inpainted, axis=2)
    inpainted_onehot = np.empty((inpainted.shape[0], inpainted.shape[1], 3))
    inpainted_onehot[:,:,0] = inpainted_argmax==0
    inpainted_onehot[:,:,1] = inpainted_argmax==1
    inpainted_onehot[:,:,2] = inpainted_argmax==2

    inpainted_onehot_tensor = torch.from_numpy(inpainted_onehot).permute(2,0,1).unsqueeze(0)
    
    # plt.figure(figsize=(10,10))
    # plt.subplot(1,3,1)
    # plt.imshow(onehot_3chan)
    # plt.subplot(1,3,2)
    # plt.imshow(inpainted)
    # plt.subplot(1,3,3)
    # plt.imshow(inpainted_onehot)
    # plt.show()
    return inpainted_onehot_tensor

def change_onehot_unobserved_to_occupied(onehot_tensor):
    """
    Change all instances of unobserved class to occupied class.
    Change from class 0 to class 1

    Args:
        onehot_tensor (torch.tensor): (B x C x H x W)
    
    Returns:
        onehot_tensor (torch.tensor): (B x C x H x W)
    """
    onehot_tensor = onehot_tensor.clone()
    unobserved_class_idx = 0
    occupied_class_idx = 1

    onehot_tensor[:,occupied_class_idx,:,:] = onehot_tensor[:,unobserved_class_idx,:,:] + onehot_tensor[:,occupied_class_idx,:,:]

    onehot_tensor[:, unobserved_class_idx, :, :] = 0

    return onehot_tensor

def get_ssim(ssim_metric, pred_onehot_tensor, target_onehot_tensor):
    """
    # TODO: add documentation 
    """
    # TODO: maybe move outside since it's repeated for each map
    target_tensor_argmax = torch.argmax(target_onehot_tensor, dim=1).float()
    pred_tensor_argmax = torch.argmax(pred_onehot_tensor, dim=1).float()

    # Add a class dimension to the tensors
    target_tensor_argmax = target_tensor_argmax.unsqueeze(1)
    pred_tensor_argmax = pred_tensor_argmax.unsqueeze(1)

    # Calculate SSIM metrics 
    ssim_score, ssim_image = ssim_metric(pred_tensor_argmax, target_tensor_argmax)

    # plt.subplot(1,3,1)
    # plt.imshow(pred_tensor_argmax[0,0,:,:])
    # plt.colorbar()
    # plt.title("pred_tensor_argmax")
    # plt.subplot(1,3,2)
    # plt.imshow(target_tensor_argmax[0,0,:,:])
    # plt.colorbar()
    # plt.title("target_tensor_argmax")

    # plt.subplot(1,3,3)
    # plt.imshow(ssim_image[0,0,:,:])
    # plt.title("ssim_image")
    # plt.colorbar()

    # plt.show()
    return ssim_score


def convert_maputils_labelmaps_to_model_input_format(labelmaps, transform):
    # Convert the labelmaps from maputils to the format expected by the model
    # labelmaps: grid_dim x grid_dim. values are (0, 0.5, 1)
    # model_input: 1 x num_class x resize_shape x resize_shape (in map_predict ordering)

    unique_colors = (0.5, 1, 0)# (void, occupied, free)
    # print("labelmaps.shape: ", labelmaps.shape)
    one_hot = np.zeros((labelmaps.shape[0], labelmaps.shape[1], len(unique_colors)), dtype=np.uint8)
    for i, color in enumerate(unique_colors):
        mask = (labelmaps == color) # slightly different since it is 1 cchannel
        one_hot[:, :, i][mask] = 1
    # # Read as PIL then transform to tensor, and add a batch dimension as the first dimension
    one_hot = (one_hot * 255).astype(np.uint8)
    pil_image = PIL.Image.fromarray(one_hot)
    input_image_transformed = transform(pil_image)
    input_image_transformed = input_image_transformed.unsqueeze(0) # (1,3,256,256) apply batch dimension

    return input_image_transformed


def convert_model_input_format_to_maputils_labelmaps(model_input):
    # Convert the model input to the format expected by maputils
    # model_input: 1 x num_class x resize_shape x resize_shape (in map_predict ordering)
    # labelmaps: grid_dim x grid_dim. values are (0, 0.5, 1)
    # print("model input", model_input.shape)
    # Remove the batch dimension
    model_input = model_input.squeeze(0)
    # Convert to numpy
    model_input = model_input.cpu().numpy()

    # Convert to (0.5, 1, 0) from (0, 1, 2)
    model_input = np.argmax(model_input, axis=0)
    model_input = model_input.astype(np.float32)
    model_input_copy = model_input.copy()
    model_input[model_input_copy == 0] = 0.5
    model_input[model_input_copy == 1] = 0
    model_input[model_input_copy == 2] = 1

    return model_input

def make_map_larger(map, multiplier):
    """
    Make map larger in both axis (i,j) by repeating each element along each axis

    Arg:
        map (np.array): (height, width)
        multiplier (int): multiplier for each axis
    """
    assert len(map.shape) == 2, "map must be 2D" # TODO: may be able to handle 3D too
    map_large = np.repeat(map, multiplier, axis=0)
    map_large = np.repeat(map_large, multiplier, axis=1)
    return map_large

def update_global_map_with_local(global_onehot_tensor, local_onehot_tensor, robot_pose, local_map_size):
    """
    Update/accumulate global map with local measurments which can be direct observations or predictions

    Args:
        global_onehot_tensor (tensor): (1,3,50,60)
        local_onehot_tensor (tensor): (1,3,256,256)
        robot_pose (list): [x,y] in global map coordinates
    
    Output:
        global_onehot_tensor (tensor): (1,3,50,60)
    """
    # Resize the local map to the size of the crop in global map space (local_map_size)
    local_onehot_resized = Resize((local_map_size, local_map_size))(local_onehot_tensor)

    # print("local_onehot_resized.shape: ", local_onehot_resized.shape)
    # print("global_onehot_tensor.shape: ", global_onehot_tensor.shape)
    # # Convert robot pose to indices in global space
    x_idx = robot_pose[0] # TODO: may be flipped
    y_idx = robot_pose[1]

    # Calculate the indices of the local map in global space
    # # handles at the edge of map
    x_start_idx = int(max(0, x_idx - local_map_size//2, 0)) 
    x_skip_idx_start = int(x_start_idx - (x_idx - local_map_size//2)) 
    x_end_idx = int(min(x_idx + local_map_size//2, global_onehot_tensor.shape[2]))
    x_skip_idx_end = int(local_map_size + (x_end_idx - (x_idx + local_map_size//2)) )
    y_start_idx = int(max(0, y_idx - local_map_size//2, 0))
    y_skip_idx_start = int(y_start_idx - (y_idx - local_map_size//2) )
    y_end_idx = int(min(y_idx + local_map_size//2, global_onehot_tensor.shape[3]))
    y_skip_idx_end = int(local_map_size + (y_end_idx - (y_idx + local_map_size//2)) )

    # print("x_start_idx, x_end_idx, y_start_idx, y_end_idx: ", x_start_idx, x_end_idx, y_start_idx, y_end_idx)
    # print("x_skip_idx_start, x_skip_idx_end, y_skip_idx_start, y_skip_idx_end: ", x_skip_idx_start, x_skip_idx_end, y_skip_idx_start, y_skip_idx_end)
    # print("global_onehot_tensor_cropped.shape: ", global_onehot_tensor[:, :, x_start_idx:x_end_idx, y_start_idx:y_end_idx].shape)
    
    # # Naive replacement for now, directly change those global indice with local map
    # print("Putting patches at ", x_start_idx, x_end_idx, y_start_idx, y_end_idx)
    local_onehot_cropped = local_onehot_resized[:,:, x_skip_idx_start:x_skip_idx_end, \
                               y_skip_idx_start:y_skip_idx_end]
    # plt_row = 1 
    # plt_col = 4
    # plt.figure(figsize=(15,15))
    # plt.subplot(plt_row,plt_col,1)
    # plt.imshow(local_onehot_tensor[0,1,:,:].cpu().numpy())
    # plt.title("local_onehot_tensor")
    # plt.subplot(plt_row,plt_col,2)
    # plt.imshow(local_onehot_resized[0,1,:,:].cpu().numpy())
    # plt.title("local_onehot_resized")
    # plt.subplot(plt_row,plt_col,3)
    # plt.imshow(local_onehot_cropped[0,1,:,:].cpu().numpy())
    # plt.title("local_onehot_cropped")
    # plt.show()
    # print("local_onehot_cropped.shape: ", local_onehot_cropped.shape)
    global_onehot_tensor[:, :, x_start_idx:x_end_idx, y_start_idx:y_end_idx] = local_onehot_cropped
    return global_onehot_tensor



def get_patch_centers(map_shape, patch_shape, stride, random_shift=False):
# # TODO: somehow find if there are areas that is not covered by any patch
    # print("map_shape: ", map_shape)
    map_height = map_shape[0]
    map_width = map_shape[1]
    patch_height = patch_shape[0]
    patch_width = patch_shape[1]
    stride_x = stride[0]
    stride_y = stride[1]

    patch_shift = np.array([0, 0])
    if random_shift:
        patch_shift = np.random.randint(0, stride, size=2)
    x_coords = np.arange(patch_height // 2 + patch_shift[0], map_height - patch_height // 2 + 1, stride_x)
    y_coords = np.arange(patch_width // 2 + patch_shift[1], map_width - patch_width // 2 + 1, stride_y)

    xv, yv = np.meshgrid(x_coords, y_coords)
    patch_centers = np.column_stack((xv.ravel(), yv.ravel()))

    patch_grid = np.zeros((x_coords.shape[0], y_coords.shape[0]))
    return patch_centers, patch_grid

def get_patches_given_centers(patch_centers, global_onehot, local_map_size_multipled):
    """
    Get batches of patches from global map given the centers of the patches

    Args:
        patch_centers (np.array): (num_patches, 2)
        global_onehot (tensor): (1,num_class,height,width). Most recently (1,3,500,600)
        local_map_size_multipled (int): size of the local map in global space

    Output:
        patches (tensor): (num_patches, num_class, local_map_size_multipled, local_map_size_multipled)
    """
    # Get the patch size
    patch_size = local_map_size_multipled

    # Get the number of patches
    num_patches = patch_centers.shape[0]
    
    # Get number of classes
    num_class = global_onehot.shape[1]

    # Initialize the patches
    patches = torch.zeros((num_patches, num_class, patch_size, patch_size))

    # # Get the indices of the patches
    # patch_indices = torch.zeros((num_patches, 2))
    # for i in range(num_patches):
    #     patch_indices[i, 0] = patch_centers[i, 0] - patch_size // 2
    #     patch_indices[i, 1] = patch_centers[i, 1] - patch_size // 2
    # print("patch_centers.shape: ", patch_centers.shape)
    # Get the patches
    for i in range(num_patches):

        patch_start_i = int(patch_centers[i, 0] - patch_size // 2)
        patch_end_i = int(patch_centers[i, 0] + patch_size // 2)
        patch_start_j = int(patch_centers[i, 1] - patch_size // 2)
        patch_end_j = int(patch_centers[i, 1] + patch_size // 2)
        # print("Getting patches at: patch_start_i, patch_end_i, patch_start_j, patch_end_j: ", patch_start_i, patch_end_i, patch_start_j, patch_end_j)
        patches[i, :, :, :] = global_onehot[:, :, patch_start_i:patch_end_i, \
            patch_start_j:patch_end_j]
    return patches

def convert_colorized_rgb_to_label_onehot(colorized):
    """
    Convert colorized map to onehot map. NOT TENSOR. Last dimension is number of channels
    
    Args:
        colorized (np.array): (height, width, 3)

    Output:
        one_hot (np.array): (height, width, num_class)
    """


    unique_colors = np.array([[255, 255, 255], [0, 0, 204], [51, 255, 51]]) #! hardcode given input data (white, blue, green)
    one_hot = np.zeros((colorized.shape[0], colorized.shape[1], unique_colors.shape[0]), dtype=np.uint8)

    for i, color in enumerate(unique_colors):
        mask = np.all(colorized == color, axis=-1)
        print(np.sum(mask))

        # print(np.sum(mask))
        one_hot[:, :, i][mask] = 255

    return one_hot

def flood_fill_observed_onehot_tensor(cur_obs_onehot, flood_depth=0):
    flood_depth = flood_depth
    cur_obs_class = np.argmax(cur_obs_onehot, axis=2)
    def flood_fill(matrix):
        rows = len(matrix)
        if rows == 0:
            return

        cols = len(matrix[0])
        visited = [[False] * cols for _ in range(rows)]

        stack = []

        def is_valid(row, col):
            return row >= 0 and row < rows and col >= 0 and col < cols

        # Find the occupied cells
        free_cells = [(r, c) for r in range(rows) for c in range(cols) if matrix[r][c] == 2]

        # Initialize the stack with the unknown cells adjacent to occupied cells
        for row, col in free_cells:
            if is_valid(row + 1, col) and matrix[row + 1][col] == 0:  # Down
                stack.append((row + 1, col, 1))
            if is_valid(row - 1, col) and matrix[row - 1][col] == 0:  # Up
                stack.append((row - 1, col, 1))
            if is_valid(row, col + 1) and matrix[row][col + 1] == 0:  # Right
                stack.append((row, col + 1, 1))
            if is_valid(row, col - 1) and matrix[row][col - 1] == 0:  # Left
                stack.append((row, col - 1, 1))

        while stack:
            row, col, depth = stack.pop()
            if visited[row][col]:
                continue
            visited[row][col] = True

            if matrix[row][col] != 0 or depth > flood_depth:
                continue

            # if any of the neighboring cells 3 cells away are occupied, then continue
            neighbor_obs_buffer = 2
            neighbor_cells = matrix[row-neighbor_obs_buffer:row+neighbor_obs_buffer, col-neighbor_obs_buffer:col+neighbor_obs_buffer] 
            if np.any(neighbor_cells == 1):
                continue
            

            matrix[row][col] = 2

            # Add neighboring unknown cells to the stack
            if is_valid(row + 1, col):  # Down
                stack.append((row + 1, col, depth + 1))
            if is_valid(row - 1, col):  # Up
                stack.append((row - 1, col, depth + 1))
            if is_valid(row, col + 1):  # Right
                stack.append((row, col + 1, depth + 1))
            if is_valid(row, col - 1):  # Left
                stack.append((row, col - 1, depth + 1))

        return matrix
    flood_filled_obs = flood_fill(cur_obs_class) #!
    # flood_filled_obs = cur_obs_class #!
    flood_filled_obs_onehot = np.zeros_like(cur_obs_onehot)
    flood_filled_obs_onehot[flood_filled_obs == 0, 0] = 255
    flood_filled_obs_onehot[flood_filled_obs == 1, 1] = 255
    flood_filled_obs_onehot[flood_filled_obs == 2, 2] = 255
    return flood_filled_obs_onehot


def flood_fill_observed_onehot_tensor_from_point(cur_obs_onehot, cur_pred_onehot, flood_start_point, flood_depth=0):
    flood_depth = flood_depth
    cur_obs_class = np.argmax(cur_obs_onehot, axis=2) # change from [H x W x #class] to [H x W]
    cur_pred_class = np.argmax(cur_pred_onehot, axis=2) # change from [H x W x #class] to [H x W]
    enum_void = 0
    enum_occ = 1
    enum_free = 2
    def flood_fill(obs_matrix, pred_matrix):
        """Depth-first search based flood fill algorithm.
        """
        rows = len(obs_matrix)
        if rows == 0:
            return

        cols = len(obs_matrix[0])
        visited = [[False] * cols for _ in range(rows)]

        stack = []

        def is_valid(row, col):
            return row >= 0 and row < rows and col >= 0 and col < cols

        # Find the free cells
        free_cells = [flood_start_point]

        # Initialize the stack with the unknown cells adjacent to occupied cells
        for row, col in free_cells:
            if is_valid(row + 1, col) and obs_matrix[row + 1][col] == enum_void:  # Down
                stack.append((row + 1, col, 1))
            if is_valid(row - 1, col) and obs_matrix[row - 1][col] == enum_void:  # Up
                stack.append((row - 1, col, 1))
            if is_valid(row, col + 1) and obs_matrix[row][col + 1] == enum_void:  # Right
                stack.append((row, col + 1, 1))
            if is_valid(row, col - 1) and obs_matrix[row][col - 1] == enum_void:  # Left
                stack.append((row, col - 1, 1))

        while stack:
            row, col, depth = stack.pop()
            if visited[row][col]:
                continue
            visited[row][col] = True

            # Stopping conditions for flood fill 
            # (1) if the cell is not both: void in observed map and free in predicted map, then continue
            # (2) if the depth is greater than the flood depth, then continue

            if not (obs_matrix[row][col] == enum_void and pred_matrix[row][col] == enum_free):
                continue

            if depth > flood_depth:
                continue

            # if any of the neighboring cells 3 cells away are occupied, then continue
            neighbor_obs_buffer = 2
            neighbor_cells = obs_matrix[row-neighbor_obs_buffer:row+neighbor_obs_buffer, col-neighbor_obs_buffer:col+neighbor_obs_buffer] 
            if np.any(neighbor_cells == enum_occ):
                continue
            

            obs_matrix[row][col] = enum_free

            # Add neighboring unknown cells to the stack
            if is_valid(row + 1, col):  # Down
                stack.append((row + 1, col, depth + 1))
            # Down-left 
            if is_valid(row + 1, col - 1):
                stack.append((row + 1, col - 1, depth + 1))
            # Down-right
            if is_valid(row + 1, col + 1):
                stack.append((row + 1, col + 1, depth + 1))
            if is_valid(row - 1, col):  # Up
                stack.append((row - 1, col, depth + 1))
            # Up-left
            if is_valid(row - 1, col - 1):
                stack.append((row - 1, col - 1, depth + 1))
            # Up-right
            if is_valid(row - 1, col + 1):
                stack.append((row - 1, col + 1, depth + 1))
            if is_valid(row, col + 1):  # Right
                stack.append((row, col + 1, depth + 1))
            if is_valid(row, col - 1):  # Left
                stack.append((row, col - 1, depth + 1))

        return obs_matrix
    flood_filled_obs = flood_fill(cur_obs_class, cur_pred_class) #!
    flood_filled_obs_onehot = np.zeros_like(cur_obs_onehot)
    flood_filled_obs_onehot[flood_filled_obs == 0, 0] = 255
    flood_filled_obs_onehot[flood_filled_obs == 1, 1] = 255
    flood_filled_obs_onehot[flood_filled_obs == 2, 2] = 255
    return flood_filled_obs_onehot