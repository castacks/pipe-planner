import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from skimage.measure import block_reduce
import multiprocessing
from enum import Enum
import time 
import albumentations as A
import argparse
import copy

def calculate_iou_binary(predicted, ground_truth, show_plt=False):
    # Threshold the predicted result to convert to binary mask
    predicted_binary = (predicted > 0.5).astype(np.uint8)
    ground_truth_binary = (ground_truth > 0.5).astype(np.uint8) if ground_truth.max() > 1 else ground_truth
    
    predicted_binary_onechan = predicted_binary[:,:,0]
    ground_truth_binary_onechan = ground_truth_binary[:,:,0]
    
    # Calculate TP, FP, FN
    tp = np.logical_and(predicted_binary_onechan, ground_truth_binary_onechan)
    fp = np.logical_and(predicted_binary_onechan, np.logical_not(ground_truth_binary_onechan))
    fn = np.logical_and(np.logical_not(predicted_binary_onechan), ground_truth_binary_onechan)
    
    # Calculate intersection and union for IoU
    intersection = tp.sum()
    union = fp.sum() + fn.sum() + tp.sum()
    iou = intersection / union if union != 0 else 0
    
    # Visualization
    if show_plt:
        vis_image = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3), dtype=np.uint8)
        vis_image[tp] = [0, 255, 0]  # Green for TP
        vis_image[fp] = [255, 0, 0]  # Red for FP
        vis_image[fn] = [0, 0, 255]  # Blue for FN
        
        plt.imshow(vis_image)
        plt.title(f'IoU: {iou:.4f}, TP (G): {tp.sum()}, FP (R): {fp.sum()}, FN (B): {fn.sum()}')
        plt.show()
    
    return intersection, union, iou

# Function to calculate IoU
def calculate_iou(pred_array, gt_array, show_plt=False):
    """Calculate IOU given prediction and ground truth arrays
    
    Design decisions
    - Get IOU of occupied space
    - If unknown in prediction, it is considered as free space (0)
    - Use mean of all models for evaluation
    
    
    Args:
        pred_array (np.array): H x W x 3 (RGB) [Possible values: 0- 255]
        gt_array (np.array): H x W x 3 [Possible values: 0, 255]

    Returns:
        IOU: float
    """
    start_time = time.time()


    assert pred_array.shape == gt_array.shape, 'Prediction and ground truth arrays must have the same shape {} and {}'.format(pred_array.shape, gt_array.shape)
    
    intersection, union, iou = calculate_iou_binary(pred_array, gt_array, show_plt)
    return iou

# Function to get IoU data for a folder
def get_iou_data(folder, gt_path, load_npy):
    global_pred_path = os.path.join(folder, 'global_pred')
    # import pdb; pdb.set_trace()
    if load_npy: 
        iou_data = np.load(os.path.join(folder, 'iou.npy'))
    else:
        pred_files = sorted([f for f in os.listdir(global_pred_path) if f.endswith('pred.npy')])
        gt_array = np.array(Image.open(gt_path))
        padding_transform = A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=16, pad_width_divisor=16, border_mode=cv2.BORDER_CONSTANT, value=0) # TODO: move this elsewhere
        gt_array = padding_transform(image=gt_array)['image'][500:-500, 500:-500] #! added cropping
        
        # Initialize an empty list to store IoU values along with indices
        iou_data = []
        for fi, f in tqdm(enumerate(pred_files)):
            start_time = time.time()
            index = int(f.split('_')[0])  # Assuming the index is the first part of the filename
            show_plt = False
            global_pred_npy = np.load(os.path.join(global_pred_path, f))[500:-500, 500:-500] #! added cropping
            iou = calculate_iou(global_pred_npy, gt_array, show_plt=show_plt)
            iou_data.append((index, iou))

        iou_save_path = os.path.join(folder, 'iou.npy')
        np.save(iou_save_path, iou_data)
    return iou_data

# Function to calculate coverage
def calculate_coverage(image_path, valid_space_bool, show_plt=False):
    
    image_i = int(image_path.split('/')[-1].split('.')[0])
    image = cv2.imread(image_path)
    image = image[500:-500, 500:-500] #! added cropping
    

    # Check that the image and valid space are the same size (in the first two channels)
    assert image.shape[:2] == valid_space_bool.shape, 'Image shape: {}, valid space shape: {}'.format(image.shape, valid_space_bool.shape)

    # Calculate coverage where valid_space_bool is True
    # image_valid_area = image[valid_space_bool]
    image_known = (image[:, :, 0] != 128)
    image_known_valid_area = image_known[valid_space_bool]
    coverage = np.count_nonzero(image_known_valid_area) / np.count_nonzero(valid_space_bool)
    
    
    # Visualize
    if show_plt:
        plt_row = 1
        plt_col = 2
        plt.figure(figsize=(10, 5))
        plt.subplot(plt_row, plt_col, 1) 
        plt.imshow(image_known)
        plt.title('Known area. Coverage: {:.2f}'.format(coverage))
        plt.subplot(plt_row, plt_col, 2)
        plt.imshow(valid_space_bool)
        plt.title('Valid space')
        plt.savefig('coverage.png')
        plt.close()
    
    return [image_i, coverage]

# Function to get coverage data for a folder
def get_coverage_data(folder):
    # Get the map_id from the folder name
    folder_name = os.path.basename(folder)
    print('Processing folder: {}'.format(folder_name))
    mission_info = parse_mission_info(folder_name)
    print('Processing mission: {}'.format(mission_info))
    map_id = mission_info[0]
    
    # Get the ground truth map from the map_id
    # TODO: parametrize this, later on, should save visble map in each experiment folder
    kth_map_folder_path = os.path.join(root_path, 'kth_test_maps')
    validspace_npy_path = os.path.join(kth_map_folder_path, map_id, 'valid_space.npy')

    assert os.path.exists(validspace_npy_path), 'Valid space does not exist for map_id: {}'.format(map_id)
    valid_space = np.load(validspace_npy_path)
    
    global_obs_path = os.path.join(folder, 'global_obs')
    image_files = sorted(os.listdir(global_obs_path))
    print('Number of images: {}'.format(len(image_files)))
    coverages = [] 
    # TODO: change this to take in directly from experiment folder, when it is available (need to change kth_explore_sim code)
    block_size_pix = 2
    laser_range_pix = 0 #lidar_sim_configs['laser_range_m'] * lidar_sim_configs['pixel_per_meter'] 

    valid_space = block_reduce(valid_space, block_size=(block_size_pix, block_size_pix), func=np.max, cval=0)
    valid_space = np.pad(valid_space, int(laser_range_pix), mode='constant', constant_values=0)

    valid_space_bool = valid_space > 0 # Convert to boolean, valid: 1, invalid/doorways: 0
    for img_i, f in enumerate(image_files[::skip_file_freq]):
        show_plt = False
        coverage_img = calculate_coverage(os.path.join(global_obs_path, f), valid_space_bool, show_plt=show_plt)
        coverages.append(coverage_img)
    coverages = np.array(coverages)
    return coverages

# Main function to process all folders
def process_folder(folder):
    folder_path = os.path.join(exp_parent_dir, folder)
    if os.path.isdir(folder_path):
        coverage_data = get_coverage_data(folder_path)
        np.save(os.path.join(folder_path, 'coverage_data.npy'), coverage_data)
        gt_path = os.path.join(folder_path, 'gt_map.png')
        get_iou_data(folder_path, gt_path, load_npy=False) # Calculate IoU data

def process_folders_with_multiprocessing(parent_folder, num_processes=None):
    exp_folders = os.listdir(parent_folder)
    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_folder, exp_folders), total=len(exp_folders), desc='Processing folders'))
        
def process_folders_single_process(parent_folder):
    exp_folders = sorted(os.listdir(parent_folder))
    input_exp_folders = []
    for exp_folder in exp_folders:
        # if world_list_for_trimming[0] in exp_folders:
        input_exp_folders.append(exp_folder)
    for folder in tqdm(input_exp_folders, desc='Processing folders'):
        process_folder(folder)


def analyze_missions(parent_folder, method_types):
    missions = {}
    method_coverage_data = {}
    method_iou_data = {}

    # Step 1: Identify valid missions
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        coverage_file_path = os.path.join(folder_path, 'coverage_data.npy')
        if os.path.isdir(folder_path) and os.path.exists(coverage_file_path):
            mission_info = parse_mission_info(folder)
            method_type = folder.split('_')[-1]
            coverage_data = np.load(coverage_file_path)
            if coverage_data.size == 0:
                continue
            iou_data_path = os.path.join(folder_path, 'iou.npy')
            if not os.path.exists(iou_data_path):
                continue
            iou_data = np.load(iou_data_path)
            if len(coverage_data) >= mission_length_to_analyze and len(iou_data) >= mission_length_to_analyze:  # Check if the mission has at least 500 timesteps
                if mission_info in missions:
                    missions[mission_info].add(method_type)
                else:
                    missions[mission_info] = {method_type}
            else:
                print('skipping due to insufficient coverage data')
        else:
            print('skipping due to no coverage file')
    valid_missions = {mi for mi, methods in missions.items() if method_types == methods}
    print('Number of valid missions:', len(valid_missions))

    # Step 2: Aggregate data for valid missions
    valid_iou_count = 0 
    invalid_iou_count = 0
    for folder in sorted(os.listdir(parent_folder)):
        folder_path = os.path.join(parent_folder, folder)
        mission_info = parse_mission_info(folder)
        if mission_info in valid_missions:
            method_type = folder.split('_')[-1]
            coverage_data = np.load(os.path.join(folder_path, 'coverage_data.npy'))
            iou_data_path = os.path.join(folder_path, 'iou.npy')
            if os.path.exists(iou_data_path):
                iou_data = np.load(os.path.join(folder_path, 'iou.npy'))
                
            if method_type in method_coverage_data:
                method_coverage_data[method_type].append((mission_info, coverage_data))
                method_iou_data[method_type].append((mission_info, iou_data))
            else:
                method_coverage_data[method_type] = [(mission_info, coverage_data)]
                method_iou_data[method_type] = [(mission_info, iou_data)]

    return method_coverage_data, method_iou_data, valid_missions


def parse_mission_info(folder_name):
    # Split the folder name into parts
    parts = folder_name.split('_')
    # Assuming the first two parts are date and time, and the last part is the method
    # Everything in between is considered part of the map_id
    map_id = '_'.join(parts[2:-3])  # Joining parts of the map_id
    start_x = parts[-3]  # Extracting the start_x
    start_y = parts[-2]  # Extracting the start_y
    return (map_id, start_x, start_y)

def calculate_average_and_variance(method_coverage_data, method_iou_data):
    statistics_data = {}
    # sort items by key name
    method_coverage_data = dict(sorted(method_coverage_data.items()))
    for method, info_data_list in method_coverage_data.items():
        data_list = [elem[1] for elem in info_data_list] # 0: info, 1: data
        # Filter out empty arrays
        non_empty_coverage_data = [] 
        for data in data_list:
            data_i_cap_at_max = []
            # Trim out the data to max_mission_length based on first index 
            # Find index where first index is less than l to max_mission_length
            less_than_equalto_max_mission_length = np.where(data[:,0] < max_mission_length)[0]
            data_i_cap_at_max = data[less_than_equalto_max_mission_length]
            
            non_empty_coverage_data.append(data_i_cap_at_max)
            
        if non_empty_coverage_data:
            stacked_coverage_data = np.stack(non_empty_coverage_data, axis=-1)
            average_coverage = np.mean(stacked_coverage_data, axis=-1)
            variance_coverage = np.var(stacked_coverage_data, axis=-1) / len(non_empty_coverage_data)
            iou_info_data = method_iou_data[method]
            iou_data = [elem[1] for elem in iou_info_data] # 0: info, 1: data
            iou_data_before_mission_length_end = [iou_data_i[iou_data_i[:,0] < max_mission_length] for iou_data_i in iou_data]
            
            stacked_iou_data = np.stack(iou_data_before_mission_length_end, axis=-1)
            average_iou = np.mean(stacked_iou_data, axis=-1)
            variance_iou = np.var(stacked_iou_data, axis=-1) / len(iou_data)
            statistics_data[method] = {
                'average_coverage': average_coverage,
                'variance_coverage': variance_coverage, 
                'average_iou': average_iou,
                'variance_iou': variance_iou
            }
        else:
            statistics_data[method] = {'average_coverage': np.array([]), 
                                       'variance_coverage': np.array([]), 
                                    #    'average_iou': np.array([]), 
                                    #    'variance_iou': np.array([])
                                    }

    return statistics_data

def plot_average_and_variance_separately(statistics_data, valid_missions):
    plt.figure()
    for method, stats in statistics_data.items():
        if stats['average_coverage'].size > 0:
            time_steps = stats['average_coverage'][:, 0]
            average = stats['average_coverage'][:, 1]
            std_deviation = np.sqrt(stats['variance_coverage'][:, 1])
            plt.plot(time_steps, average, label=method, color=method_colors[method],linewidth=3)
    plt.xlabel('Time Step')
    plt.ylabel('Coverage')
    plt.ylim(0, 1)
    plt.title('Coverage over Time, over {} missions (same map, start pose)'.format(len(valid_missions)))
    plt.legend(loc='lower right')
    
    plt.figure()
    for method, stats in statistics_data.items():
        if stats['average_iou'].size > 0:
            time_steps = stats['average_iou'][:, 0]
            average = stats['average_iou'][:, 1]
            # import pdb; pdb.set_trace()
            std_deviation = np.sqrt(stats['variance_iou'][:, 1])

            # Plot average line
            plt.plot(time_steps, average, label=method, color=method_colors[method], linewidth=3)
            print(average)

            # Create envelope representing the variance
            if method == 'nearest' or method == 'visunk':
                plt.fill_between(time_steps, average - std_deviation, average + std_deviation, alpha=0.2, color=method_colors[method])    
    plt.xlabel('Time Step')
    plt.ylabel('IOU')    
    plt.ylim(0, 1)

    plt.title('IOU over Time, over {} missions (same map, start pose)'.format(len(valid_missions)))
    plt.legend()
    plt.show()


def plot_average_and_variance(statistics_data, valid_missions):
    subplot_row = 2
    subplot_col = 1
    fig, ax = plt.subplots(subplot_row, subplot_col, figsize=(10, 10)) 
    ax = ax.flatten()
    for method, stats in statistics_data.items():
        if stats['average_coverage'].size > 0:
            time_steps = stats['average_coverage'][:, 0]
            average = stats['average_coverage'][:, 1]
            std_deviation = np.sqrt(stats['variance_coverage'][:, 1])

            # Plot average line
            ax[0].plot(time_steps, average, label=method, color=method_colors[method], linewidth=3)

            # Create envelope representing the variance
            # if method == 'nearest' or method == 'visunk':
            #     ax[0].fill_between(time_steps, average - std_deviation, average + std_deviation, alpha=0.2, color=method_colors[method])
    ax[0].set_xlabel('Time Step')
    ax[0].set_ylabel('Coverage')
    ax[0].set_ylim(0, 1)
    ax[0].set_title('Coverage over Time, over {} missions (same map, start pose)'.format(len(valid_missions)))
    ax[0].legend(loc='lower right')
    
    for method, stats in statistics_data.items():
        if stats['average_iou'].size > 0:
            time_steps = stats['average_iou'][:, 0]
            average = stats['average_iou'][:, 1]
            # import pdb; pdb.set_trace()
            std_deviation = np.sqrt(stats['variance_iou'][:, 1])

            # Plot average line
            ax[1].plot(time_steps, average, label=method, color=method_colors[method], linewidth=3)
            print(average)

            # Create envelope representing the variance
            if method == 'nearest' or method == 'visunk':
                ax[1].fill_between(time_steps, average - std_deviation, average + std_deviation, alpha=0.2, color=method_colors[method])    
    ax[1].set_xlabel('Time Step')
    ax[1].set_ylabel('IOU')    
    ax[1].set_ylim(0, 1)

    ax[1].set_title('IOU over Time, over {} missions (same map, start pose)'.format(len(valid_missions)))
    ax[1].legend()    
    plt.savefig('coverage_iou.png')
    plt.show()


def plot_average_and_variance_into_files(statistics_data, valid_missions):
    results_file_path = "/home/seungchan/Documents/map_prediction_toolbox/experiments/results/graph_data/"
    for method, stats in statistics_data.items():
        if stats['average_coverage'].size > 0:
            time_steps = stats['average_coverage'][:, 0]
            average = stats['average_coverage'][:, 1]
            std_deviation = np.sqrt(stats['variance_coverage'][:, 1])
            coverage_method_file = os.path.join(results_file_path, 'coverage_'+method+'.npy')
            coverage_data = {'timesteps': time_steps, 'average': average}
            print("coverage", method, len(average))
            np.save(coverage_method_file, coverage_data)
    
    for method, stats in statistics_data.items():
        if stats['average_iou'].size > 0:
            time_steps = stats['average_iou'][:, 0]
            average = stats['average_iou'][:, 1]
            std_deviation = np.sqrt(stats['variance_iou'][:, 1])
            iou_method_file = os.path.join(results_file_path, 'iou_'+method+'.npy')
            iou_data = {'timesteps': time_steps, 'average': average}
            print("iou", method, len(average))
            np.save(iou_method_file, iou_data)


def plot_average_and_variance_twocolumns(statistics_data, valid_missions):
    subplot_row = 1
    subplot_col = 2
    fig, ax = plt.subplots(subplot_row, subplot_col, figsize=(8, 10)) 
    ax = ax.flatten()
    for method, stats in statistics_data.items():
        if stats['average_coverage'].size > 0:
            time_steps = stats['average_coverage'][:, 0]
            average = stats['average_coverage'][:, 1]
            std_deviation = np.sqrt(stats['variance_coverage'][:, 1])

            # Plot average line
            method_label = method
            if method == 'visvarprob':
                method_label = 'MapEx'
            elif method == 'upen':
                method_label = 'UPEN'
            elif method == 'obsunk':
                method_label = 'Observed Map'
            elif method == 'visunk':
                method_label = 'No Variance'
            elif method == 'visvar':
                method_label = 'Deterministic'
            elif method == 'hectoraug':
                method_label = 'IG-Hector'
            elif method == 'nearest':
                method_label = 'Nearest-Frontier'
                
            ax[0].plot(time_steps, average, label=method_label, color=method_colors[method], linewidth=3)

            # Create envelope representing the variance
            # if method == 'nearest' or method == 'visunk':
            #     ax[0].fill_between(time_steps, average - std_deviation, average + std_deviation, alpha=0.2, color=method_colors[method])
    ax[0].set_xlabel('Time Step')
    ax[0].set_ylabel('Coverage')
    ax[0].set_ylim(0, 1)
    ax[0].set_title('Coverage over Time')#, over {} missions (same map, start pose)'.format(len(valid_missions)))
    ax[0].legend() #loc='lower right')
    
    for method, stats in statistics_data.items():
        if stats['average_iou'].size > 0:
            time_steps = stats['average_iou'][:, 0]
            average = stats['average_iou'][:, 1]
            # import pdb; pdb.set_trace()
            std_deviation = np.sqrt(stats['variance_iou'][:, 1])

            # Plot average line
            method_label = method
            if method == 'visvarprob':
                method_label = 'MapEx'
            elif method == 'upen':
                method_label = 'UPEN'
            elif method == 'obsunk':
                method_label = 'Observed Map'
            elif method == 'visunk':
                method_label = 'No Variance'
            elif method == 'visvar':
                method_label = 'Deterministic'
            elif method == 'hectoraug':
                method_label = 'IG-Hector'
            elif method == 'nearest':
                method_label = 'Nearest-Frontier'
            
            ax[1].plot(time_steps, average, label=method_label, color=method_colors[method], linewidth=3)
            print(average)

    ax[1].set_xlabel('Time Step')
    ax[1].set_ylabel('IOU')    
    ax[1].set_ylim(0, 1)

    ax[1].set_title('IOU over Time')#, over {} missions (same map, start pose)'.format(len(valid_missions)))
    ax[1].legend()    
    plt.savefig('coverage_iou.png')
    plt.show()



def plot_individual(method_coverage_data, method_iou_data):
    subplot_row = 2
    subplot_col = 1

    # ind_to_show = 0
    for ind_to_show in range(len(method_coverage_data[list(method_coverage_data.keys())[0]])):
        mission_info = None 
        print("Plotting for ", ind_to_show)
        fig, ax = plt.subplots(subplot_row, subplot_col, figsize=(10, 10)) 
        ax = ax.flatten()
        for method, stats in method_coverage_data.items(): # loop through methods
            # import pdb; pdb.set_trace()
            if mission_info is None:
                mission_info = stats[ind_to_show][0]
            else:
                assert stats[ind_to_show][0] == mission_info
            time_steps = stats[ind_to_show][1][:,0]
            coverage = stats[ind_to_show][1][:,1]
            ax[0].plot(time_steps, coverage, label=method, color=method_colors[method], linewidth=3)
            ax[0].set_xlabel('Time Step')
            ax[0].set_ylabel('Coverage')
            ax[0].set_ylim(0, 1)
            ax[0].set_title('Coverage over Time, mission: {}'.format(mission_info))
            ax[0].legend(loc='lower right')
        
        for method, stats in method_iou_data.items():
            assert stats[ind_to_show][0] == mission_info
            time_steps = stats[ind_to_show][1][:,0]
            iou = stats[ind_to_show][1][:,1]
            # Plot average line
            ax[1].plot(time_steps, iou, label=method, color=method_colors[method], linewidth=3)

    
            ax[1].set_xlabel('Time Step')
            ax[1].set_ylabel('IOU')    
            ax[1].set_ylim(0, 1)

            ax[1].set_title('IOU over Time')
            ax[1].legend()    
        plt.savefig('indi_coverage_iou_{:02d}.png'.format(ind_to_show))
        # plt.show()
        plt.close()

def remove_coverage_data_files(parent_folder):
    print('Removing coverage data files in this parent folder {}, press enter if confirm'.format(parent_folder))
    # Loop through all subdirectories in the parent folder
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        coverage_file_path = os.path.join(folder_path, 'coverage_data.npy')

        # Check if the directory and the coverage data file exist
        if os.path.isdir(folder_path) and os.path.exists(coverage_file_path):
            try:
                os.remove(coverage_file_path)
                print(f"Removed: {coverage_file_path}")
            except OSError as e:
                print(f"Error: {coverage_file_path} : {e.strerror}")
                
def pad_coverage_data_incrementally(coverage_data, target_length=5000):
    """
    Pads the coverage data arrays with the last value to reach a target length,
    while incrementally increasing the value in the first column based on the
    existing increment.

    Args:
        coverage_data (np.array): The original coverage data array.
        target_length (int): The desired length of the array after padding.

    Returns:
        np.array: The padded coverage data array.
    """
    current_length = len(coverage_data)
    if current_length >= target_length:
        return coverage_data

    # Calculate the increment
    if current_length > 1:
        increment = coverage_data[-1, 0] - coverage_data[-2, 0]
    else:
        increment = 1  # Default increment if there's only one element or the array is empty

    # Last value of the second column
    last_value_second_col = coverage_data[-1, 1] if current_length > 0 else 0

    # Create additional values
    additional_length = target_length - current_length
    additional_first_col = coverage_data[-1, 0] + increment * np.arange(1, additional_length + 1)
    additional_second_col = np.full(additional_length, last_value_second_col)

    # Concatenate the original and additional values
    padded_coverage_data = np.vstack((coverage_data, np.column_stack((additional_first_col, additional_second_col))))

    return padded_coverage_data

def get_valid_space(arbitary_value, file_name, desired_shape):
    file_path1 = os.path.join('/home/bsj/MapEx/kth_test_maps', file_name, 'occ_map.npy')
    file_path2 = os.path.join('/home/bsj/MapEx/kth_test_maps', file_name, 'valid_space.npy')
    
    occ_map = np.load(file_path1)
    valid_space = np.load(file_path2)
    
    valid_space[valid_space != 0] = 255
    valid_space[valid_space == 0] = arbitary_value
    valid_space[occ_map == 0] = 255
    valid_space = cv2.resize(valid_space, (desired_shape[1], desired_shape[0]), interpolation=cv2.INTER_NEAREST)
    #valid_space = np.repeat(valid_space[:, :, np.newaxis], 3, axis=2)
    return valid_space

def filter_predicted_map(file_name, predicted_binary_onechan):
    arbitary_value = 200
    desired_shape = (predicted_binary_onechan.shape[0], predicted_binary_onechan.shape[1])
    # plot_data(predicted_binary_onechan, 'pred_before.png', 'Prediction Before Filter')
    resized_occ_map_gt = get_valid_space(arbitary_value, file_name, desired_shape)

    assert resized_occ_map_gt.shape == predicted_binary_onechan.shape, "Shapes do not match. Cannot proceed with the operation."
    
    mask = (resized_occ_map_gt == arbitary_value) & (predicted_binary_onechan != 0)
    predicted_binary_onechan[mask] = 0
    predicted_binary_onechan[predicted_binary_onechan != 0] = 1
    # plot_data(predicted_binary_onechan, 'pred_after.png', 'Prediction After Filter')
    
    return predicted_binary_onechan

def calculate_iou_kth(pred_array, gt_array, show_plt=False, file_name=None):
    pred_array = pred_array[:,:,0]
    #gt_array = gt_array[:,:,0]
    dilation_kernel_size = 5
    # print("original pred_array shape is: ", pred_array.shape)
    # print("original pred_array sum is: ", np.sum(pred_array))
    # print("original gt_array shape is: ", gt_array.shape)
    # print("original gt_array sum is: ", np.sum(gt_array))

    # Threshold the predicted result to convert to binary mask
    predicted_binary = (pred_array > 0.5).astype(np.uint8)

    #gt_array_binary = (gt_array > 0.5).astype(np.uint8) if gt_array_mask.max() > 1 else gt_array_mask.astype(np.uint8)
    gt_array_binary = (gt_array > 0.5).astype(np.uint8) if gt_array.max() > 1 else gt_array.astype(np.uint8)

    pred_correcter = copy.deepcopy(predicted_binary)
    pred_correcter = filter_predicted_map(file_name, pred_correcter)

    # Applying dilation to both images
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    # # giving less advantage to the false predictions
    # kernel2 = np.ones((2, 2), np.uint8)
    predicted_dilated = cv2.dilate(predicted_binary, kernel, iterations=1)
    gt_array_dilated = cv2.dilate(gt_array_binary, kernel, iterations=1)

    # print("pred array is: ", predicted_dilated.shape)
    # print("pred array sum is: ", np.sum(predicted_dilated))
    # print("gt_array is: ", gt_array_dilated.shape)
    # print("gt_array sum is: ", np.sum(gt_array_dilated))

    
    tp = np.logical_and(predicted_dilated, gt_array_binary)
    fp = np.logical_and(np.logical_and(predicted_binary, pred_correcter), np.logical_not(gt_array_dilated))
    fn = np.logical_and(np.logical_not(predicted_dilated), gt_array_binary)
    
    # Calculate intersection and union for IoU
    intersection = tp.sum()
    union = fp.sum() + fn.sum() + tp.sum()
    iou = intersection / union if union != 0 else 0

    # Visualization
    # if show_plt:
    #     vis_image = np.zeros((predicted_dilated.shape[0], gt_array_dilated.shape[1], 3), dtype=np.uint8)
    #     vis_image[tp] = [0, 255, 0]  # Green for TP
    #     vis_image[fp] = [255, 0, 0]  # Red for FP
    #     vis_image[fn] = [0, 0, 255]  # Blue for FN
        
    #     plt.figure
    #     plt.imshow(vis_image)
    #     plt.title(f'IoU: {iou:.4f}, TP (G): {tp.sum()}, FP (R): {fp.sum()}, FN (B): {fn.sum()}')
    #     plt.show()    
    return iou

# Call the main function with the parent directory
# this is generally directory that is used to run a lot of compare missions with kth_explore_sim.py
# Toggle modes
if __name__ == '__main__':
    run_remove_coverage_data_files = True
    run_process = True
    run_plotting = True
    # Predefine colors
    method_colors = {
        'nearest': 'black',
        'obsunk': 'blue',
        'onlyvar': 'orange',
        'visunk': 'purple',
        'visvar': 'green',
        'visvarprob': 'red',
        'upen': 'green',
        #'hectoraug': 'cyan'
    }

    root_path = '/home/seungchan/MapEx/'#
    experiment_folder_path = os.path.join(root_path, 'experiments')
    exp_parent_dir = '20250110_test' #customize
    exp_parent_dir = os.path.join(experiment_folder_path, exp_parent_dir)
    skip_file_freq = 50
    max_mission_length = 1000
    num_processes = 5
    use_multiprocessing = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_world_list', nargs='+', help='List of worlds to collect data from')
    args = parser.parse_args()
    world_list_for_trimming = None 

    os.environ["MKL_NUM_THREADS"] = "1"
    os.system('taskset -p --cpu-list 0-128 %d' % os.getpid()) 

    mission_length_to_analyze = max_mission_length // skip_file_freq
    print('Mission length to analyze: {}'.format(mission_length_to_analyze))
    if run_remove_coverage_data_files:
        remove_coverage_data_files(exp_parent_dir)
    if run_process:
        if use_multiprocessing:
            process_folders_with_multiprocessing(exp_parent_dir, num_processes)
        else: 
            process_folders_single_process(exp_parent_dir) # usually for debugging, since pdb doesn't work with multiprocessing
    if run_plotting:
        method_coverage_data, method_iou_data, valid_missions = analyze_missions(exp_parent_dir, method_colors.keys())
        # plot_individual(method_coverage_data, method_iou_data)
        statistics_data  = calculate_average_and_variance(method_coverage_data, method_iou_data)
        
        plot_average_and_variance(statistics_data, valid_missions)
        #plot_average_and_variance_twocolumns(statistics_data, valid_missions)
        #plot_average_and_variance_separately(statistics_data, valid_missions)
        #plot_average_and_variance_into_files(statistics_data, valid_missions)
