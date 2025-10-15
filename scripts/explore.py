import numpy as np
import os 
import cv2
import time
from omegaconf import OmegaConf
import hydra 
import torch 
from torchvision.transforms import ToTensor, Resize, Compose
import pyastar2d    
import json
from skimage.measure import block_reduce
import albumentations as A
import traceback
import argparse
from pdb import set_trace as bp
from collections import deque
from queue import PriorityQueue
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from lama_pred_utils import load_lama_model, visualize_prediction, get_lama_transform, convert_obsimg_to_model_input
from calc_metrics_subdirectory import calculate_iou_kth
import sys
sys.path.append('../')
from scripts import simple_mask_utils as smu 
import scripts.sim_utils as sim_utils
import upen_baseline
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from contextlib import contextmanager

@contextmanager
def dummy_context_manager():
    yield None


def get_options_dict_from_yml(config_name):
    cwd = os.getcwd()
    hydra_config_dir_path = os.path.join(cwd, '../configs')
    print(hydra_config_dir_path)
    with hydra.initialize_config_dir(config_dir=hydra_config_dir_path):
        cfg = hydra.compose(config_name=config_name)
    options_dict = OmegaConf.to_container(cfg)
    options = OmegaConf.create(options_dict)
    return options

def update_mission_status(start_time, cur_step, mission_complete, fail_reason, mission_status_save_path):
    mission_status = {}
    mission_status['start_time'] = start_time
    mission_status["cur_step"] = cur_step
    mission_status["mission_complete"] = mission_complete
    mission_status["fail_reason"] = fail_reason
    mission_status["last_exp_time_s"] = time.time() - mission_status['start_time']
    with open(mission_status_save_path, 'w') as f:
        json.dump(mission_status, f)

def get_lama_pred_from_obs(cur_obs_img, lama_model, lama_map_transform, device):
    cur_obs_img_3chan = np.stack([cur_obs_img, cur_obs_img, cur_obs_img], axis=2)
    input_lama_batch, lama_mask = convert_obsimg_to_model_input(cur_obs_img_3chan, lama_map_transform, device)
    lama_pred_alltrain = lama_model(input_lama_batch)
    lama_pred_alltrain_viz = visualize_prediction(lama_pred_alltrain, lama_mask)
    return cur_obs_img_3chan, input_lama_batch, lama_mask, lama_pred_alltrain, lama_pred_alltrain_viz

def get_pred_maputils_from_viz(viz_map):
    pred_maputils = np.zeros((viz_map.shape[0], viz_map.shape[1]))
    pred_maputils[viz_map[:,:,0] > 128] = 1 # occ
    return pred_maputils

def get_lama_padding_transform():
    lama_padding_transform = A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=16, pad_width_divisor=16, border_mode=cv2.BORDER_CONSTANT, value=0)
    return lama_padding_transform

def get_padded_obs_map(obs_map):
    lama_padding_transform = get_lama_padding_transform()
    padded_obs_map = lama_padding_transform(image=obs_map)['image']
    return padded_obs_map

def get_padded_gt_map(gt_map):
    lama_padding_transform = get_lama_padding_transform()
    padded_gt_map = lama_padding_transform(image=gt_map)['image']
    return padded_gt_map

def is_locked_frontier_center_valid(locked_frontier_center, occ_grid_pyastar, cur_pose, collect_opts, pixel_per_meter):
    if locked_frontier_center is None:
        #print("Locked frontier center is None")
        return False
    if occ_grid_pyastar[locked_frontier_center[0], locked_frontier_center[1]] == np.inf:
        #print("Locked frontier center is inf")
        return False
    if np.linalg.norm(locked_frontier_center - cur_pose) < collect_opts.cur_pose_dist_threshold_m * pixel_per_meter:
        #print("Locked frontier center is too close to current pose")
        return False
    return True

def reselect_frontier_from_frontier_region_centers(frontier_region_centers, total_cost, t, start_exp_time, mission_status_save_path):
    frontier_selected = False
    
    # First check if we have any frontiers left
    if len(frontier_region_centers) == 0:
        mission_failed = True
        update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, fail_reason="frontier_region_centers", mission_status_save_path=mission_status_save_path)
        return frontier_selected, None, None, None
    
    # Find the minimum cost index
    min_cost_idx = np.argmin(total_cost)
    
    # Store the minimum cost frontier before deletion
    locked_frontier_center = frontier_region_centers[min_cost_idx]
    
    # Remove the selected frontier and its cost
    frontier_region_centers = np.delete(frontier_region_centers, min_cost_idx, axis=0)
    total_cost = np.delete(total_cost, min_cost_idx, axis=0)
    
    # Check if we have any frontiers left after deletion
    if len(frontier_region_centers) == 0:
        mission_failed = True
        update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, fail_reason="frontier_region_centers", mission_status_save_path=mission_status_save_path)
        return frontier_selected, None, None, None
    
    frontier_selected = True
    return frontier_selected, locked_frontier_center, frontier_region_centers, total_cost

def determine_local_planner(mode):
    if mode == 'upen':
        return 'astar'
    elif mode in ['nearest', 'visvar', 'visunk', 'nbv-2d', 'onlyvar', 'mapex', 'pipe', 'pw-nbv-2d']:
        return 'astar'
    elif mode == 'hector' or mode == 'hectoraug':
        return 'gradient'
    else:
        raise ValueError("Invalid mode: {}".format(mode))

def get_hector_exploration_transform_map(occgrid, frontiers, init_cost, mode, infogain_val_list, info_gain_weight):
    # Check that correct mode is used
    if mode == 'hector':
        assert(infogain_val_list is None), "init_frontier_value must be None for hector mode"
    elif mode == 'hectoraug':
        assert(len(infogain_val_list) == len(frontiers)), "init_frontier_value must be the same length as frontiers for hector aug"
    # If mode is augmented hector, we calculate the info_gain_cost from info_gain_value
    if mode == 'hectoraug':
        max_infogain = np.max(infogain_val_list)
        infogain_cost_list = []
        for infogain_val in infogain_val_list:
            # https://github.com/rakeshshrestha31/hector_navigation/blob/a211fca5926c1bdf00d71e9f6db8899fc11e7a58/hector_exploration_planner/src/hector_exploration_planner.cpp#L540
            infogain_cost_list.append(int(info_gain_weight * (np.sqrt(max_infogain) - np.sqrt(infogain_val))))
    # Do BFS to get the exploration transform map, queue initialized with frontiers
    cost_map =  np.full_like(occgrid, np.inf)    
    # Initialize a queue for BFS, and initialize the cost map with the frontier costs
    queue = PriorityQueue()
    for frontier_i, frontier in enumerate(frontiers):
        queue.put((0, (frontier[0], frontier[1])))
        if mode == 'hector':
            cost_map[frontier[0], frontier[1]] = 0 # Initialize frontier cost to 0, as it is a distance transform where frontiers are the goals
        elif mode == 'hectoraug':
            cost_map[frontier[0], frontier[1]] = infogain_cost_list[frontier_i] # Initialize frontier cost to infogain value
        else:
            raise ValueError("Invalid mode: {}".format(mode))
    neighbors = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    # Perform BFS
    count = 0
    while not queue.empty():
        cur_ind = queue.get()[1]
        # Current cost of this cell
        cur_cost = cost_map[cur_ind[0], cur_ind[1]]
        for neighbor in neighbors:
            new_x = cur_ind[0] + neighbor[0]
            new_y = cur_ind[1] + neighbor[1]
            # Check if the neighbor is within bounds and is a free space (not an obstacle or unknown cell)
            if 0 <= new_x < cost_map.shape[0] and 0 <= new_y < cost_map.shape[1] and (occgrid[new_x, new_y] == 0):
                new_cost = (cur_cost + 1) + init_cost[new_x, new_y]  # Distance to neighbor is 1 more than current
                # If the neighbor has not been visited or the new distance is shorter
                if cost_map[new_x, new_y] > new_cost:
                    cost_map[new_x, new_y] = new_cost
                    queue.put((new_cost, (new_x, new_y)))
        count += 1
    return cost_map

def gradient_planner(cur_pose, cost_transform_map):
    # Find next pose (neighbor with the steepest gradient)
    neighbor_inds = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    cur_query_pose = np.array([cur_pose[0], cur_pose[1]])
    cur_max_grad = np.inf 
    # Find the neighbor with the most negative gradient, use it as next pose
    grad_list = []
    for neighbor_ind in neighbor_inds:
        neighbor_pose = cur_query_pose + neighbor_ind
        grad_to_neighbor = cost_transform_map[neighbor_pose[0], neighbor_pose[1]] - cost_transform_map[cur_query_pose[0], cur_query_pose[1]]
        grad_list.append(grad_to_neighbor)
        if grad_to_neighbor < cur_max_grad:
            cur_max_grad = grad_to_neighbor
            next_pose = neighbor_pose
    return next_pose

def run_exploration_comparison_for_map(args):
    map_folder_path = args['map_folder_path']
    models_list = args['models_list']
    lama_model = args['lama_model']
    lama_map_transform = args['lama_map_transform']
    pred_vis_configs = args['pred_vis_configs']
    lidar_sim_configs = args['lidar_sim_configs']
    start_pose = args['start_pose']
    modes_to_test = args['modes_to_test']
    unknown_as_occ = args['unknown_as_occ']
    use_distance_transform_for_planning = args['use_distance_transform_for_planning']
    upen_config  = args['upen_config']
    print("Running exploration comparison for map: ", map_folder_path)
    
    # Load in occupancy map and valid space map
    map_occ_npy_path = os.path.join(map_folder_path, 'occ_map.npy')
    assert os.path.exists(map_occ_npy_path), "Occupancy map path does not exist: {}".format(map_occ_npy_path)
    map_valid_space_npy_path = os.path.join(map_folder_path, 'valid_space.npy')
    assert os.path.exists(map_valid_space_npy_path), "Valid space map path does not exist: {}".format(map_valid_space_npy_path)
    occ_map, validspace_map = sim_utils.get_kth_occ_validspace_map(map_occ_npy_path, map_valid_space_npy_path)
    map_name = os.path.dirname(map_occ_npy_path)

    # Sample random start pose, if start_pose is None
    if start_pose is None:
        buffer_start_pose = 2
        start_pose = smu.sample_free_position_given_buffer(occ_map, validspace_map, buffer_start_pose)
        assert start_pose is not None, "Could not sample start pose"

    plt.imshow(occ_map)
    plt.scatter(start_pose[1], start_pose[0], c='r', s=10, marker='*')
    plt.title('GT Map (Red: Start Pose)')
    plt.savefig('gt_map.png')
    plt.close()
        
    # get overall experiment name
    # exp_title based on start time YYYYMMDD_HHMMSS, and name of map
    folder_name = os.path.basename(map_name)
    comp_exp_title = time.strftime("%Y%m%d_%H%M%S") + '_' + folder_name + '_' + str(start_pose[0]) + '_' + str(start_pose[1])
        
    for mode in modes_to_test:
        run_exploration_for_map(occ_map, comp_exp_title, models_list, lama_model, lama_map_transform, pred_vis_configs, lidar_sim_configs, mode, start_pose, unknown_as_occ, \
            use_distance_transform_for_planning=use_distance_transform_for_planning, upen_config=upen_config)

def run_exploration_for_map(occ_map, exp_title, models_list,lama_alltrain_model, lama_map_transform, pred_vis_configs, lidar_sim_configs, mode, \
    start_pose, unknown_as_occ, use_distance_transform_for_planning, upen_config=None):
    try: 
        print("exp_title:", exp_title)
        start_exp_time = time.time()
        pixel_per_meter = lidar_sim_configs['pixel_per_meter']
        use_model = determine_use_model(mode) #determine if the mode requires lama model

        # Planner setup (Mapper, Planner)
        mapper = sim_utils.Mapper(occ_map, lidar_sim_configs, use_distance_transform_for_planning=use_distance_transform_for_planning)
        if mode != 'upen':
            frontier_planner = sim_utils.FrontierPlanner(score_mode=mode)

        # Create a new directory for experiment/exp_title
        exp_title = exp_title + '_' + mode
        exp_dir = os.path.join(output_root_dir, exp_title)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        # Create subdirectory (global_obs, run_viz) and save paths
        global_obs_dir = os.path.join(exp_dir, 'global_obs')
        run_viz_dir = os.path.join(exp_dir, 'run_viz')
        for dir_path in [global_obs_dir, run_viz_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        gt_map_save_path = os.path.join(exp_dir, 'gt_map.png')
        odom_npy_save_path = os.path.join(exp_dir, 'odom.npy')


        # Visualization Setup 
        if mode in ['pipe', 'mapex']:
            plt_row = 2
            plt_col = 2
            fig, ax = plt.subplots(plt_row, plt_col, figsize=(20, 10))
            ax_flatten = ax.flatten()
            ax_gt = ax_flatten[0]
            ax_obs = ax_flatten[1]
            ax_pred_var = ax_flatten[2]
            ax_mean_map = ax_flatten[3]
            pred_maputils = None
            var_map = None
            mean_map = None
            lama_reduced_pred_var = None 
            padded_gt_map = None
        elif mode in ['nbv-2d', 'pw-nbv-2d']:
            plt_row = 2
            plt_col = 1
            fig, ax = plt.subplots(plt_row, plt_col, figsize=(15, 10))
            ax_flatten = ax.flatten()
            ax_gt = ax_flatten[0]
            ax_obs = ax_flatten[1]
            pred_maputils = None
            var_map = None
            mean_map = None
            lama_reduced_pred_var = None 
            padded_gt_map = None
        else:
            raise ValueError(f"Unknown method: {mode}")

        # Initialize mission status save path
        mission_status_save_path = os.path.join(exp_dir, 'mission_status.json')
        mission_failed = False
        update_mission_status(start_time=start_exp_time, cur_step=0, mission_complete=False, fail_reason="", mission_status_save_path=mission_status_save_path)

        #initial observation
        cur_pose = start_pose
        mapper.observe_and_accumulate_given_pose(cur_pose)
        ind_to_move_per_step = 3
        pose_list = np.atleast_2d(cur_pose) # Assumes last pose is the current pose

        pd_size = 500 #padding_size; padded pixels for raycast at the boundary of map
        #there is a small offset between observed map size and lama prediction output size
        gt_h, gt_w = mapper.gt_map.shape[0],mapper.gt_map.shape[1]
        pad_h = gt_h%16
        pad_w = gt_w%16
        if pad_h == 0:
            pad_h1 = 0
            pad_h2 = 0
        else:
            pad_h1 = int((16-pad_h)/2)
            pad_h2 = 16-pad_h - pad_h1
        if pad_w == 0:
            pad_w1 = 0
            pad_w2 = 0
        else:
            pad_w1 = int((16-pad_w)/2)
            pad_w2 = 16-pad_w - pad_w1

        # initial saves (gt_map, pose_list)
        cv2.imwrite(gt_map_save_path, smu.convert_01_single_channel_to_0_255_3_channel(mapper.gt_map))
        np.save(odom_npy_save_path, pose_list)

        locked_frontier_center = None
        
        ### Main Loop
        # Create a timestamp string with date and time, e.g. "20250319_155031"
        if collect_opts.log_iou:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_subdirectory_name = f"{timestamp}_test"

            # Use this for your log directory
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', output_subdirectory_name)
            os.makedirs(log_dir, exist_ok=True)
            iou_log_file = os.path.join(log_dir, f"{exp_title}.txt")


        # Determine time_step based on exp_title
        parts = exp_title.split('_')
        file_name = '_'.join(parts[2:-3])
        # Define time step settings for each category: (with_log, without_log)
        time_settings = {
            'large': (6001, 2001),
            'small': (1501, 501),
            'medium': (3001, 1001)
        }

        # Determine the map category based on file_name
        if file_name in ('50015847', '50015848'):
            category = 'large'
        elif file_name in ('50052749', '50052750'):
            category = 'small'
        else:
            category = 'medium'

        # Choose the appropriate time step based on whether logging is enabled
        time_step = time_settings[category][0] if collect_opts.log_iou else time_settings[category][1]

        iou_90_flag = False
        iou_95_flag = False
        context = open(iou_log_file, "a") if collect_opts.log_iou else dummy_context_manager()

        with context as log_file:
            for t in range(time_step):
                start_mission_i_time = time.time()
                show_plt = (t % collect_opts.show_plt_freq == 0) or (t == collect_opts.mission_time - 1)
                
                # Frontier detection
                if t == 0:
                    if mode != 'upen':
                        frontier_region_centers_unscored, filtered_map, num_large_regions = frontier_planner.get_frontier_centers_given_obs_map(mapper.obs_map)
                
                if mode != 'upen':
                    if len(frontier_region_centers_unscored) == 0:
                        mission_failed = True
                        update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, fail_reason="frontier_region_no_large_region", mission_status_save_path=mission_status_save_path)
                        break
                
                if mode == 'hector' or mode == 'hectoraug':
                    do_hector_plan = (t % 1 == 0) #Hector Explorer (baseline), repeat exploration every 1s 
                    if do_hector_plan:
                        frontier_region_centers_unscored, filtered_map, num_large_regions = frontier_planner.get_frontier_centers_given_obs_map(mapper.obs_map)
                        frontier_region_centers = frontier_region_centers_unscored # Hector Explorer (baseline), there are no scores to frontiers, so we just use zeros for total costs
                        frontier_cost_list = np.zeros(len(frontier_region_centers))
                        
                        if mode == 'hectoraug':
                            hector_cur_obs_img = mapper.obs_map.copy()
                            cur_obs_img_3chan, input_lama_batch, lama_mask, lama_pred_alltrain, lama_pred_alltrain_viz = \
                                get_lama_pred_from_obs(hector_cur_obs_img, lama_alltrain_model, lama_map_transform, device)
                            pred_maputils = get_pred_maputils_from_viz(lama_pred_alltrain_viz)
                            hector_frontier_region_centers_unscored = frontier_region_centers_unscored
                            padded_obs_map = get_padded_obs_map(mapper.obs_map)
                            padded_gt_map = get_padded_gt_map(mapper.gt_map)
                            frontier_region_centers, frontier_cost_list, viz_most_flooded_grid, viz_medium_flooded_grid, best_ind, medium_ind = \
                                    frontier_planner.score_frontiers(hector_frontier_region_centers_unscored, cur_pose, \
                                                                    pose_list, pred_maputils, pred_vis_configs, \
                                                                        obs_map=padded_obs_map, mean_map=mean_map, var_map=var_map)

                # Get inflated obs map for local planner
                occ_grid_pyastar = mapper.get_inflated_planning_maps(unknown_as_occ=unknown_as_occ)   
                if mode not in ['hector', 'hectoraug']: # Non-hector methods, uses some sort of end goal to plan towards (Frontiers or UPEN RRT goals)
                    # Check if close enough to locked frontier center, if so unlock 
                    if locked_frontier_center is not None:
                        if np.linalg.norm(locked_frontier_center - cur_pose) < collect_opts.cur_pose_dist_threshold_m * pixel_per_meter:
                            locked_frontier_center = None

                    need_new_locked_frontier = False
                    if mode == 'upen': #UPEN baseline only
                        if locked_frontier_center is None:
                            need_new_locked_frontier = True
                        else: 
                            upen_goal_pose_freq = upen_config['goal_pose_freq']
                            at_goal_pose_freq = t % upen_goal_pose_freq == 0
                            lock_frontier_center_is_invalid = not is_locked_frontier_center_valid(locked_frontier_center, occ_grid_pyastar, cur_pose, collect_opts, pixel_per_meter)  
                            cannot_reach_frontier_center = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, locked_frontier_center, allow_diagonal=False) is None
                            need_new_locked_frontier = at_goal_pose_freq \
                                or lock_frontier_center_is_invalid \
                                or cannot_reach_frontier_center
                            if need_new_locked_frontier:
                                print("UPEN: freq: {}, invalid: {}, cannot reach: {}".format(at_goal_pose_freq, lock_frontier_center_is_invalid, cannot_reach_frontier_center))
                    else:
                        need_new_locked_frontier = not is_locked_frontier_center_valid(locked_frontier_center, occ_grid_pyastar, cur_pose, collect_opts, pixel_per_meter)
                                    
                    if need_new_locked_frontier:
                        show_plt = True
                                                        
                        # Predict map -----
                        pred_maputils = None
                        var_map = None
                        mean_map = None
                        padded_gt_map = None
                        if use_model:
                            cur_obs_img = mapper.obs_map.copy()
                            # LAMA global prediction (trained with all training set)
                            cur_obs_img_3chan, input_lama_batch, lama_mask, lama_pred_alltrain, lama_pred_alltrain_viz = \
                                get_lama_pred_from_obs(cur_obs_img, lama_alltrain_model, lama_map_transform, device)
                            # Get ensemble prediction trained on subsets
                            lama_pred_list = []
                            #lama_reduced_pred_list = []
                            for model_i, model in enumerate(model_list):
                                #print("predicting with model: ", model_i)
                                pred_time_start = time.time()
                                lama_pred = model(input_lama_batch)
                                #print("Prediction took {} seconds".format(np.round(time.time() - pred_time_start, 2)))
                                lama_pred_viz = visualize_prediction(lama_pred, lama_mask)
                                lama_pred_onechan = lama_pred['inpainted'][0][0]
                                lama_pred_list.append(lama_pred_onechan)

                            
                            # Get variance across batch dimension 
                            lama_pred_list = torch.stack(lama_pred_list)
                            var_map = torch.var(lama_pred_list, dim=0)
                            mean_map = np.mean(lama_pred_list.cpu().numpy(), axis=0)

                            pred_maputils = get_pred_maputils_from_viz(lama_pred_alltrain_viz)
                        
                        padded_obs_map = get_padded_obs_map(mapper.obs_map)
                        padded_gt_map = get_padded_gt_map(mapper.gt_map)

                        if mode == 'upen': # not a frontier-based approach, but we will output UPEN RRT goal as a frontier to visit to use go to frontier logic
                            upen_path = None
                            while upen_path is None:
                                assert upen_config is not None, "upen_config must be provided in yml"
                                upen_rrt_max_iters = upen_config['rrt_max_iters']
                                upen_expand_dis = upen_config['expand_dis']
                                upen_goal_sample_rate = upen_config['goal_sample_rate']
                                upen_connect_circle_dist = upen_config['connect_circle_dist']
                                upen_rrt_num_path = upen_config['rrt_num_path']
                                upen_rrt_straight_line = upen_config['rrt_straight_line']
                                upen_reach_horizon = upen_config['reach_horizon']
                                pose_coords = torch.tensor([[[cur_pose[1], cur_pose[0]]]]).cuda()
                                planning_grid = torch.tensor(padded_obs_map).cuda() 
                                end_pose = [planning_grid.shape[0]-20, planning_grid.shape[1]-20]
                                goal_pose_coords = torch.tensor([[[end_pose[1], end_pose[0]]]]).cuda()
                                
                                occ_chan = 1
                                ensemble = torch.zeros((lama_pred_list.shape[0], 1, 2, planning_grid.shape[0], planning_grid.shape[1])).cuda()
                                ensemble[:, 0, occ_chan, :, :] = lama_pred_list
                                upen_rrt_goal, upen_rrt_best_path, upen_path_dict = upen_baseline.get_rrt_goal(pose_coords=pose_coords.clone(), goal=goal_pose_coords.clone(), grid=planning_grid, ensemble=ensemble,  
                                                                                                            rrt_max_iters=upen_rrt_max_iters, expand_dis=upen_expand_dis, goal_sample_rate=upen_goal_sample_rate, connect_circle_dist=upen_connect_circle_dist, rrt_num_path=upen_rrt_num_path,  
                                                                                                            rrt_straight_line=upen_rrt_straight_line, reach_horizon=upen_reach_horizon, upen_mode='exploration')
                                if upen_rrt_goal is not None:
                                    intermediate_goal_pose = upen_rrt_goal[0][0].cpu().numpy()
                                    intermediate_goal_pose = intermediate_goal_pose[::-1]
                                else: # Use A* to get intermediate goal
                                    print("No intermediate goal found, using A* to get intermediate goal")
                                    astar_path_to_goal = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, end_pose, allow_diagonal=False)
                                    intermediate_goal_pose = astar_path_to_goal[np.min([20, len(astar_path_to_goal)-1])]
                                # Use the goal as the frontier to go to
                                locked_frontier_center = intermediate_goal_pose
                                upen_path = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, locked_frontier_center, allow_diagonal=False)
                        
                        else:
                            #frontier-based methods: recalculate all frontier center scores
                            frontier_region_centers_unscored, filtered_map, num_large_regions = frontier_planner.get_frontier_centers_given_obs_map(mapper.obs_map)
                            

                            if mode in ['pipe', 'pw-nbv-2d']: # Do pathwise method
                                # Initialize an empty path collection and a new list for valid frontiers
                                path_collection = []
                                valid_frontiers_unscored = []

                                # Iterate through the frontier list and find paths
                                for frontier in frontier_region_centers_unscored:
                                    path = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, frontier, allow_diagonal=False)
                                    
                                    # If a valid path is found, append both the path and the frontier to respective lists
                                    if path is not None:
                                        valid_frontiers_unscored.append(frontier)
                                        path_collection.append(path)

                                frontier_region_centers, frontier_cost_list, viz_most_flooded_grid, viz_medium_flooded_grid, best_ind, medium_ind, locked_frontier_center \
                                    = frontier_planner.score_frontiers_pathwise(t, valid_frontiers_unscored, path_collection, pred_maputils, pred_vis_configs, \
                                                                            obs_map=padded_obs_map, mean_map=mean_map, var_map=var_map)    
                            
                            else: # Do point-wise method
                                frontier_region_centers, frontier_cost_list, viz_most_flooded_grid, viz_medium_flooded_grid, best_ind, medium_ind = \
                                        frontier_planner.score_frontiers(frontier_region_centers_unscored, cur_pose, \
                                                                        pose_list, pred_maputils, pred_vis_configs, \
                                                                            obs_map=padded_obs_map, mean_map=mean_map, var_map=var_map)
                                locked_frontier_center = frontier_region_centers[np.argmin(frontier_cost_list)]

                            while not is_locked_frontier_center_valid(locked_frontier_center, occ_grid_pyastar, cur_pose, collect_opts, pixel_per_meter):
                                print("Invalid locked frontier center, reselecting")
                                frontier_selected, locked_frontier_center, frontier_region_centers, frontier_cost_list = reselect_frontier_from_frontier_region_centers(frontier_region_centers, frontier_cost_list, t, start_exp_time, mission_status_save_path)
                                if not frontier_selected:
                                    mission_failed = True
                                    update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, fail_reason="frontier_region_centers", mission_status_save_path=mission_status_save_path)
                                    break

                else: # Hector Explorer Baseline, from frontier clusters (frontier region centers), make a cost map
                    if do_hector_plan: # Do Hector cost map calculation every 1 time steps
                    
                        # Get distance transform to obstacles 
                        obstacle_transform_map = np.ones(mapper.obs_map.shape)
                        obs_obstacles = np.where(mapper.obs_map == 1)
                        obstacle_transform_map[obs_obstacles] = 0
                        obstacle_transform_map = cv2.distanceTransform(obstacle_transform_map.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) 
                        # Follow Eq.5 of the Hector Explorer paper 
                        obs_ramp_dist = 15
                        obs_min_dist = 2
                        danger_transform_map = np.zeros(mapper.obs_map.shape)
                        danger_transform_map[obstacle_transform_map < obs_ramp_dist] = (obs_ramp_dist - obstacle_transform_map[obstacle_transform_map < obs_ramp_dist]) * 5                
                        # Make a path planning distance transform map of the same size as observed map from the frontier region centers 
                        # Algo: do a BFS from frontiers and expand into free space 
                        if mode == 'hector':
                            init_frontier_value = None # Initialize to None since we are not using frontier value cost as in Hector Explorer paper
                        elif mode == 'hectoraug':
                            init_frontier_value = -1 * frontier_cost_list # Initialize to the frontier value cost as in Enhanced Exploration with Predicted Map paper. Need to negate since we want to pass in as higher information gain is higher
                        else:
                            raise ValueError("Invalid mode: {}, mode not available for getting init_frontier_value".format(mode))
                        cost_transform_map = get_hector_exploration_transform_map(mapper.obs_map, frontier_region_centers, init_cost=danger_transform_map, mode=mode, infogain_val_list=init_frontier_value, info_gain_weight=200)

                # Local planning
                chosen_local_planner = determine_local_planner(mode) # astar or gradient
                
                if chosen_local_planner == 'astar':
                    path = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, locked_frontier_center, allow_diagonal=False)
                    while path is None:
                        frontier_selected, locked_frontier_center, frontier_region_centers, frontier_cost_list = reselect_frontier_from_frontier_region_centers(frontier_region_centers, frontier_cost_list, t, start_exp_time, mission_status_save_path)
                        if not frontier_selected:
                            mission_failed = True
                            update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, fail_reason="frontier_region_centers", mission_status_save_path=mission_status_save_path)
                            break
                        path = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, locked_frontier_center, allow_diagonal=False)
                    if mission_failed:
                        break
                
                    plan_x = path[:,0]
                    plan_y = path[:,1]        
                        
                    next_pose = sim_utils.psuedo_traj_controller(plan_x, plan_y, plan_ind_to_use=ind_to_move_per_step)
                elif chosen_local_planner == 'gradient':
                    # Given current pose, find the steepest gradient direction to go in
                    for _ in range(ind_to_move_per_step): # Move the same steps as psuedo_traj_controller + A*
                        next_pose = gradient_planner(cur_pose, cost_transform_map=cost_transform_map) 
                        cur_pose = next_pose
                else:
                    raise ValueError("Invalid local planner: {}".format(chosen_local_planner))
                
                if collect_opts.log_iou: # Toggle: Do real-time iou check
                    _, _, _, _, lama_pred_alltrain_viz_t = \
                            get_lama_pred_from_obs(mapper.obs_map.copy(), lama_alltrain_model, lama_map_transform, device)

                    iou = calculate_iou_kth(lama_pred_alltrain_viz_t[500:-500, 500:-500, :], 
                                            padded_gt_map[500:-500, 500:-500], 
                                            show_plt=False, 
                                            file_name=file_name)

                    # Log every 20 timesteps
                    if t % 20 == 0:
                        log_file.write(f"Time step is {t} and IoU is {iou}\n")
                        log_file.flush()
                    
                    if not iou_90_flag and iou > 0.8999:
                        log_file.write(f"IoU 90% reached : Time step is {t} and IoU is {iou}\n")
                        log_file.flush()
                        iou_90_flag = True 

                    if not iou_95_flag and iou > 0.9499:
                        log_file.write(f"IoU 95% reached : Time step is {t} and IoU is {iou}\n")
                        log_file.flush()
                        iou_95_flag = True 
                    

                    if iou_95_flag and t > (time_step // 3):
                        update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=True,
                                            fail_reason="", mission_status_save_path=mission_status_save_path)
                        print("\033[94mMission complete for {}!\033[0m".format(exp_title))
                        break
     

                # Visualize
                start_time = time.time()
                if show_plt: 
                    for a in ax.flatten():
                        a.clear()

                    # make a general map kwarg that is gray cmap
                    map_kwargs = {
                        'cmap': 'gray',
                        'vmin': 0,
                        'vmax': 1,
                    }
                    ax_gt.imshow(1-mapper.gt_map[pd_size:-pd_size,pd_size:-pd_size], **map_kwargs)
                    if pose_list is not None:
                        ax_gt.plot(pose_list[:, 1]-pd_size, pose_list[:, 0]-pd_size, c='r', alpha=0.5)
                        ax_gt.scatter(pose_list[-1, 1]-pd_size, pose_list[-1, 0]-pd_size, c='g', s=10, marker='*')
                    if mode != 'upen':
                        # if frontier cost list is all zeros, make it red
                        if frontier_cost_list is not None and np.all(frontier_cost_list == 0):
                            frontier_colors = 'r'
                        else:
                            frontier_colors = -frontier_cost_list
                            
                        if mode == 'pipe' or mode == 'pw-nbv-2d':
                            # Check the length of frontier_colors to avoid accessing out-of-bounds elements
                            if len(frontier_colors) > 0:
                                # Plot the best frontier with size 20 and its color
                                ax_gt.scatter(locked_frontier_center[1] - pd_size,
                                            locked_frontier_center[0] - pd_size,
                                            c='r', s=20, marker='x', cmap='plasma')

                            if len(frontier_colors) > 1:
                                # Plot the second-best frontier with size 10 and its color (only if available)
                                ax_gt.scatter(np.array(frontier_region_centers)[medium_ind, 1] - pd_size,
                                            np.array(frontier_region_centers)[medium_ind, 0] - pd_size,
                                            c='b', s=10, marker='x', cmap='plasma')

                            # Plot the remaining frontiers with size 5 and a neutral color (e.g., gray)
                            remaining_indices = [i for i in range(len(frontier_region_centers)) if i not in [best_ind, medium_ind]]

                            if len(remaining_indices) > 0:
                                ax_gt.scatter(np.array(frontier_region_centers)[remaining_indices, 1] - pd_size,
                                            np.array(frontier_region_centers)[remaining_indices, 0] - pd_size,
                                            c='gray', s=5, marker='x')
                        else:
                            ax_gt.scatter(np.array(frontier_region_centers)[:, 1]-pd_size, np.array(frontier_region_centers)[:, 0]-pd_size, c=frontier_colors, s=10, marker='x',cmap='plasma')
                    ax_gt.set_title('GT Map')
                    # print("3a. Visualizing GT Map took {} seconds".format(np.round(time.time() - start_time, 2)))

                    #colors_ = ["#68e371", "#D9D9D9", "#f0432b"]
                    colors_ = ["#FFFFFF", "#D9D9D9", "#000000"]
                    cmap = ListedColormap(colors_)
                    ax_obs.imshow(mapper.obs_map[pd_size:-pd_size,pd_size:-pd_size], cmap = cmap)#, **map_kwargs)

                    

                    if pose_list is not None:
                        ax_obs.plot(pose_list[:, 1]-pd_size, pose_list[:, 0]-pd_size, c='#eb4205', alpha=1.0)
                    if mode not in ['upen', 'hector', 'hectoraug']: # UPEN and Hector do not have locked frontiers
                        ax_obs.scatter(locked_frontier_center[1]-pd_size, locked_frontier_center[0]-pd_size, c='#eb4205', s=10)
                    if mode not in ['hector', 'hectoraug']: # Hector does not have path planning
                        ax_obs.plot(plan_y-pd_size, plan_x-pd_size,c='#eb4205', linestyle=':')

                    if mode not in ['upen', 'hector', 'hectoraug']: # UPEN and Hector are not a frontier planner
                        if viz_medium_flooded_grid is not None:
                            second_most_flooded_grid = viz_medium_flooded_grid[pd_size:-pd_size,pd_size:-pd_size]
                            second_flooded_ind = np.where(second_most_flooded_grid==True)
                            #ax_obs.scatter(second_flooded_ind[1]-pad_w1,second_flooded_ind[0]-pad_h1,c='c',s=1,alpha=0.05)

                        if viz_most_flooded_grid is not None:
                            most_flooded_grid = viz_most_flooded_grid#[pd_size:-pd_size,pd_size:-pd_size]
                            flooded_ind = np.where(most_flooded_grid==True)
                            flooded_ind_colors_alpha = np.zeros((mapper.obs_map.shape[0],mapper.obs_map.shape[1],4))
                            flooded_ind_colors_alpha[flooded_ind[0],flooded_ind[1],:] = (255/255,159/255,28/255,0.3)    # orange
                            #flooded_ind_colors_alpha[flooded_ind[0],flooded_ind[1],:] = (207/255,3/255,252/255,0.25)   # purple                            
                            ###ax_obs.scatter(flooded_ind[1]-pad_w1, flooded_ind[0]-pad_h1,c="#FF9F1C",s=1,alpha=0.05)
                            if mode in ['nbv-2d', 'pw-nbv-2d']:  # Show flooded grid for nbv-2d and pw-nbv-2d modes
                                ax_obs.imshow(flooded_ind_colors_alpha[pd_size+pad_h1:-(pd_size+pad_h2),pd_size+pad_w1:-(pd_size+pad_w2)])

                    ax_obs.set_title('Observed Map')

                    if pred_maputils is not None:
                        white = "#FFFFFF"
                        blue = "#0000FF"
                        colors = [white, blue]
                        n_bins = 10
                        cmap = LinearSegmentedColormap.from_list("customwhiteblue", colors, N=n_bins)
                        
                        if mode not in ['hectoraug']: # Hector aug does not have variance
                            orange = "#FF9F1C"
                            colors = [white, orange]
                            n_bins = 10
                            var_cmap = LinearSegmentedColormap.from_list("custom_green_var", colors, N=n_bins)
                            ax_pred_var.imshow(var_map[pd_size+pad_w1:-(pd_size+pad_w2),pd_size+pad_h1:-(pd_size+pad_h2)].cpu().numpy(), vmin=0, vmax=0.3, cmap=var_cmap)
                            ax_pred_var.set_title('Predicted Map Variance')
                            
                            # for model_i in range(len(models_list)):
                            #     ax_pred_lama_list[model_i].imshow(lama_pred_list[model_i].cpu().numpy())
                            #     ax_pred_lama_list[model_i].set_title('LAMA {}'.format(model_i))                
                            #ax_reduced_var.imshow(lama_reduced_pred_var[pd_size+pad_w1:-(pd_size+pad_w2),pd_size+pad_h1:-(pd_size+pad_h2)].cpu().numpy(), vmin=0, vmax=0.3)
                            #ax_reduced_var.set_title('LAMA Variance (reduced)')

                            if mean_map is not None:
                                white = "#FFFFFF"
                                #black = "#000000"
                                #green = "#b0f542"
                                #red = "#ed0c0c"
                                blue = "#0000FF"
                                orange = "#FF9F1C"
                                if mode == 'pipe':
                                    colors = [white, orange]
                                else:
                                    colors = [white, blue]
                                n_bins = 100
                                cmap = LinearSegmentedColormap.from_list("customgreenred", colors, N=n_bins)
                                ax_mean_map.imshow(mean_map[pd_size:-(pd_size),pd_size:-(pd_size)],cmap=cmap) #predicted map

                                #overlay observed(known) occupied cells on top of the predicted map
                                obs_occ_mask = np.zeros_like(mean_map[pd_size:-(pd_size),pd_size:-(pd_size)])
                                occupied_indices_in_obsmap = np.where(mapper.obs_map[pd_size-pad_h1:-(pd_size-pad_h2),pd_size-pad_w1:-(pd_size-pad_w2)] == 1.0) #indices where obs_map is occupied
                                obs_occ_mask[occupied_indices_in_obsmap] = 1
                                obs_occ_mask_colors = ["#000000","#000000"]
                                obs_occ_mask_cmap = LinearSegmentedColormap.from_list("mask_black",obs_occ_mask_colors,N=2)
                                obs_occ_mask_alpha = np.zeros_like(obs_occ_mask, dtype=float)
                                obs_occ_mask_alpha[obs_occ_mask==1] = 1.0
                                obs_occ_mask_alpha[obs_occ_mask==0] = 0.0
                                ax_mean_map.imshow(obs_occ_mask, cmap=obs_occ_mask_cmap, alpha=obs_occ_mask_alpha) #obs_map known occ cells -> black

                                #overlay unknown(obs_map) cells as gray tint
                                obs_unk_mask = np.zeros_like(mean_map[pd_size:-pd_size,pd_size:-pd_size])
                                unknown_indices_in_obs_map = np.where(mapper.obs_map[pd_size-pad_h1:-(pd_size-pad_h2),pd_size-pad_w1:-(pd_size-pad_w2)] == 0.5) #indices of unknown cells in obs_map
                                obs_unk_mask[unknown_indices_in_obs_map] = 1
                                grey = "#909090" #tunable grey value
                                obs_unk_mask_colors = [grey,grey]
                                obs_unk_mask_cmap = LinearSegmentedColormap.from_list("mask_grey", obs_unk_mask_colors, N=2)
                                obs_unk_mask_alpha = np.zeros_like(obs_unk_mask, dtype=float)
                                obs_unk_mask_alpha[obs_unk_mask==1] = 0.3 #tunable opacity for grey unknown area
                                obs_unk_mask_alpha[obs_unk_mask==0] = 0.0
                                ax_mean_map.imshow(obs_unk_mask, cmap=obs_unk_mask_cmap, alpha=obs_unk_mask_alpha)
                                
                                #path_color = "#417CF2" #blue
                                path_color = "#eb4205" #coral(red)
                                if pose_list is not None:
                                    ax_mean_map.plot(pose_list[:, 1]-(pd_size-pad_w1), pose_list[:, 0]-(pd_size-pad_h1), c=path_color, alpha=1.0)
                                    #ax_mean_map.scatter(pose_list[-1, 1]-(pd_size-pad_w1), pose_list[-1, 0]-(pd_size-pad_h1), c='g', s=10, marker='*')
                                if mode not in ['upen', 'hector', 'hectoraug']: # UPEN and Hector do not have locked frontiers
                                    ax_mean_map.scatter(locked_frontier_center[1]-(pd_size-pad_w1), locked_frontier_center[0]-(pd_size-pad_h1), c='#eb4205', s=10)
                                #ax_mean_map.scatter(cur_pose[1]-(pd_size-pad_w1), cur_pose[0]-(pd_size-pad_h1), c='r', s=5, marker='x')
                                #ax_mean_map.scatter(next_pose[1]-(pd_size-pad_w1), next_pose[0]-(pd_size-pad_h1), c='g', s=5, marker='x')
                                if mode not in ['hector', 'hectoraug']: # Hector does not have path planning
                                    ax_mean_map.plot(plan_y-(pd_size-pad_w1), plan_x-(pd_size-pad_h1), c='#eb4205', linestyle=':')
                                ax_mean_map.set_title('Mean Map of Prediction Ensembles')

                                #visualize frontiers on the mean_map
                                #if frontier_cost_list is not None and np.all(frontier_cost_list == 0):
                                #    frontier_colors = 'r'
                                #else:
                                #    frontier_colors = -frontier_cost_list
                                #ax_mean_map.scatter(np.array(frontier_region_centers)[:, 1]-(pd_size-pad_w1), np.array(frontier_region_centers)[:, 0]-(pd_size-pad_h1), c=frontier_colors, s=15, marker='x',cmap='plasma')

                                if mode != 'upen':
                                    if viz_most_flooded_grid is not None:
                                        most_flooded_grid = viz_most_flooded_grid#[pd_size:-pd_size,pd_size:-pd_size]
                                        flooded_ind = np.where(most_flooded_grid==True)
                                        flooded_ind_colors_alpha = np.zeros((mean_map.shape[0],mean_map.shape[1],4))
                                        if mode == 'pipe':
                                            flooded_ind_colors_alpha[flooded_ind[0],flooded_ind[1],:] = (0, 0, 1, 0.15) #blue color for visibility mask
                                        else:
                                            flooded_ind_colors_alpha[flooded_ind[0],flooded_ind[1],:] = (255/255,159/255,28/255,0.3) #orange color for visibility mask
                                        ax_mean_map.imshow(flooded_ind_colors_alpha[pd_size:-pd_size,pd_size:-pd_size])
                                    if viz_medium_flooded_grid is not None:
                                        medium_flooded_ind = np.where(viz_medium_flooded_grid == True)
                                        medium_flooded_ind_colors_alpha = np.zeros((mean_map.shape[0],mean_map.shape[1],4))
                                        #medium_flooded_ind_colors_alpha[medium_flooded_ind[0], medium_flooded_ind[1],:] = (154/255,230/255,72/255,0.4)
                                        medium_flooded_ind_colors_alpha[medium_flooded_ind[0], medium_flooded_ind[1],:] = (207/255,3/255,252/255,0.25)
                                        #ax_mean_map.imshow(medium_flooded_ind_colors_alpha[pd_size:-pd_size,pd_size:-pd_size])
                                        #if medium_ind is not None:
                                        #    ax_mean_map.scatter(frontier_region_centers[medium_ind,1]-(pd_size-pad_w1), frontier_region_centers[medium_ind,0]-(pd_size-pad_h1), c='#390ccc',s=10)

                    if mode in ['hector', 'hectoraug']:
                        ax_mean_map.imshow(cost_transform_map[pd_size:-pd_size,pd_size:-pd_size], cmap='gray')
                        ax_mean_map.set_title('Cost Transform Map')
                    # # Display the flood fill
                    # if viz_min_flooded_grid is not None:
                    #     ax_min_flooded_grid.imshow(viz_min_flooded_grid)
                    #     ax_min_flooded_grid.set_title('Min Frontier Val')
                    # if viz_most_flooded_grid is not None:
                    #     ax_max_flooded_grid.imshow(viz_most_flooded_grid)
                    #     ax_max_flooded_grid.set_title('Most Frontier Val')

                    plt.tight_layout()
                    # plt.show()
                    # plt.pause(0.001)
                    
                    # fig name include experiment title and time step
                    print("saving fig:", t)
                    plt.savefig(run_viz_dir + '/{}_{}.png'.format(exp_title, str(t).zfill(8)),dpi=100)
                    # print("3d. Visualizing took {} seconds".format(np.round(time.time() - start_time, 2)))
                    # import pdb; pdb.set_trace() 
                    # print("4. Visualizing took {} seconds".format(np.round(time.time() - start_time, 2)))
                    #plt.close()

                    
                    show_plt = False

                # Save obs map and pose list 
                # Save obs map 
                cv2.imwrite(global_obs_dir + '/{}.png'.format(str(t).zfill(8)), smu.convert_01_single_channel_to_0_255_3_channel(mapper.obs_map))
                # Save pose list
                np.save(odom_npy_save_path, pose_list)
                
                # # Go to next pose
                # start_time = time.time()
                cur_pose = next_pose
                if mapper.gt_map[cur_pose[0], cur_pose[1]] == 1:
                    print("Hit wall!")
                    mission_failed = True
                    update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, fail_reason="hit_wall", mission_status_save_path=mission_status_save_path)
                    break
                pose_list = np.concatenate([pose_list, np.atleast_2d(cur_pose)], axis=0)
                # # Observation: Get instant observation and accumulate
                mapper.observe_and_accumulate_given_pose(cur_pose)
                # print("5. Accumulating obs took {} seconds".format(np.round(time.time() - start_time, 2)))

                # Save current mission status 
                update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, fail_reason="", mission_status_save_path=mission_status_save_path)
                # if (t % 99) == 0:
                #print("Total time for step {} is {} seconds".format(t, np.round(time.time() - start_mission_i_time, 2)))
                t += 1

            #TODO:  Save final mission status 
            if mission_failed:
                # Don't update mission status if it's already failed
                print("\033[91mMission failed for {}!\033[0m".format(exp_title))

            else:
                update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=True, fail_reason="", mission_status_save_path=mission_status_save_path)
                print("\033[94mMission complete for {}!\033[0m".format(exp_title))
    except Exception as e:
        print("\033[93mMission failed with exception for {}!\033[0m".format(exp_title))
        print(e)
        # print the exception and line number 
        print(traceback.format_exc())
        update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, fail_reason=str(e), mission_status_save_path=mission_status_save_path)
    
def determine_use_model(mode):
    possible_mode_list = 'nearest', 'nbv-2d', 'onlyvar', 'visunk', 'visvar', 'mapex', 'upen', 'hectoraug', 'pipe', 'pw-nbv-2d'
    assert mode in possible_mode_list, "mode must be one of {}".format(possible_mode_list)
    if mode == 'onlyvar' or mode == 'visunk' or mode == 'visvar' or mode == 'mapex' or mode == 'pipe':
        use_model = True 
    elif mode == 'nearest' or mode == 'nbv-2d'  or mode == 'pw-nbv-2d': 
        use_model = False
    elif mode == 'hectoraug':
        use_model = True
    elif mode == 'upen':
        use_model = True
    else:
        raise NotImplementedError
    return use_model


if __name__ == '__main__':
    data_collect_config_name = 'base.yaml' #customize yaml file, as needed
    today = datetime.today()
    output_subdirectory_name = str(today.year)+"{:02d}".format(today.month)+"{:02d}".format(today.day)+'_test'
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_world_list', nargs='+', help='List of worlds to collect data from')
    parser.add_argument('--start_pose', nargs='+', help='List of start pose')
    args = parser.parse_args()
    
    # Auto-detect root path (PIPE directory is parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(script_dir)
    
    collect_opts = get_options_dict_from_yml(data_collect_config_name)
    collect_opts.root_path = root_path  # Set root_path dynamically
    if args.collect_world_list is not None:
        collect_opts.collect_world_list = args.collect_world_list
    if args.start_pose is not None: 
        start_pose = []
        for pose_elem in args.start_pose:
            start_pose.append(int(pose_elem))
        collect_opts.start_pose = start_pose

    kth_map_folder_path = os.path.join(root_path, 'kth_test_maps')
    kth_map_paths = os.listdir(kth_map_folder_path)

    if collect_opts.collect_world_list is not None:
        kth_map_paths_collect = []
        for folder_name in kth_map_paths:
            if folder_name in collect_opts.collect_world_list:
                kth_map_paths_collect.append(folder_name)
        kth_map_paths = kth_map_paths_collect

    kth_map_folder_paths = [os.path.join(kth_map_folder_path, p) for p in kth_map_paths] * collect_opts.num_data_per_world

    # Make output_subdirectory_name if it doesn't exist 
    output_root_dir = os.path.join(root_path, collect_opts.output_folder_name, output_subdirectory_name)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    
    model_list = []
    device = collect_opts.lama_device

    #ensemble of lama models (G_i; G_1, G_2, G_3 in the paper) - fine-tuned with split, smaller training sets
    if collect_opts.ensemble_folder_name is not None:
        ensemble_folder_name = collect_opts.ensemble_folder_name
        ensemble_folder_path = os.path.join(root_path, 'pretrained_models', ensemble_folder_name)
        ensemble_model_dirs = sorted(os.listdir(ensemble_folder_path))
        for ensemble_model_dir in ensemble_model_dirs:
            ensemble_model_path = os.path.join(ensemble_folder_path, ensemble_model_dir)
            model = load_lama_model(ensemble_model_path, device=collect_opts.lama_device)
            print("Loaded model: ", ensemble_model_dir)
            model_list.append(model)
    
    #setup a big lama model (G in the paper) - fine-tuned with the entire training set
    big_lama_path = os.path.join(root_path, 'pretrained_models', collect_opts.big_lama_model_folder_name)
    lama_model = load_lama_model(big_lama_path, device=collect_opts.lama_device)
    lama_map_transform = get_lama_transform(collect_opts.lama_transform_variant, collect_opts.lama_out_size)

    run_exploration_args = []
    for kth_map_folder_path in kth_map_folder_paths:
        args_dict = {
            'map_folder_path': kth_map_folder_path,
            'models_list': model_list,
            'lama_model': lama_model,
            'lama_map_transform': lama_map_transform,
            'pred_vis_configs': collect_opts.pred_vis_configs,
            'lidar_sim_configs': collect_opts.lidar_sim_configs,
            'start_pose': collect_opts.start_pose,
            'modes_to_test': collect_opts.modes_to_test,
            'unknown_as_occ': collect_opts.unknown_as_occ,
            'use_distance_transform_for_planning': collect_opts.use_distance_transform_for_planning,
            'upen_config': collect_opts.upen_config
        }
        run_exploration_args.append(args_dict)
    
    for run_exploration_arg in run_exploration_args:
        run_exploration_comparison_for_map(run_exploration_arg)

