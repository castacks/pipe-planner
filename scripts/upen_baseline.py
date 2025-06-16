### Script to prototype UPEN baseline exploration method with same simulator
import os 
import time 
import pyastar2d
import numpy as np 
from matplotlib import pyplot as plt
import time 
import torch
import hydra
from omegaconf import OmegaConf
import scripts.sim_utils as sim_utils
from rrt_star import RRTStar

def get_options_dict_from_yml(config_name):
    cwd = os.getcwd()
    hydra_config_dir_path = os.path.join(cwd, '../configs')
    print(hydra_config_dir_path)
    with hydra.initialize_config_dir(config_dir=hydra_config_dir_path):
        cfg = hydra.compose(config_name=config_name)
    options_dict = OmegaConf.to_container(cfg)
    options = OmegaConf.create(options_dict)
    return options

def eval_path_expl(ensemble, paths, reach_horizon):
    # evaluate each path based on its average occupancy uncertainty
    #N, B, C, H, W = ensemble.shape # number of models, batch, classes, height, width
    ### Estimate the variance only of the occupied class (1) for each location # 1 x B x object_classes x grid_dim x grid_dim
    ensemble_occupancy_var = torch.var(ensemble[:,:,1,:,:], dim=0, keepdim=True).squeeze(0) # 1 x H x W
    path_sum_var = []
    for k in range(len(paths)):
        path = paths[k]
        path_var = []
        for idx in range(min(reach_horizon,len(path))-1):
            node1 = path[idx]
            node2 = path[idx+1]
            maxdist = max(abs(node1[0]-node2[0]), abs(node1[1]-node2[1])) +1
            xs = np.linspace(int(node1[0]), int(node1[0]), int(maxdist))
            ys = np.linspace(int(node1[1]), int(node2[1]), int(maxdist))
            for i in range(len(xs)):
                x = int(xs[i])
                y = int(ys[i])      
                # if ensemble_occupancy_var[0,x,y] > 0:
                #     print(x,y, ensemble_occupancy_var[0,x,y])    
                path_var.append(ensemble_occupancy_var[0,x,y])
        path_sum_var.append( np.sum(np.asarray(path_var)) )
    return path_sum_var


def get_rrt_goal(pose_coords, goal, grid, ensemble, rrt_max_iters, expand_dis, goal_sample_rate, connect_circle_dist, rrt_num_path, rrt_straight_line, reach_horizon, upen_mode):
    # probability_map, indexes = torch.max(grid,dim=1)
    # probability_map = probability_map[0]
    # indexes = indexes[0]
    # import pdb; pdb.set_trace()
    # binarymap = (indexes == 1)
    # assert it's a 2D map
    assert len(grid.shape) == 2, "Grid should be 2D, but got shape: {}".format(grid.shape)
    binarymap = (grid == 1)
    start = [int(pose_coords[0][0][1]), int(pose_coords[0][0][0])]
    finish = [int(goal[0][0][1]), int(goal[0][0][0])]
    rrt_star = RRTStar(start=start, 
                        obstacle_list=None, 
                        goal=finish, 
                        rand_area=[0,binarymap.shape[0]], 
                        max_iter=rrt_max_iters,
                        expand_dis=expand_dis,
                        goal_sample_rate=goal_sample_rate,
                        connect_circle_dist=connect_circle_dist,
                        occupancy_map=binarymap)
    best_path = None
    
    path_dict = {'paths':[], 'value':[]} # visualizing all the paths
    #TODO: Implement exploration objective
    if upen_mode == 'exploration':
        exploration_mode = True
        paths = rrt_star.planning(animation=False, use_straight_line=rrt_straight_line, exploration=exploration_mode, horizon=reach_horizon)
        ## evaluate each path on the exploration objective
        path_sum_var = eval_path_expl(ensemble, paths, reach_horizon)
        path_dict['paths'] = paths
        path_dict['value'] = path_sum_var

        best_path_var = 0 # we need to select the path with maximum overall uncertainty
        for i in range(len(paths)):
            if path_sum_var[i] > best_path_var:
                best_path_var = path_sum_var[i]
                best_path = paths[i]

    elif upen_mode == 'pathplan':
        best_path_reachability = float('inf')   
        for i in range(rrt_num_path):
            path = rrt_star.planning(animation=False, use_straight_line=rrt_straight_line)
            
            if path:
                # TODO: Implement reachability metric
                # if self.options.rrt_path_metric == "reachability":
                #     reachability = self.eval_path(ensemble, path, prev_path)
                # elif self.options.rrt_path_metric == "shortest":
                reachability = len(path)
                path_dict['paths'].append(path)
                path_dict['value'].append(reachability)
                
                if reachability < best_path_reachability:
                    best_path_reachability = reachability
                    best_path = path
    else:
        raise NotImplementedError("Mode {} not implemented".format(upen_mode))
    
    if best_path:
        best_path.reverse()
        last_node = min(len(best_path)-1, reach_horizon)
        return torch.tensor([[[int(best_path[last_node][1]), int(best_path[last_node][0])]]]).cuda(), best_path, path_dict
    
    return None, None, None

if __name__ == '__main__':
    data_collect_config_name = 'cherie-perceptron-path-planning.yaml'

    collect_opts = get_options_dict_from_yml(data_collect_config_name)

    toolbox_path = collect_opts.map_prediction_toolbox_path
    #change the map_folder_dir_path; this is a path that stores all the directories of map environments with global_obs, global_viz, gt_map, odom.npy (1st robot)
    map_folder_dir_path = collect_opts.map_folder_dir_path
    #change the processed_map_folder_path; this is a path that stores all the directories of map environments with occ_map and valid_space map in npy file format
    processed_map_folder_path = os.path.join(toolbox_path, 'data_factory/processed_kth_maps/')
    lidar_sim_configs = collect_opts.lidar_sim_configs

    # Define variables
    mission_time = 1000
    map_id = '50052750'
    mode = 'exploration'
    rrt_max_iters = 2500 # UPEN: 2500
    expand_dis = 5 # UPEN: 5 # expand distance, essentially number of pixels between two nodes in RRT TODO: check resolution
    rrt_num_path = 100
    rrt_straight_line = False
    reach_horizon = 50 # UPEN: 10 
    connect_circle_dist = 20 # UPEN: 20
    start_pose = np.array([515, 700])
    goal_pose_freq = 10
    goal_dist_thresh = 2
    dist_to_intermediate_goal = 1000
    if mode == 'pathplan':
        # end_pose = np.array([515, 700])
        end_pose = np.array([800, 583])
        goal_sample_rate = 20
    elif mode == 'exploration':
        # when exploration mode, use a dummy unreachable goal
        end_pose = np.array([1000, 1000])
        # ensure that the goal sampling rate is below 0
        goal_sample_rate = -1
    else:
        raise NotImplementedError("Mode {} not implemented".format(mode))
    show_plt = True
    run_viz_dir = os.path.join(toolbox_path, 'experiments', 'upen_baseline')
    os.makedirs(run_viz_dir, exist_ok=True)
    exp_title = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Get exp directory 
    run_viz_dir_exp_title = os.path.join(run_viz_dir, exp_title)
    os.makedirs(run_viz_dir_exp_title, exist_ok=False)
    # Get Occ Map
    #retrieve occ_map and valid_space map from the directories
    map_occ_npy_path = os.path.join(processed_map_folder_path, map_id, 'occ_map.npy')
    map_valid_space_npy_path = os.path.join(processed_map_folder_path, map_id, 'valid_space.npy')
    occ_map, validspace_map = sim_utils.get_kth_occ_validspace_map(map_occ_npy_path, map_valid_space_npy_path)


    model_list = []
    device = collect_opts.lama_device


    # Start Path Planning Experiment    
    mapper = sim_utils.Mapper(occ_map, 
                        lidar_sim_configs, 
                        use_distance_transform_for_planning=collect_opts.use_distance_transform_for_planning)
    
    cur_pose = start_pose
    online_pose_list = np.atleast_2d(cur_pose)
    intermediate_goal_pose = None
    
    for t in range(mission_time):
        start_mission_i_time = time.time()
        occ_grid_pyastar = mapper.get_inflated_planning_maps()
        

        if intermediate_goal_pose is not None:
            dist_to_intermediate_goal = np.linalg.norm(cur_pose - intermediate_goal_pose)
            # Doing A* planning to intermediate goal the first time to check if it's reachable
            path = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, intermediate_goal_pose, allow_diagonal=False)
        if t % goal_pose_freq == 0 or dist_to_intermediate_goal < goal_dist_thresh or path is None:
            path = None # Reset path
            while path is None: # Get a new intermediate goal, until there's a valid one
                # Use RRT to get intermediate goal position
                pose_coords = torch.tensor([[[cur_pose[1], cur_pose[0]]]]).cuda() # TODO: find it why the x and y are flipped
                goal_pose_coords = torch.tensor([[[end_pose[1], end_pose[0]]]]).cuda() # TODO: find it why the x and y are flipped
                # import pdb; pdb.set_trace()
                planning_grid = torch.tensor(mapper.obs_map).cuda()
                # RRT Params from UPEN: https://github.com/ggeorgak11/UPEN/blob/master/train_options.py#L4

                
                # TODO: Add ensemble output to get_rrt_goal
                # Ensemble: N x B x C x H x W. Current planning_grid.shape = H x W [1326, 1428]
                # ! MOCK ensemble that is N x 1 x 2 x 1326 x 1428, with top right corner as high variance 
                mock_ensemble_N = 2
                ensemble = torch.zeros((mock_ensemble_N, 1, 2, planning_grid.shape[0], planning_grid.shape[1])).cuda()
                ensemble[0, 0, 1, 500:700, 800:900] = 1 # right
                # ensemble[0, 0, 1, 500:700, 500:600] = 1 # left
                # TODO: ensemble should be None for path plan
                
                rrt_goal, rrt_best_path, path_dict = get_rrt_goal(pose_coords=pose_coords.clone(), goal=goal_pose_coords.clone(), grid=planning_grid, ensemble=ensemble, \
                    rrt_max_iters=rrt_max_iters, expand_dis=expand_dis, goal_sample_rate=goal_sample_rate, connect_circle_dist=connect_circle_dist, rrt_num_path=rrt_num_path, \
                        rrt_straight_line=rrt_straight_line, reach_horizon=reach_horizon, \
                            upen_mode=mode)
                
                # Given path_dict, visualize the paths
                if rrt_goal is not None:
                    intermediate_goal_pose = rrt_goal[0][0].cpu().numpy()
                    intermediate_goal_pose = intermediate_goal_pose[::-1] # TODO: find it why the x and y are flipped
                else: # Use A* to get intermediate goal
                    print("No intermediate goal found, using A* to get intermediate goal")
                    astar_path_to_goal = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, end_pose, allow_diagonal=False)
                    intermediate_goal_pose = astar_path_to_goal[np.min([20, len(astar_path_to_goal)-1])]
                plt_row = 1
                plt_col = 6
                fig, ax = plt.subplots(plt_row, plt_col, figsize=(12, 4))
                plt.subplot(plt_row, plt_col, 1)
                plt.imshow(mapper.obs_map, cmap='gray')
                
                if path_dict is not None:
                    for path in path_dict['paths']:
                        path = np.array(path)
                        plt.plot(path[:,1], path[:,0], 'b-')
                
                    rrt_best_path = np.array(rrt_best_path)
                    plt.plot(rrt_best_path[:,1], rrt_best_path[:,0], 'r-')
                plt.scatter(intermediate_goal_pose[1], intermediate_goal_pose[0], c='green', marker='x', s=20)
                plt.subplot(plt_row, plt_col, 2)
                plt.imshow(occ_map, cmap='gray')
                
                if path_dict is not None:
                    for path in path_dict['paths']:
                        path = np.array(path)
                        plt.plot(path[:,1], path[:,0], 'b-')
                    rrt_best_path = np.array(rrt_best_path)
                    plt.plot(rrt_best_path[:,1], rrt_best_path[:,0], 'r-')
                
                plt.subplot(plt_row, plt_col, 3)
                plt.imshow(mapper.obs_map[500:-500,500:-500], cmap='gray')
                plt.title('Cropped')
                
                plt.subplot(plt_row, plt_col, 4)
                plt.imshow(mapper.obs_map[500:-500,500:-500], cmap='gray')
                plt.title('Cropped')
                if path_dict is not None:
                    plt.plot(rrt_best_path[:,1]-500, rrt_best_path[:,0]-500, 'r-')
                
                plt.subplot(plt_row, plt_col, 5)
                plt.imshow(ensemble[0,0,1,:,:][500:-500,500:-500].cpu().numpy(), cmap='gray')
                plt.title('Cropped Ens.[0]')
                
                plt.subplot(plt_row, plt_col, 6)
                plt.imshow(ensemble[1,0,1,:,:][500:-500,500:-500].cpu().numpy(), cmap='gray')
                plt.title('Cropped Ens.[1]')
                
                plt.savefig(run_viz_dir_exp_title + '/rrt_paths_{}.png'.format(str(t).zfill(8)))
                plt.clf()
                plt.close(fig)
                # import pdb; pdb.set_trace()
                path = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, intermediate_goal_pose, allow_diagonal=False)
                
        # Doing A* planning to intermediate goal the second time to get path
        path = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, intermediate_goal_pose, allow_diagonal=False) 
        plan_x = path[:,0]
        plan_y = path[:,1]

        # psuedo-trajectory controller (pick the third point in the plan, around 3-5 pixels from current)
        plan_ind_to_use = np.min([3, len(plan_x)-1]) 
        next_pose = np.array([plan_x[plan_ind_to_use], plan_y[plan_ind_to_use]]).astype(int)
        #bp()
        # # Go to next pose
        print("Step {}".format(t))
        print("next - cur pose: ", np.sqrt((next_pose[1]-cur_pose[1])**2+(next_pose[0]-cur_pose[0])**2))
        cur_pose = next_pose
        
        if mapper.gt_map[cur_pose[0], cur_pose[1]] == 1:
           next_pose = np.array([plan_x[1],plan_y[1]]).astype(int)
           cur_pose = next_pose
           if mapper.gt_map[cur_pose[0],cur_pose[1]] == 1:
                print("Hit wall!")
                break

        online_pose_list = np.concatenate([online_pose_list, np.atleast_2d(cur_pose)], axis=0)
        mapper.observe_and_accumulate_given_pose(cur_pose)
        if np.linalg.norm(cur_pose - end_pose) < 5:
            distance = np.sum(np.linalg.norm(online_pose_list[1:] - online_pose_list[:-1], axis=1))
            break
        if t % 100 == 0:
            print("Total time for step {} is {} seconds".format(t, np.round(time.time() - start_mission_i_time, 2)))            
        if show_plt:
            #Visualization Setup 
            plt_row = 1
            plt_col = 2
            fig, ax = plt.subplots(plt_row, plt_col, figsize=(12, 16))
            ax_gt = ax[0]
            ax_obs = ax[1]
            print()
            for a in ax.flatten():
                a.clear()
            
            map_kwargs = {
                'cmap': 'gray',
                'vmin': 0,
                'vmax': 1,
            }
            # TODO: parametrize padding
            ax_gt.imshow(mapper.gt_map[500:-500,500:-500], **map_kwargs)
            ax_gt.scatter(start_pose[1]-500, start_pose[0]-500, c='yellow', marker='x', s=20)  # Start pose in red
            ax_gt.scatter(end_pose[1]-500, end_pose[0]-500, c='cyan', marker='D', s=20)
            # for r1_i in range(len(r1_pose_list)): #visualize robot1's trajectory
            #     ax_gt.scatter(r1_pose_list[r1_i,1], r1_pose_list[r1_i,0], c='red', marker='.',s=10)
            ax_gt.scatter(online_pose_list[:,1]-500, online_pose_list[:,0]-500, c='yellow', marker='.', s=10, label='online_pose_list')
            ax_gt.plot(plan_y-500, plan_x-500, 'r-', linewidth=2, label='A* Plan')
            ax_gt.scatter(intermediate_goal_pose[1]-500, intermediate_goal_pose[0]-500, c='green', marker='x', s=20, label='Intermediate Goal Pose')
            ax_gt.set_title('GT Map')
            ax_gt.legend()

            ax_obs.imshow(mapper.obs_map[500:-500,500:-500], **map_kwargs)
            ax_obs.set_title('observed map')
            

            plt.savefig(run_viz_dir_exp_title + '/{}.png'.format(str(t).zfill(8)))
            # plt.clf()
            plt.close(fig)
