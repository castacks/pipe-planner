import numpy as np
import cv2
from tqdm.contrib.concurrent import process_map  # or thread_map
from skimage.measure import block_reduce
from scipy.ndimage import convolve, label, generate_binary_structure
from scipy.ndimage import binary_dilation
import scipy
import torch
import range_libc
import multiprocessing as mp
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# custom imports
import sys
sys.path.append('../')
from scripts import simple_mask_utils as smu

def makePyOMap(occ_grid):
    return range_libc.PyOMap(occ_grid)

def psuedo_traj_controller(plan_x, plan_y, plan_ind_to_use):
    """
    Given a plan, return the next pose to go to.
    """
    plan_ind_to_use = np.min([plan_ind_to_use, len(plan_x)-1]) # psuedo-trajectory controller
    next_pose = np.array([plan_x[plan_ind_to_use], plan_y[plan_ind_to_use]]).astype(int)
    return next_pose

def get_kth_occ_validspace_map(occ_npy_path, validspace_npy_path):
    block_size_pix = 2
    # Load npy path which has value either 0: occupied and 254: free 
    # and Convert to occupancy map (0: unknown, 1: occupied, 2: free)
    occ_npy = np.load(occ_npy_path)
    assert np.array_equal(np.unique(occ_npy), [0, 254]), "unique values in occ_npy should be 0 and 254"
    occ_map = np.zeros_like(occ_npy)
    occ_map[occ_npy == 0] = 1 # occupied
    occ_map[occ_npy == 254] = 2 # free

    # Make the image block_size_pix X smaller for faster planning, when interpolating choose the cell with lower number (occupied)
    occ_map = block_reduce(occ_map, block_size=(block_size_pix, block_size_pix), func=np.min, cval=1)
    assert np.min(occ_map) == 1, "kth_occ_map should be 1 (occupied) or 2 (free). There is unknown..."
    # Convert occ_map to what is needed for mask_utils
    # before: (0: unknown, 1: occupied, 2: free)
    # after: (0: free, 0.5: unknown, 1: occupied)
    occ_map = smu.convert_012_labels_to_maskutils_labels(occ_map)
    
    # Load validspace_npy_path which has value either 0: not space, > 0: space
    # import pdb; pdb.set_trace()
    validspace_npy = np.load(validspace_npy_path)
    validspace_map = np.zeros_like(validspace_npy)
    validspace_map[validspace_npy > 0] = 1 # space
    validspace_map[validspace_npy == 0] = 0 # not space
    # Make the image block_size_pix X smaller for faster planning, when interpolating choose the cell with higher number (space)
    validspace_map = block_reduce(validspace_map, block_size=(block_size_pix, block_size_pix), func=np.max, cval=0)
    
    # # Pad both occ_map and validspace_map with lidar_range_pix to account for the possible lidar range 
    laser_range_pix = 500 # 100 #lidar_sim_configs['laser_range_m'] * lidar_sim_configs['pixel_per_meter'] 
    occ_map = np.pad(occ_map, int(laser_range_pix), mode='constant', constant_values=00)
    validspace_map = np.pad(validspace_map, int(laser_range_pix), mode='constant', constant_values=0)

    return occ_map, validspace_map

def init_worker():
    pass

class FrontierPlanner():
    def __init__(self, score_mode=None):
        self.region_size_threshold = 10 # Filters out frontier regions that are smaller than this
        self.score_mode = score_mode
        assert score_mode in ['nearest', 'visvar', 'visunk','nbv-2d','onlyvar', 'mapex', 'hector', 'hectoraug', 'pipe', 'pw-nbv-2d'],\
            "score_mode must be one of ['nearest', 'visvar', 'visunk','nbv-2d','onlyvar', 'mapex', 'hector, 'hectoraug', 'pipe', 'pw-nbv-2d']"
    def get_frontier_centers_given_obs_map(self, obs_map):
        """
        Get frontier centers given global observed map. 

        Args:
            obs_map (np.array): 2D array representing the observed map. (0: free, 0.5: unknown, 1: occupied)

        Returns:
            frontier_region_centers (list): List of frontier region centers. Each element is a list of [row, col]
            frontier_map (np.array): 2D array representing the frontier map. (0: free, 0.5: unknown, 1: occupied)
            num_large_regions (int): Number of large frontier regions (filtered out small regions)
        """
            ## Get current frontiers 
        # Define a kernel that will identify frontier edge cells
        kernel = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])

        # Identify frontier edge cells
        edge_cells = convolve((obs_map == 0.5).astype(int), kernel) > 0
        edge_cells = edge_cells & (obs_map == 0)

        # Group adjacent edge cells into frontier regions
        structure = generate_binary_structure(2, 2)  # 2D structure for connectivity
        frontier_regions, num_regions = label(edge_cells, structure)

        # Filter out small frontier regions
        region_sizes = np.bincount(frontier_regions.ravel())
        large_regions = region_sizes > self.region_size_threshold 
        large_regions[0] = False  # Ignore background
        frontier_map = large_regions[frontier_regions]

        # Get frontier region centers 
        frontier_region_centers = []
        num_large_regions = 0
        for i in range(1, num_regions+1):
            if large_regions[i]:
                region_i = np.argwhere(frontier_regions == i)
                region_center = np.mean(region_i, axis=0)
                # find point in region closest to center 
                dist_to_center = np.linalg.norm(region_i - region_center, axis=1)
                closest_point_to_center = region_i[np.argmin(dist_to_center)]
                frontier_region_centers.append(closest_point_to_center)
                num_large_regions += 1

        return frontier_region_centers, frontier_map, num_large_regions

    def get_frontier_val(self, frontier_i, cost_dist, obs_map, flooded_grid, var_map=None):
        
        # Find currently unknown cells in observed map 
        assert obs_map is not None, "obs_map must be provided if use_visibility_unknown is True"
        obs_unknown = obs_map == 0.5 

        # new flooded grid is intersection between obs_unknown and flooded_grid
        flooded_grid = (flooded_grid == 0) & obs_unknown
        vis_ind = np.argwhere(flooded_grid)
        assert vis_ind.shape[1] == 2
        
        if self.score_mode == 'visvar' or self.score_mode == 'mapex': # count variance of pixels in areas currently unobserved but predicted will be seen
            assert var_map is not None, "var_map must be provided if use_visibility_variance is True"
            frontier_val = torch.sum(var_map[vis_ind[:,0], vis_ind[:,1]])
        elif self.score_mode == 'visunk': # count number of pixels in areas currently unobserved but predicted will be seen
            frontier_val = len(vis_ind)
        elif self.score_mode == 'nbv-2d': # count number of pixels in areas currently not observed but observed will be seen
            frontier_val = len(vis_ind) 
        elif self.score_mode == 'onlyvar':
            assert var_map is not None, "var_map must be provided if use_visibility_variance is True"
            frontier_val = torch.sum(var_map[vis_ind[:,0], vis_ind[:,1]])
        elif self.score_mode == 'hectoraug':
            frontier_val = len(vis_ind) # count number of pixels in areas currently not observed but observed will be seen
            
        else:
            raise NotImplementedError("score_mode not implemented: ", self.score_mode)
        
        if self.score_mode == 'hectoraug':
            frontier_val = np.float64(frontier_val) # no normalization, have to do np.float64 since later there is a .item() call
        else:
            # Normalize frontier_val by the distance to current pose 
            frontier_val = frontier_val / (cost_dist[frontier_i]) #!
        return frontier_val, flooded_grid
    
    def score_frontiers(self, frontier_region_centers, cur_pose, pose_list, pred_maputils, pred_vis_config, obs_map=None, mean_map=None, var_map=None):
        frontier_region_centers = np.array(frontier_region_centers)
        pose_list = np.array(pose_list)

        # euclidean distance to current pose 
        cost_dist = np.linalg.norm(frontier_region_centers - cur_pose, axis=1)
        total_frontier_cost_list = np.zeros_like(cost_dist)
        
        # add euclidean distance if nearest distance
        if self.score_mode == 'nearest':
            total_frontier_cost_list += cost_dist

        viz_most_flooded_grid = None # flood grid with most vis_ind
        viz_medium_flooded_grid = None

        ind1 = None
        ind2 = None

        # Convert different options of map to pyomap
        zeros_map_pyomap = {"PyOMap": makePyOMap(np.zeros_like(obs_map)), "occ_grid": np.zeros_like(obs_map)}
        obs_map_pyomap = {"PyOMap": makePyOMap((obs_map == 1)), "occ_grid":(obs_map==1)}
        
        # Individually go through each frontier if it is one of the modes that is doing a raycast
        if self.score_mode in ['visvar', 'visunk', 'nbv-2d', 'onlyvar', 'mapex','hectoraug']: #!
            # Number of pixels seen
            frontier_val_list = []
            flooded_grid_list = []
            skip_raycast = False 

            # If score mode is visvar or visunk or visvarsubsample, we get map_for_raycast from pred_maputils
            if self.score_mode in ['visvar', 'visunk', 'mapex']:
                assert obs_map.shape == pred_maputils.shape, "obs_map and pred_maputils must have the same shape, but got {} and {}".format(obs_map.shape, pred_maputils.shape)
                pred_maputils_pyomap = {"PyOMap": makePyOMap(pred_maputils), "occ_grid": pred_maputils}
                map_for_raycast = pred_maputils_pyomap 
                raycast_range = pred_vis_config['laser_range_m'] * pred_vis_config['pixel_per_meter']
            elif self.score_mode == 'nbv-2d': 
                map_for_raycast = obs_map_pyomap
                raycast_range = pred_vis_config['laser_range_m'] * pred_vis_config['pixel_per_meter']
            elif self.score_mode == 'onlyvar':
                map_for_raycast = zeros_map_pyomap
                raycast_range = 5 * pred_vis_config['pixel_per_meter']
            # For augmented hector which uses the predicted map for flood fill calculation, we use the predicted map. There should be no raycast range    
            elif self.score_mode == 'hectoraug':
                assert obs_map.shape == pred_maputils.shape, "obs_map and pred_maputils must have the same shape, but got {} and {}".format(obs_map.shape, pred_maputils.shape)
                pred_maputils_pyomap = {"PyOMap": makePyOMap(pred_maputils), "occ_grid": pred_maputils}
                map_for_raycast = pred_maputils_pyomap
                raycast_range = pred_vis_config['laser_range_m'] * pred_vis_config['pixel_per_meter'] #! MAKE SURE TO REMOVE
                skip_raycast = True
                
            else: 
                print("Score_mode not implemented: ", self.score_mode)
                
            for frontier_i, frontier_center in enumerate(frontier_region_centers):
                vis_ind, lidar_mask, inited_flood_grid, actual_hit_points, flooded_grid = \
                smu.get_vis_mask(map_for_raycast['occ_grid'],
                            (frontier_center[0], frontier_center[1]), 
                            laser_range=raycast_range, num_laser=pred_vis_config['num_laser'],
                            occ_map_type='PyOMap', 
                            occ_map_obj=map_for_raycast['PyOMap'],
                            skip_raycast=skip_raycast)
                
                if self.score_mode == 'mapex':
                    # Calculate the probabilistic raycast using mean predicted occupancy map
                    vis_ind_prob, lidar_mask_prob, inited_flood_grid_prob, actual_hit_points_prob, flooded_grid_prob = \
                    smu.get_vis_mask(mean_map, # mean predicted occupancy map
                                    (frontier_center[0], frontier_center[1]),
                                        laser_range=raycast_range, num_laser=pred_vis_config['num_laser'],
                                        raycast_mode='probabilistic',
                                        hit_prob_threshold=0.8, 
                                        skip_raycast=skip_raycast)
                    vis_ind = vis_ind_prob
                    flooded_grid = flooded_grid_prob
                    
                    
                frontier_val, flooded_grid = self.get_frontier_val(frontier_i, cost_dist, obs_map, flooded_grid, var_map)
                
                # Visualization of the most and least flooded grid
                frontier_val_list.append(frontier_val.item())
                flooded_grid_list.append(flooded_grid)
            
            indexed_fv = list(enumerate(frontier_val_list))
            sorted_indexed_fv = sorted(indexed_fv,key=lambda x:x[1],reverse=True)
            ind1 = sorted_indexed_fv[0][0]
            viz_most_flooded_grid = flooded_grid_list[ind1]
            if len(frontier_val_list) > 2:
                ind2 = sorted_indexed_fv[2][0]
                viz_medium_flooded_grid = flooded_grid_list[ind2]

            total_frontier_cost_list += - np.array(frontier_val_list)

        elif self.score_mode == 'nearest':
            pass
        else:
            raise NotImplementedError("score_mode not implemented: ", self.score_mode)
        return frontier_region_centers, total_frontier_cost_list, viz_most_flooded_grid, viz_medium_flooded_grid, ind1, ind2
    
    @staticmethod
    def process_path(path, raycast_range, pred_vis_config, score_mode, mean_map, obs_map, h, w):
        # Retrieve the shared arrays from global variables.

        big_array = []
        start_point = None
        obs_map_pyomap = {"PyOMap": makePyOMap((obs_map == 1)), "occ_grid":(obs_map==1)}

        for point in path:
            if np.array_equal(path[0], point):
                start_point = (point[0], point[1])
                
            if score_mode in ['visvarsubsampleprob', 'visunkprob', 'pipe']:
                new_occ_grid, hit_points = smu.get_hit_ponits(
                    mean_map,
                    (point[0], point[1]),
                    laser_range=raycast_range,
                    num_laser=pred_vis_config['num_laser'],
                    raycast_mode='probabilistic',
                    hit_prob_threshold=0.8
                )
            elif score_mode in ['nbv-2d', 'pw-nbv-2d']:
                map_for_raycast = obs_map_pyomap
                new_occ_grid, hit_points = smu.get_hit_ponits(
                    map_for_raycast['occ_grid'],
                    (point[0], point[1]), 
                    laser_range=raycast_range, num_laser=pred_vis_config['num_laser'],
                    occ_map_type='PyOMap', 
                    occ_map_obj=map_for_raycast['PyOMap']
                )
            # Extend with other modes as needed.
            big_array.append(hit_points)
        
        polygons = [Polygon(coords).buffer(1) for coords in big_array if len(coords) >= 3]

        unioned_geom = unary_union(polygons)

        # Gather holes from interiors
        holes = []
        if unioned_geom.geom_type == 'Polygon':
            holes.extend([list(hole.coords) for hole in unioned_geom.interiors if hole])
        elif unioned_geom.geom_type == 'MultiPolygon':
            for poly in unioned_geom.geoms:
                holes.extend([list(hole.coords) for hole in poly.interiors if hole])

        # Instead of forcing a single Polygon with holes, check the geometry type first
        if not unioned_geom.is_empty:
            if unioned_geom.geom_type == 'Polygon':
                unioned_geom = Polygon(list(unioned_geom.exterior.coords), holes=holes)
            elif unioned_geom.geom_type == 'MultiPolygon':
                print("Unioned geom is a MultiPolygon; skipping conversion to a single Polygon with holes.")

        # Process boundary arrays
        if isinstance(unioned_geom, Polygon):
            boundary_arrays = [np.array(unioned_geom.exterior.coords).astype(int)]
        elif isinstance(unioned_geom, MultiPolygon):
            print("Unioned geom is a MultiPolygon")
            boundary_arrays = [np.array(poly.exterior.coords).astype(int) for poly in unioned_geom.geoms]
        for boundary_array in boundary_arrays:
            inited_flood_grid = smu.init_flood_fill(
                start_point,
                boundary_array,
                new_occ_grid.shape
            )

        for hole in holes:
            hole_array = np.array(hole).astype(int)
            inited_flood_grid_hole = smu.init_flood_fill(
                start_point,
                hole_array,
                new_occ_grid.shape
            )
            # Combine the flood fill results
            inited_flood_grid[inited_flood_grid_hole == 0] = 0

        flooded_grid = smu.flood_fill_simple(start_point, inited_flood_grid)
        flooded_grid[inited_flood_grid == 0] = 0.5


        if len(path) > 1:
            diffs = np.diff(np.array(path), axis=0)
            L = np.sum(np.linalg.norm(diffs, axis=1))
        else:
            L = 0

        r = 200  # for example, the raycast radius in pixels
        minkowski_area = 2 * r * L + np.pi * (r ** 2)

        zero_count = np.count_nonzero(flooded_grid == 0)
        if zero_count > minkowski_area:
            flooded_grid.fill(np.nan)
            return flooded_grid

        if np.any(flooded_grid != 0):
            obs_unknown = obs_map == 0.5
            flooded_grid = (flooded_grid == 0) & obs_unknown

        return flooded_grid

    def score_frontiers_pathwise(self, t, frontier_region_centers, path_collection,
                                 pred_maputils, pred_vis_config,
                                 obs_map, mean_map, var_map):
        
        frontier_region_centers = np.array(frontier_region_centers)
        best_frontier_val = -np.inf
        second_best_frontier_val = -np.inf
        viz_most_flooded_grid = None
        viz_medium_flooded_grid = None
        locked_frontier_center = None
        second_locked_frontier_center = None
        best_ind = -1
        medium_ind = -1

        # Subsample each path.
        n = pred_vis_config['pathwise_index'] 
        for i in range(len(path_collection)):
            path_collection[i] = path_collection[i][n-1::n]

        raycast_range = pred_vis_config['laser_range_m'] * pred_vis_config['pixel_per_meter']

        path_args = []
        for path in path_collection:
            args_tuple = (
                path,
                raycast_range,
                pred_vis_config,
                self.score_mode,
                mean_map,
                obs_map,
                obs_map.shape[0],
                obs_map.shape[1]
            )

            path_args.append(args_tuple)

        # Tune the pool: use half the available CPUs and a specific chunksize.
        pool_size = max(1, mp.cpu_count() - 1)
        chunksize = max(1, len(path_args) // (pool_size * 2))
        pool = mp.Pool(
            processes=pool_size,
            initializer=init_worker,
            initargs=()
        )
        flooded_results = pool.starmap(self.process_path, path_args, chunksize=chunksize)
        pool.close()
        pool.join()

        frontier_cost_list  = []
        flooded_grid_list   = []
        frontier_vals       = []

        # Process the results to determine frontier values
        for path, flooded in zip(path_collection, flooded_results):
            length = max(1, len(path))
            if not np.any((flooded == 0) | (flooded == 0.5) | (flooded == 1)):
                val = -np.inf
            if self.score_mode == 'pipe':
                mask = torch.tensor(flooded, dtype=torch.bool)
                val  = torch.sum(var_map[mask]).item() / length
            elif self.score_mode == 'pw-nbv-2d':
                val  = np.sum(flooded) / length
            else:
                raise ValueError(f'unknown score_mode {self.score_mode}')

            frontier_vals.append(val)
            flooded_grid_list.append(flooded)
            frontier_cost_list.append(-val)

        frontier_cost_list = np.asarray(frontier_cost_list)
        frontier_vals      = np.asarray(frontier_vals)


        if len(frontier_vals) == 1:
            best_ind   = 0
            medium_ind = -1
        else:
            order = np.argpartition(-frontier_vals, 1)
            best_ind, medium_ind = order[:2]
            if frontier_vals[medium_ind] > frontier_vals[best_ind]:
                best_ind, medium_ind = medium_ind, best_ind



        viz_most_flooded_grid   = flooded_grid_list[best_ind]
        viz_medium_flooded_grid = (flooded_grid_list[medium_ind]
                                if medium_ind != -1 else None)

        locked_frontier_center  = frontier_region_centers[best_ind]

        return (frontier_region_centers,
                frontier_cost_list,
                viz_most_flooded_grid,
                viz_medium_flooded_grid,
                best_ind,
                medium_ind,
                locked_frontier_center)



class Mapper():
    def __init__(self, gt_map, lidar_sim_configs, use_distance_transform_for_planning=False, dt_floor_val=10):
        self.gt_map = gt_map
        self.accum_hit_points = np.zeros((0,2)).astype(int) # Keep track of the hit points that the agent has seen, this is what makes the map
        self.obs_map = np.ones_like(gt_map) * 0.5 # Keep track of the map that the agent has seen
        self.lidar_sim_configs = lidar_sim_configs
        self.dilate_diam_for_planning = lidar_sim_configs['dilate_diam_for_planning']
        self.use_distance_transform_for_planning = use_distance_transform_for_planning
        self.dt_floor_val = dt_floor_val

        self.prev_obs_map = np.ones_like(gt_map)
        self.prev_pred_map = np.ones_like(gt_map) 
        self.curr_pred_map = np.ones((gt_map.shape[0],gt_map.shape[1],3))#((1000,1000,3))*255#np.ones_like(gt_map)
        self.combined_pred_map = np.ones_like(gt_map)
        self.combined_obs1_pred2_map = np.ones_like(gt_map) 

        self.combined_obs_map = np.ones_like(gt_map)
        
        self.gt_map_pyomap = makePyOMap(self.gt_map) 

    def get_instant_obs_at_pose(self, xy_pose):
        """
        Get instantaneous observation with LiDAR sim at a given pose.
        No accumulation."""
        vis_ind, lidar_mask, inited_flood_grid, actual_hit_points, flooded_grid = \
        smu.get_vis_mask(self.gt_map,
                     (xy_pose[0], xy_pose[1]), 
                     laser_range=self.lidar_sim_configs['laser_range_m'] * self.lidar_sim_configs['pixel_per_meter'], num_laser=self.lidar_sim_configs['num_laser'],
                     occ_map_type='PyOMap', occ_map_obj=self.gt_map_pyomap)
        obs_dict = {
            'vis_ind': vis_ind,
            'actual_hit_points': actual_hit_points,
        }
        return obs_dict
    
    def accumulate_obs_given_dict(self, obs_dict):
        """
        Accumulate current observation into the underlying observed map.
        """
        # Get needed member variables
        vis_ind = obs_dict['vis_ind']
        actual_hit_points = obs_dict['actual_hit_points']

        # Add hit poitns to accum_hit_points
        self.accum_hit_points = np.concatenate([self.accum_hit_points, actual_hit_points], axis=0)

        # Update obs_map
        # keep track of original index that is occupied in the previous obs_map
        occ_mask = (self.obs_map == 1)
        self.obs_map[vis_ind[:,0], vis_ind[:,1]] = 0 # update free !
        self.obs_map[occ_mask] = 1 # update occupied !
        self.obs_map[actual_hit_points[:,0], actual_hit_points[:,1]] = 1 # update occupied !

    def observe_and_accumulate_given_pose(self, pose):
        """
        Observe and accumulate given current pose.
        """
        # Number of previously observed cells
        num_prev_obs_cells = np.sum(self.obs_map == 0) + np.sum(self.obs_map == 1)

        # Get observation
        obs_dict = self.get_instant_obs_at_pose(pose)
        # Accumulate observation
        self.accumulate_obs_given_dict(obs_dict)
        # Number of newly observed cells
        num_new_obs_cells = np.sum(self.obs_map == 0) + np.sum(self.obs_map == 1) - num_prev_obs_cells

    def inflate_map(self, map, unknown_as_occ=False):
        # Get inflated obs map for local planner
        
        dilated_map_for_planning = map.copy()
        inverted_binary_map_for_planning = map.copy() > 0.5 
        inverted_dilated_map_for_planning = binary_dilation(inverted_binary_map_for_planning, structure=np.ones((self.dilate_diam_for_planning, self.dilate_diam_for_planning)))
        # Find areas of 1s in inverted_dilated_map_for_planning, then add it to dilated_map_for_planning
        dilated_map_for_planning[inverted_dilated_map_for_planning] = 1
        # Get pyastar-compatible cost map
        occ_grid_pyastar = np.zeros((dilated_map_for_planning.shape[0], dilated_map_for_planning.shape[1]), dtype=np.float32) # 0: free/unknown, np.inf: occupied
        
        if self.use_distance_transform_for_planning:
            distance_transform = scipy.ndimage.distance_transform_cdt(dilated_map_for_planning == 0) * -1 
            biased_distance_transform = np.clip(distance_transform + self.dt_floor_val, 1, self.dt_floor_val) 
            if unknown_as_occ: 
                # Unknown as occupied: higher costs closer to walls
                # Assign distance transform values only to free cells
                free_space_mask = (dilated_map_for_planning == 0)
                occ_grid_pyastar[free_space_mask] =  biased_distance_transform[free_space_mask]
                
                occ_grid_pyastar[dilated_map_for_planning > 0] = np.inf # occupied or unknown
            else:
                # Unknown as free: lower costs closer to walls
                # Assign distance transform values to free and unknown spaces 
                free_or_unknown_space_mask = (dilated_map_for_planning >= 0)
                occ_grid_pyastar[free_or_unknown_space_mask] =  biased_distance_transform[free_or_unknown_space_mask]
                occ_grid_pyastar[dilated_map_for_planning == 1] = np.inf
                
        else:
            if unknown_as_occ:
                occ_grid_pyastar[dilated_map_for_planning == 0] = 1 # free
                occ_grid_pyastar[dilated_map_for_planning > 0] = np.inf # occupied or unknown
            else: # unknown is free
                occ_grid_pyastar[dilated_map_for_planning >= 0] = 1 # free or unknown
                occ_grid_pyastar[dilated_map_for_planning == 1] = np.inf # occupied

        return occ_grid_pyastar
    
    def get_inflated_planning_maps(self, unknown_as_occ=False):
        return self.inflate_map(self.obs_map, unknown_as_occ=False)


    