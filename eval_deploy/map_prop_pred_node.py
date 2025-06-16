### Ros node for map prediction with local map propagataion

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
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
#import rospkg
import time
from std_msgs.msg import Float32MultiArray
from numba import njit

# Custom imports
import sys
# sys.path.append('/workspace/src/map_prediction_toolbox/')
sys.path.append('/home/cherie/research/hmap/map_prediction_ws/src/map_prediction_toolbox/')
# from gen_building_utils import * 
from options.deploy_options import *
from models.predictors import get_predictor_from_options
from eval_deploy import deploy_utils as dutils
from eval_deploy import viz_utils as vutils
from eval_deploy import glocal_utils as glocal
from data_factory import gen_building_utils 
from data_factory import simple_mask_utils as smu

from core_trajectory_msgs.srv import *

class GetVizMask:

    def __init__(self):
        pass

    @njit(fastmath=True)
    def get_vis_mask(occupancy_grid, robot_pos, laser_range=50, num_laser=100):
        #! should return vis_mask, or rename to get_vis_ind
        # free_points = np.zeros((num_laser,2), np.int64) # free points
        hit_points = np.zeros((num_laser,2), np.int64) # actual hit points + last bresenham point (for some reason need this for flodding)
        actual_hit_points = np.zeros((num_laser,2), np.int64) # actual hit points
        # free_points_index = 0
        hit_points_index = 0
        actual_hit_points_index = 0
        for orientation in np.linspace(0, 2*np.pi, num_laser):
            end_point = np.array([int(robot_pos[0] + laser_range * np.cos(orientation)), int(robot_pos[1] + laser_range * np.sin(orientation))])
            
            # setup initial conditions
            x1, y1 = robot_pos
            x2, y2 = end_point
            dx = x2 - x1
            dy = y2 - y1
            is_steep = abs(dy) > abs(dx)  # determine how steep the line is
            if is_steep:  # rotate line
                x1, y1 = y1, x1
                x2, y2 = y2, x2
            # swap start and end points if necessary and store swap state
            swapped = False
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                swapped = True
            dx = x2 - x1  # recalculate differentials
            dy = y2 - y1  # recalculate differentials
            error = int(dx / 2.0)  # calculate error
            y_step = 1 if y1 < y2 else -1
            # iterate over bounding box generating points between start and end
            y = y1
            points = []
            for x in range(x1, x2 + 1):
                coord = [y, x] if is_steep else [x, y]
                points.append(coord)
                error -= abs(dy)
                if error < 0:
                    y += y_step
                    error += dx
            if swapped:  # reverse the list if the coordinates were swapped
                points.reverse()
            bresenham_points = np.array(points)

            # Go through the points and see the first hit
            # TODO: do a check if any first?
            for i in range(len(bresenham_points)):
                # if bresenham point is in the map 
                if bresenham_points[i,0] < 0 or bresenham_points[i,0] >= occupancy_grid.shape[0] or bresenham_points[i,1] < 0 or bresenham_points[i,1] >= occupancy_grid.shape[1]:
                    if i != 0:
                        hit_points[hit_points_index] = bresenham_points[i-1]
                        hit_points_index += 1
                    break # don't use this bresenham point 
                
                if occupancy_grid[bresenham_points[i,0], bresenham_points[i,1]] == 1: # hit if it is void or occupied #! THINK IF THIS IS A GOOD ASSUMPTION
                    # actual_hit_points = np.vstack((actual_hit_points, np.atleast_2d(bresenham_points[i])))
                    # hit_points = np.vstack((hit_points, np.atleast_2d(bresenham_points[i])))
                    actual_hit_points[actual_hit_points_index] = bresenham_points[i]
                    actual_hit_points_index += 1
                    hit_points[hit_points_index] = bresenham_points[i]
                    hit_points_index += 1
                    
                    break
                else: # no hits
                    # free_points = np.vstack((free_points, np.atleast_2d(bresenham_points[i])))
                    # free_points[free_points_index] = np.atleast_2d(bresenham_points[i])
                    # free_points_index += 1

                    if i == len(bresenham_points) - 1:
                        # hit_points = np.vstack((hit_points, np.atleast_2d(end_point))) # need to add this for proper flooding for vis mask
                        hit_points[hit_points_index] = np.atleast_2d(end_point)
                        hit_points_index += 1
                        # hit_points.append(end_point) # need to add this for proper flooding for vis mask
                        break
                    
        
        # Convert to np.array
        # free_points_1 = np.array(free_points[:,0]).as 
        # free_points = np.array(free_points)
        # hit_points = np.array(hit_points)
        # actual_hit_points = np.array(actual_hit_points)

        # Define the extent of the image
        grid_xmin, grid_xmax = (0, occupancy_grid.shape[0])
        grid_ymin, grid_ymax = (0, occupancy_grid.shape[1])
        extent = [grid_xmin, grid_xmax, grid_ymin, grid_ymax]


        new_occ_grid = np.ones_like(occupancy_grid) * 0.5

        # if len(free_points) == 0:
        #     return [], new_occ_grid
        # else:

        # for i in range(free_points_index):
        #     new_occ_grid[free_points[i,0], free_points[i,1]] = 0.0

        for i in range(hit_points_index):
            new_occ_grid[hit_points[i,0], hit_points[i,1]] = 1.0

        # new_occ_grid[free_points[:,0], free_points[:,1]] = 0.0 # free points
        # new_occ_grid[hit_points[:,0], hit_points[:,1]] = 1.0 # hit points

        # Get vis mask by flood filling free space boundary
        center_x, center_y = robot_pos
        prev_ix, prev_iy = center_x - 1, center_y
        occupancy_map = (np.ones(new_occ_grid.shape)) * 0.5
        # append first obstacle point to last
        obstacle_points = np.vstack((hit_points, np.atleast_2d(hit_points[0])))
        for (x, y) in zip(obstacle_points[:,0], obstacle_points[:,1]):
            # x coordinate of the the occupied area
            ix = int(x)
            # y coordinate of the the occupied area
            iy = int(y)
            x1, y1 = (prev_ix, prev_iy)
            x2, y2 = (ix, iy)
            dx = x2 - x1
            dy = y2 - y1
            is_steep = abs(dy) > abs(dx)  # determine how steep the line is
            if is_steep:  # rotate line
                x1, y1 = y1, x1
                x2, y2 = y2, x2
            # swap start and end points if necessary and store swap state
            swapped = False
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                swapped = True
            dx = x2 - x1  # recalculate differentials
            dy = y2 - y1  # recalculate differentials
            error = int(dx / 2.0)  # calculate error
            y_step = 1 if y1 < y2 else -1
            # iterate over bounding box generating points between start and end
            y = y1
            points = []
            for x in range(x1, x2 + 1):
                coord = [y, x] if is_steep else [x, y]
                points.append(coord)
                error -= abs(dy)
                if error < 0:
                    y += y_step
                    error += dx
            if swapped:  # reverse the list if the coordinates were swapped
                points.reverse()
            free_area = np.array(points)
            for fa in free_area:
                occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
            prev_ix = ix
            prev_iy = iy

        # stop here


        occupancy_map = np.copy(occupancy_map)
        sx, sy = occupancy_map.shape
        fringe = []
        fringe.append(robot_pos)
        while fringe:
            
            n = fringe.pop()
            nx, ny = n
            unknown_val = 0.5
            # West
            if nx > 0:
                if occupancy_map[nx - 1, ny] == unknown_val:
                    occupancy_map[nx - 1, ny] = 0
                    fringe = [((nx - 1, ny))] + fringe
            # East
            if nx < sx - 1:
                if occupancy_map[nx + 1, ny] == unknown_val:
                    occupancy_map[nx + 1, ny] = 0
                    fringe = [((nx + 1, ny))] + fringe
            # North
            if ny > 0:
                if occupancy_map[nx, ny - 1] == unknown_val:
                    occupancy_map[nx, ny - 1] = 0
                    fringe = [((nx, ny - 1))] + fringe
            # South
            if ny < sy - 1:
                if occupancy_map[nx, ny + 1] == unknown_val:
                    occupancy_map[nx, ny + 1] = 0
                    fringe = [((nx, ny + 1))] + fringe
        # flooded_grid = occupancy_map
        # print("flooded grid shape: ", flooded_grid.shape)
        # then add hit points back in
        # flooded_grid[hit_points[:,0], hit_points[:,1]] = 1

        # get index of vis mask
        vis_ind = np.argwhere(occupancy_map == 0)
        # print(vis_ind.shape)
        return vis_ind, new_occ_grid, occupancy_map, actual_hit_points

class MapPropPredNode:

    def __init__(self):
        # initialize ros node
        rospy.init_node('map_prop_pred_node')

        ensemble_folder_name = '20230910_varyhallway_roomsize' # Path to ensemble folder, in weights folder
        self.map_configs = {
            'min_start_end_dist': 50,
            'collect_interval_m': 5,
            'laser_range': 50,
            'num_laser': 500,
            'num_rand_traj_per_map':1,
            'percent_test': 0.2,
            "local_map_size": 30, # meters
        }
        self.stride_m = 15 # meters in between patches of global map
        self.multiplier = 5 # how many cells in 1 meter of global map. 1/cell_size_meter
        self.known_threshold = 0.01 # how much of the local map needs to be known to be used as 0th layer for map propagation
        self.subsampling_freq = 30 # how many cells to subsample for variance map
        self.max_propagation_layer = 1 # how many layers to propagate
        self.getvizmask = GetVizMask()

        # Save off the current predicted map for service call
        self.pred_map_class = None
        self.viz_flag = True

        # Initialize deploy variables given main variables
        cmd_line_args_dict = ['--name', 'deploy']
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #os.environ['L2M_root_path'] = rospack.get_path()
        # os.environ['L2M_root_path'] = '/home/seungchan/Documents/map_prediction_ws/src/map_prediction_toolbox/' # change for different computer
        os.environ['L2M_root_path'] = '/home/cherie/research/hmap/map_prediction_ws/src/map_prediction_toolbox/' # change for different computer

        options = DeployOptions().parse_args(cmd_line_args_dict)

        # Initialize model ensembles # TODO: make into function
        ensemble_dir = os.path.join(options.root_path, 'weights', ensemble_folder_name)
        print(ensemble_dir)
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
        self.models_dict = models_dict
        print("Loaded {} models".format(ensemble_size))

        ## For loop over trajectory 
        self.global_transform = Compose([
                    ToTensor(),
                ]) # do not do any resizing
        self.bridge = CvBridge()
        
        # Initialize subscriber 
        self.sub_input_global_map_colored = rospy.Subscriber('/input_global_map', Image, self.input_global_map_cb, queue_size=1, buff_size=2**24)

        self.random_viewpoints_sub = rospy.Subscriber('/random_viewpoints', Float32MultiArray, self.callback_score_viewpoints, queue_size=32)

        # self.sub_global_obs_map_raw = rospy.Subscriber('/global_obs_map_raw', Image, self.global_obs_map_raw_cb, queue_size=1, buff_size=2**24)

        # Initialize publishers 
        self.pub_global_observed_map_raw = rospy.Publisher('/global_obs_map_raw', Image, queue_size=1) 
        self.pub_global_observed_map_colored = rospy.Publisher('/global_obs_map_colored', Image, queue_size=1) 
        self.pub_global_observed_class = rospy.Publisher('/global_obs_map_class', Image, queue_size=1) 
        self.pub_global_pred_map_colored = rospy.Publisher('/global_pred_map_colored', Image, queue_size=1)
        self.pub_global_pred_map_raw_mean = rospy.Publisher('/global_pred_map_raw_mean', Image, queue_size=1)
        self.pub_global_pred_map_class = rospy.Publisher('/global_pred_map_class', Image, queue_size=1)
        self.pub_global_pred_variance = rospy.Publisher('/global_pred_variance', Image, queue_size=1)
        self.pub_global_pred_variance_subsampled = rospy.Publisher('/global_pred_variance_subsampled', Image, queue_size=1)

        # Initialize publishers for resampled maps (for logging)
        self.resample_rate = 10 # only publish resampled maps every resample_rate
        self.pub_global_observed_map_raw_resampled = rospy.Publisher('/global_obs_map_raw_resampled', Image, queue_size=1)
        self.pub_global_observed_map_raw_count = 0
        self.pub_global_observed_map_colored_resampled = rospy.Publisher('/global_obs_map_colored_resampled', Image, queue_size=1)
        self.pub_global_observed_map_colored_count = 0
        self.pub_global_observed_class_resampled = rospy.Publisher('/global_obs_map_class_resampled', Image, queue_size=1)
        self.pub_global_observed_class_count = 0
        self.pub_global_pred_map_colored_resampled = rospy.Publisher('/global_pred_map_colored_resampled', Image, queue_size=1)
        self.pub_global_pred_map_colored_count = 0
        self.pub_global_pred_map_raw_mean_resampled = rospy.Publisher('/global_pred_map_raw_mean_resampled', Image, queue_size=1)
        self.pub_global_pred_map_raw_mean_count = 0
        self.pub_global_pred_map_class_resampled = rospy.Publisher('/global_pred_map_class_resampled', Image, queue_size=1)
        self.pub_global_pred_map_class_count = 0
        self.pub_global_pred_variance_resampled = rospy.Publisher('/global_pred_variance_resampled', Image, queue_size=1)
        self.pub_global_pred_variance_count = 0
        self.pub_global_pred_variance_subsampled_resampled = rospy.Publisher('/global_pred_variance_subsampled_resampled', Image, queue_size=1)
        self.pub_global_pred_variance_subsampled_count = 0

        self.scored_random_viewpoints_pub = rospy.Publisher('/scored_random_viewpoints', Float32MultiArray, queue_size=1)

        # Initialize services
        if not rospy.get_param("/map_prop_pred_node/use_cpp_service"):
            self.score_viewpoint_service = rospy.Service('score_viewpoints_service', TrajInfoGain, self.handle_score_viewpoints_service)
            print("Using python viewpoint service")

        self.obs_map_unknown = None

        # #! For mock data, timer to publish global observed map 
        # self.pub_mock_obs_map_timer = rospy.Timer(rospy.Duration(1), self.pub_mock_obs_map_callback)
        self.obs_map_raw_msg = None 
        self.timer_global_pred = rospy.Timer(rospy.Duration(0.25), self.global_pred_cb, reset=True)
        # self.mock_count_i = 0 

    def input_global_map_cb(self, input_maps_msg):
        # Convert the colored image to raw 
        input_image_colorized = self.bridge.imgmsg_to_cv2(input_maps_msg, desired_encoding='passthrough')
        # print("input_image", input_image_colorized)
        # Create one-hot class encoding given colors
        unique_colors = np.array([[255, 255, 255], [0, 0, 204], [51, 255, 51]]) #! hardcode given input data (white, blue, green)
        one_hot = np.zeros((input_image_colorized.shape[0], input_image_colorized.shape[1], unique_colors.shape[0]), dtype=np.uint8)
        
        for i, color in enumerate(unique_colors):
            mask = np.all(input_image_colorized == color, axis=-1)

            # print(np.sum(mask))
            one_hot[:, :, i][mask] = 255
        # print("one_hot shape:", one_hot.shape)
        input_image_onehot_topub_image = self.bridge.cv2_to_imgmsg(one_hot, 'passthrough')
        self.pub_global_observed_map_raw.publish(input_image_onehot_topub_image)
        self.obs_map_raw_msg = input_image_onehot_topub_image

        # Publish resampled
        self.pub_global_observed_map_raw_count += 1
        if self.pub_global_observed_map_raw_count % self.resample_rate == 0:
            self.pub_global_observed_map_raw_resampled.publish(input_image_onehot_topub_image)

        # print("Publisheed global observed map raw")


    def pub_mock_obs_map_callback(self, event):
        print("Publishing mock observed map")
        global_observed_map = 1-self.global_obs_map_list[self.mock_count_i]
        print("global_observed_map shape: ", global_observed_map.shape)
        global_obs_onehot_tensor = dutils.convert_maputils_labelmaps_to_model_input_format(global_observed_map, self.global_transform)
        print("global_obs_onehot_tensor shape: ", global_obs_onehot_tensor.shape)

        input_image_onehot_topub_image = global_obs_onehot_tensor[0].permute(1,2,0).cpu().numpy()
        input_image_onehot_topub_image = (input_image_onehot_topub_image * 255).astype(np.uint8)
        print("input_image_onehot_topub_image min/max: ", np.min(input_image_onehot_topub_image), np.max(input_image_onehot_topub_image))
        input_image_onehot_topub_image_msg = self.bridge.cv2_to_imgmsg(input_image_onehot_topub_image, "passthrough")
        print("input_image_onehot_topub_image shape: ", input_image_onehot_topub_image.shape)
        self.pub_global_observed_map_raw.publish(input_image_onehot_topub_image_msg)

        self.mock_count_i += 1
        self.mock_count_i = self.mock_count_i % len(self.global_obs_map_list)

    def global_pred_cb(self, event):
        # print("Received global_obs_map_raw_msg", self.obs_map_raw_msg)
        obs_map_raw_msg = self.obs_map_raw_msg
        if obs_map_raw_msg is not None:
            # print("Received global_obs_map_raw_msg: not none")
            obs_map_raw = self.bridge.imgmsg_to_cv2(obs_map_raw_msg, desired_encoding="passthrough")
            obs_map_raw_onehot = obs_map_raw / 255.0 # convert to 0-1
            obs_map_raw_onehot_transformed = self.global_transform(obs_map_raw_onehot).unsqueeze(0)
            # print("obs_map_raw_onehot_transformed shape: ", obs_map_raw_onehot_transformed.shape) # 1, 3, 80, 90
            # obs_map_raw_onehot_transformed = obs_map_raw_onehot.unsqueeze(0)

            # # Convert obs_map_raw to colorized map for visualization
            obs_map_colored = vutils.get_colorized_map(obs_map_raw_onehot_transformed).permute(1,2,0).cpu().numpy()
            #self.obs_map_colored = obs_map_colored
            #print("one hot shape: ", obs_map_raw_onehot.shape)
            #self.obs_map_unknown = obs_map_raw_onehot[:,:,0]
            # print("obs_map_colored shape: ", obs_map_colored.shape)
            obs_map_colored_msg = self.bridge.cv2_to_imgmsg(obs_map_colored, "passthrough")
            self.pub_global_observed_map_colored.publish(obs_map_colored_msg)

            # Publish resampled (observed colorized)
            self.pub_global_observed_map_colored_count += 1
            if self.pub_global_observed_map_colored_count % self.resample_rate == 0:
                self.pub_global_observed_map_colored_resampled.publish(obs_map_colored_msg)

            obs_map_class = np.argmax(obs_map_raw_onehot_transformed, axis=1)[0]
            obs_map_class_msg = self.bridge.cv2_to_imgmsg((obs_map_class).numpy().astype(np.uint8), 'passthrough')
            self.pub_global_observed_class.publish(obs_map_class_msg)
            # Publish resampled (observed class)
            self.pub_global_observed_class_count += 1
            if self.pub_global_observed_class_count % self.resample_rate == 0:
                self.pub_global_observed_class_resampled.publish(obs_map_class_msg)    

            # # # Predict global map with global observed map

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # print('before rollout')
            before_pred_time = rospy.get_time()
            global_pred_propagate_onehot_tensor_rollouts = glocal.calculate_rollouts_global_map_prediction(self.models_dict, device, obs_map_raw_onehot_transformed, 
                                                                                                           self.max_propagation_layer,
                                                                                                           self.map_configs, self.multiplier, self.stride_m, self.known_threshold, show_viz=False)

            #rospy.loginfo("Time to predict global maps: {}".format(rospy.get_time() - before_pred_time))
            # print("global_pred_propagate_onehot_tensor_rollouts.shape: ", global_pred_propagate_onehot_tensor_rollouts.shape)
            global_pred_propagate_onehot_tensor_mean = torch.mean(global_pred_propagate_onehot_tensor_rollouts, dim=0).unsqueeze(0)
            global_pred_propagate_onehot_tensor_mean_img = (global_pred_propagate_onehot_tensor_mean[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            global_pred_propagate_onehot_tensor_mean_msg = self.bridge.cv2_to_imgmsg(global_pred_propagate_onehot_tensor_mean_img, 'passthrough')
            self.pub_global_pred_map_raw_mean.publish(global_pred_propagate_onehot_tensor_mean_msg)
            # Publish resampled (predicted raw mean)
            self.pub_global_pred_map_raw_mean_count += 1
            if self.pub_global_pred_map_raw_mean_count % self.resample_rate == 0:
                self.pub_global_pred_map_raw_mean_resampled.publish(global_pred_propagate_onehot_tensor_mean_msg)

             # Save prediction for service call
            self.pred_map_class = np.argmax(global_pred_propagate_onehot_tensor_mean, axis=1)[0]
            pred_map_class_msg = self.bridge.cv2_to_imgmsg((self.pred_map_class).numpy().astype(np.uint8), 'passthrough')
            self.pub_global_pred_map_class.publish(pred_map_class_msg)
            # Publish resampled (predicted class)
            self.pub_global_pred_map_class_count += 1
            if self.pub_global_pred_map_class_count % self.resample_rate == 0:
                self.pub_global_pred_map_class_resampled.publish(pred_map_class_msg)

            # Convert global_pred_propagate_onehot_tensor_mean to colorized map for visualization
            global_pred_propagate_colored = vutils.get_colorized_map(global_pred_propagate_onehot_tensor_mean).permute(1,2,0).cpu().numpy()
            # print("global_pred_propagate_colored shape: ", global_pred_propagate_colored.shape)
            global_pred_propagate_colored_msg = self.bridge.cv2_to_imgmsg(global_pred_propagate_colored, "passthrough")
            self.pub_global_pred_map_colored.publish(global_pred_propagate_colored_msg)

            # Publish resampled (predicted colorized)
            self.pub_global_pred_map_colored_count += 1
            if self.pub_global_pred_map_colored_count % self.resample_rate == 0:
                self.pub_global_pred_map_colored_resampled.publish(global_pred_propagate_colored_msg)

            # Get the variance 
            var_class = 1
            variance_global_pred_propagate_onehot_tensor = torch.var(global_pred_propagate_onehot_tensor_rollouts[:,var_class,:,:], dim=0)
            # import pdb; pdb.set_trace()
            assert torch.all(variance_global_pred_propagate_onehot_tensor < 1), "Variance is greater than 1"
            variance_tensor_uint8 = (variance_global_pred_propagate_onehot_tensor*255).numpy().astype(np.uint8)
            var_msg = self.bridge.cv2_to_imgmsg(variance_tensor_uint8, 'passthrough')
            self.pub_global_pred_variance.publish(var_msg)
            # Publish resampled (predicted variance)
            self.pub_global_pred_variance_count += 1
            if self.pub_global_pred_variance_count % self.resample_rate == 0:
                self.pub_global_pred_variance_resampled.publish(var_msg)

            # Get the subsampled variance
            subsampled = torch.zeros((int(variance_global_pred_propagate_onehot_tensor.shape[0]/self.subsampling_freq), int(variance_global_pred_propagate_onehot_tensor.shape[1]/self.subsampling_freq)))
            for sub_i in range(int(variance_global_pred_propagate_onehot_tensor.shape[0]/self.subsampling_freq)):
                for sub_j in range(int(variance_global_pred_propagate_onehot_tensor.shape[1]/self.subsampling_freq)):
                    query_patch = variance_global_pred_propagate_onehot_tensor[sub_i*self.subsampling_freq:(sub_i+1)*self.subsampling_freq,sub_j*self.subsampling_freq:(sub_j+1)*self.subsampling_freq]
                    subsampled[sub_i,sub_j] = torch.sum(query_patch)

            subsampled_var_msg = self.bridge.cv2_to_imgmsg(subsampled.float().detach().cpu().numpy(), '32FC1')
            self.pub_global_pred_variance_subsampled.publish(subsampled_var_msg)
            # Publish resampled (predicted subsampled variance)
            self.pub_global_pred_variance_subsampled_count += 1
            if self.pub_global_pred_variance_subsampled_count % self.resample_rate == 0:
                self.pub_global_pred_variance_subsampled_resampled.publish(subsampled_var_msg)
            

            #rospy.loginfo("Time to predict, variance and subsample: {}".format(rospy.get_time() - before_pred_time))

    def callback_score_viewpoints(self, input_msg):
        print("callback")
        pred_occupancy_grid = np.zeros((self.pred_map_class.shape[0], self.pred_map_class.shape[1])).astype(np.float32)
        pred_occupancy_grid[self.pred_map_class == 0] = 0.5
        pred_occupancy_grid[self.pred_map_class == 1] = 1
        pred_occupancy_grid[self.pred_map_class == 2] = 0
        self.obs_map_unknown = pred_occupancy_grid
        random_viewpoints = input_msg.data
        l = len(random_viewpoints)
        value_vec = []
        for i in range(int(l/2)):
            x = random_viewpoints[i*2]
            y = random_viewpoints[i*2+1]
            vis_mask_tot = np.zeros((self.pred_map_class.shape[0], self.pred_map_class.shape[1])).astype(np.float32)
            #print("vis mask tot",vis_mask_tot)
            vis_ind, lidar_mask, inited_flood_grid, actual_hit_points = smu.get_vis_mask(pred_occupancy_grid,(int(1000-5*y), int(5*x)), laser_range=250, num_laser=50)
            vis_mask_tot[vis_ind[:,0], vis_ind[:,1]] = 1.0
            #print("obs_map_unknown", self.obs_map_unknown)   
            intersection_indices = np.where((vis_mask_tot == 0.0) & (self.obs_map_unknown == 0.5))
            intersection_area = np.zeros((vis_mask_tot.shape[0], vis_mask_tot.shape[1]))
            intersection_area[intersection_indices] = 1
            num_pix_vis_mask = np.sum(vis_mask_tot)
            num_pix_vis_intersect_mask = np.sum(intersection_area)
            value_vec.append(x)
            value_vec.append(y)
            value_vec.append(num_pix_vis_intersect_mask)
            plt_png_save = False
            if plt_png_save:
                plt.clf()
                plt_row = 1
                plt_col = 3
                plt.figure(figsize=(30,10))
                plt.subplot(plt_row, plt_col, 1)
                plt.imshow(self.obs_map_unknown)
                plt.scatter(int(5*x), int(1000-5*y), c='r')
                plt.title("observed_map_unknown_area")
                    
                plt.subplot(plt_row, plt_col, 2)
                plt.imshow(vis_mask_tot)
                plt.scatter(int(5*x), int(1000-5*y), c='r')
                plt.title("visbility mask, num pixels: {}".format(num_pix_vis_mask))
                    
                plt.subplot(plt_row, plt_col, 3)
                plt.imshow(intersection_area)
                plt.scatter(int(5*x), int(1000-5*y), c='r')
                plt.title("intersection mask, num pixels: {}".format(num_pix_vis_intersect_mask))
                plt.savefig('hi.png')
        msg = Float32MultiArray()
        msg.data = value_vec
        self.scored_random_viewpoints_pub.publish(msg)
    

    def handle_score_viewpoints_service(self, req): 
        print("got service call")
        # Check to make sure that the pred_map_class is not None
        if self.pred_map_class is None:
            return TrajInfoGainResponse([])
        
        print("passed return statement")
        pred_occupancy_grid = np.zeros((self.pred_map_class.shape[0], self.pred_map_class.shape[1])).astype(np.float32)
        print("pred occ grid: ", pred_occupancy_grid.shape)
        print("pred map class: ", self.pred_map_class.shape)
        pred_occupancy_grid[self.pred_map_class == 0] = 0.5
        pred_occupancy_grid[self.pred_map_class == 1] = 1
        pred_occupancy_grid[self.pred_map_class == 2] = 0
        self.obs_map_unknown = pred_occupancy_grid
        # Compute the scores for each trajectory
        print("passed this point")
        print("pred occ grid now: ", pred_occupancy_grid.shape)
        values_vec = []
        print("expTraj length (or number of viewpoints): ", len(req.expTrajs))
        time1 = time.time() #rospy.get_time()
        time_taken_vis = 0 
        time_taken_others = 0
        for traj in req.expTrajs:
            #print("traj: ", traj)
            # Compute the info gain for each point in the trajectory
            vis_mask_tot = np.zeros((self.pred_map_class.shape[0], self.pred_map_class.shape[1])).astype(np.float32)
            #print("length of traj.trajectory: ", len(traj.trajectory))
            for query_point in traj.trajectory:
                #print("query_point: ", query_point.x, query_point.y)
                #print(int(500-5*query_point.y), int(5*query_point.x))
                time_before_vis = time.time()
                vis_ind, lidar_mask, inited_flood_grid, actual_hit_points = self.getvizmask.get_vis_mask(pred_occupancy_grid,
                                                    (int(1000-5*query_point.y), int(5*query_point.x)), 
                                                    laser_range=250, num_laser=50)
                time_taken_vis += time.time() - time_before_vis
                print("out loop: get_vis mask", time.time() - time_before_vis)
                #print()
                vis_mask_tot[vis_ind[:,0], vis_ind[:,1]] = 1

                #print("vis mask tot shape: ", vis_mask_tot.shape)
                #print("unknown area tensor shape: ", self.obs_map_unknown.shape)
                #print(np.max(vis_mask_tot))
                #print(np.max(self.obs_map_unknown))
                time_before_others = time.time()
                intersection_indices = np.where((vis_mask_tot == 0.0) & (self.obs_map_unknown == 0.5))

                #print("intersection indices: ", intersection_indices)
                intersection_area = np.zeros((vis_mask_tot.shape[0], vis_mask_tot.shape[1]))
                intersection_area[intersection_indices] = 1
                num_pix_vis_mask = np.sum(vis_mask_tot)
                plt_png_save = False
                if plt_png_save:
                    plt.clf()
                    plt_row = 1
                    plt_col = 3
                    plt.figure(figsize=(30,10))
                    plt.subplot(plt_row, plt_col, 1)
                    plt.imshow(self.obs_map_unknown)
                    plt.scatter(int(5*query_point.x), int(1000-5*query_point.y), c='r')
                    plt.title("observed_map_unknown_area")
                    
                    plt.subplot(plt_row, plt_col, 2)
                    plt.imshow(vis_mask_tot)
                    plt.scatter(int(5*query_point.x), int(1000-5*query_point.y), c='r')
                    plt.title("visbility mask, num pixels: {}".format(num_pix_vis_mask))
                    
                    plt.subplot(plt_row, plt_col, 3)
                    plt.imshow(intersection_area)
                    plt.scatter(int(5*query_point.x), int(1000-5*query_point.y), c='r')
                    plt.title("intersection mask")
                    plt.savefig('hi.png')
                # self.viz_flag = False
                #print("vis mask tot sum: ", np.sum(vis_mask_tot))
                #for i in range(vis_mask_tot.shape[0]):
                #    print(vis_mask_tot[i,:])
                #print("Checked a query point")
                num_pix_vis_intersect_mask = np.sum(intersection_area)
                time_taken_others += time.time() - time_before_others

            # Here need to sum over the vis_mask_tot and append value to the response
            variance_sum = num_pix_vis_intersect_mask
            #print("Computed for one viewpoint")
            values_vec.append(variance_sum)
        time2 = time.time() #rospy.get_time()
        print("time for service call is: ", (time2-time1))
        print("average time per frontier is: ", (time2-time1)/len(req.expTrajs))
        #print("avg. for vis mask : ", time_taken_vis / len(req.expTrajs))
        #print("avg. for othersk : ", time_taken_others / len(req.expTrajs))
        #print()
        #print("almost here")

        return TrajInfoGainResponse(values_vec)




if __name__ == "__main__":
    node = MapPropPredNode()
    rospy.spin()