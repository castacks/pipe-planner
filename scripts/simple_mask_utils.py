import numpy as np
from collections import deque
import math
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')  # Replace 'TkAgg' with your preferred backend, like 'Qt5Agg', 'GTK3Agg', etc.
# import matplotlib.pyplot as plt
from tqdm import tqdm
import os 
import cv2
from tqdm.contrib.concurrent import process_map  # or thread_map
import time
from skimage.measure import block_reduce
import signal
import scipy
from numba import jit, prange, njit
import range_libc
from shapely.geometry import Polygon, Point, MultiPolygon

# custom imports
import sys
sys.path.append('../')
from sim_utils import makePyOMap

class TimeoutException(Exception):   # Custom exception class
    pass
def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException
# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException
def get_random_plan_in_occgrid(occupancy_grid, min_dist_start_goal, point_intervals_m):
    # TODO: add to make_masklist_with_rand_traj_in_map
    ox, oy = np.where(occupancy_grid == 1)
    a_star = AStarPlanner(ox, oy,resolution=1)
    sx, sy = sample_free_position(occupancy_grid) # (row, col)
    gx, gy = sample_free_position(occupancy_grid)

    # ensure that start and end are not the same, and not too close
    while np.linalg.norm(np.array([sx, sy]) - np.array([gx, gy])) < min_dist_start_goal:
        sx, sy = sample_free_position(occupancy_grid) # (row, col)
        gx, gy = sample_free_position(occupancy_grid)
    
    # Get A* trajectory
    plan_x, plan_y = a_star.planning(sx, sy, gx, gy)
    plan_points = np.array([plan_x, plan_y]).T

    # Add points with decently-consistent intervals
    cumulative_distances = np.cumsum([np.linalg.norm(np.array(plan_points[i]) - np.array(plan_points[i-1])) for i in range(1, len(plan_points))])
    # Select the points that are collect_interval_m m apart
    interval = point_intervals_m
    selected_points = [plan_points[0]]
    for i in range(1, len(plan_points)):
        if cumulative_distances[i-1] >= interval:
            selected_points.append(plan_points[i])
            interval += point_intervals_m
    # add the last point 
    selected_points.append(selected_points[-1])
    selected_points = np.array(selected_points)
    
    # Reverse the plan so it starts at the start
    selected_points = selected_points[::-1]
    plan_dict = {}

    plan_dict['plan_points'] = selected_points
    plan_dict['start'] = np.array([sx, sy])
    plan_dict['goal'] = np.array([gx, gy])

    return plan_dict

def visualize_sampled_free_positions_given_buffer(occupancy_grid, validspace_map, buffer=10, num_samples=1000):
    free_positions = []
    for i in range(num_samples):
        free_positions.append(sample_free_position_given_buffer(occupancy_grid, validspace_map, buffer=buffer))
    free_positions = np.array(free_positions)
    plt.imshow(occupancy_grid, cmap='gray')
    plt.scatter(free_positions[:,1], free_positions[:,0], color='red', s=1)
    plt.title('Free positions sampled with buffer of {}'.format(buffer))
    plt.savefig('viz_free_positions_buffer_{}.png'.format(buffer))
    plt.show()

def sample_free_position_given_buffer(occupancy_grid, validspace_map, buffer=10):
    """
    Samples a position in free space from the given occupancy grid and validspace map (which defines space where a room is inside).
    Samples given a buffer, which defines how close the sampled position can be to an occupied cell.
    Do this by first getting a distance transform of the occupancy grid, then sampling a position from the distance transform that meets the conditions.
    """
    assert occupancy_grid.shape == validspace_map.shape
    dt = scipy.ndimage.distance_transform_cdt(occupancy_grid == 0) # distance from occupied cells

    free_after_buffer = dt > buffer # free cells after buffer

    # Find all the free positions in the occupancy grid
    free_positions = np.argwhere((free_after_buffer == 1) & (validspace_map == 1))

    # plt.imshow(dt)
    # plt.savefig('dt.png')
    # plt.scatter(free_positions[:,1], free_positions[:,0], color='red', s=1)
    # plt.show()
    # Randomly sample a free position
    if len(free_positions) > 0:
        index = np.random.choice(len(free_positions))
        row, col = free_positions[index]
        return (row, col)

    else:
        print("No free positions found with buffer of {}".format(buffer))
        return None
    
    

def sample_free_position(occupancy_grid):
    """
    Samples a position in free space from the given occupancy grid.

    Parameters:
    occupancy_grid (numpy.ndarray): A 2D numpy array representing the occupancy grid.

    Returns:
    tuple: A tuple (row, col) representing the row and column indices of the sampled position.
    """
    # Find all the free positions in the occupancy grid
    free_positions = np.argwhere(occupancy_grid == 0)

    # Randomly sample a free position
    if len(free_positions) > 0:
        index = np.random.choice(len(free_positions))
        row, col = free_positions[index]
        return (row, col)
    else:
        return None
    
@jit(nopython=True)
def bresenham(start, end):
    """
    Implementation of Bresenham's line drawing algorithm with Numba optimization.
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
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
    points = np.array(points)
    return points


def bresenham_nonumba(start, end):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
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
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points

def get_free_points_no_numba(occupancy_grid, robot_pos, laser_range=10, num_laser=100):
    """
    Assumes circular lidar
    occupancy_grid: np.array (h x w)
    robot_pos: (x, y)

    Outputs: 
    free_points: np.array of hit points (x, y)
    """

    free_points = []
    hit_points = [] # actual hit points + last bresenham point (for some reason need this for flodding)
    actual_hit_points = [] # 
    for orientation in np.linspace(0, 2*np.pi, num_laser):
        end_point = (int(robot_pos[0] + laser_range * np.cos(orientation)), int(robot_pos[1] + laser_range * np.sin(orientation)))
        
        # Get index along ray to check
        bresenham_points = (bresenham_nonumba(robot_pos, end_point))

        # Go through the points and see the first hit
        # TODO: do a check if any first?
        for i in range(len(bresenham_points)):
            # if bresenham point is in the map 
            if bresenham_points[i,0] < 0 or bresenham_points[i,0] >= occupancy_grid.shape[0] or bresenham_points[i,1] < 0 or bresenham_points[i,1] >= occupancy_grid.shape[1]:
                if i != 0:
                    hit_points.append(bresenham_points[i-1])
                break # don't use this bresenham point 
            
            if occupancy_grid[bresenham_points[i,0], bresenham_points[i,1]] == 1: # hit if it is void or occupied #! THINK IF THIS IS A GOOD ASSUMPTION
                actual_hit_points.append(bresenham_points[i])
                hit_points.append(bresenham_points[i])
                
                break
            else: # no hits
                free_point = bresenham_points[i]
                free_points.append(free_point)

                if i == len(bresenham_points) - 1:
                    hit_points.append(end_point) # need to add this for proper flooding for vis mask
                    break
                
    
    # Convert to np.array
    free_points = np.array(free_points)
    hit_points = np.array(hit_points)
    actual_hit_points = np.array(actual_hit_points)

    # To match the numba version, for test case checking
    if len(free_points) == 0:
        free_points = np.empty((0, 2), dtype=np.int64)
    if len(hit_points) == 0:
        hit_points = np.empty((0, 2), dtype=np.int64)
    if len(actual_hit_points) == 0:
        actual_hit_points = np.empty((0, 2), dtype=np.int64)
    return free_points, hit_points, actual_hit_points

def get_free_points_range_libc(occ_grid_numpy, robot_pos, laser_range=10, num_laser=100, occ_map_type='numpy', occ_map_obj=None):
    # Create a PyOMap object from the occupancy grid (occ_map)
    if occ_map_type == 'numpy': # if occ_map is a numpy array, convert to PyOMap
        occ_map_load_time = time.time()
        print("Get free points: loading occ map...")
        occ_map_obj = makePyOMap(occ_grid_numpy)
        
        print("Get free points: Occ map load time", time.time()-occ_map_load_time)
    elif occ_map_type == 'PyOMap':
        occ_map_obj = occ_map_obj
    else:
        raise ValueError("occ_map_type must be 'numpy' or 'PyOMap'")

    # Create a PyBresenhamsLine object
    calc_range_time = time.time()
    bl = range_libc.PyBresenhamsLine(occ_map_obj, laser_range)

    # Create queries for rays
    queries = np.zeros((num_laser, 3), dtype=np.float32)
    queries[:, 0] = robot_pos[0]
    queries[:, 1] = robot_pos[1]
    queries[:, 2] = np.linspace(0, 2 * np.pi, num_laser)

    # Calculate ranges using range_libc
    ranges = np.zeros(num_laser, dtype=np.float32)
    
    # print("calc single range", bl.calc_range(robot_pos[0], robot_pos[1], 1.568817138671875))
    # single_query = np.array([[robot_pos[0], robot_pos[1], 1.568817138671875], [robot_pos[0], robot_pos[1], 0]], dtype=np.float32)
    # single_ranges = np.zeros(2, dtype=np.float32)
    # bl.calc_range_many(single_query, single_ranges)
    # print("single_ranges", single_ranges)
    # import pdb; pdb.set_trace()
    bl.calc_range_many(queries, ranges)
    # print("Get free points: Calc range time", time.time()-calc_range_time)
    
    # Get points from ranges
    free_points, hit_points, actual_hit_points = get_points_pos_from_ranges(ranges, queries, num_laser, robot_pos, laser_range, occ_grid_numpy)
    return free_points, hit_points, actual_hit_points

@jit(nopython=True)
def get_points_pos_from_ranges(ranges, queries, num_laser, robot_pos, laser_range, occupancy_grid):
    # Initialize arrays for free points, hit points, and actual hit points
    max_points = num_laser 
    free_points = np.empty((max_points, 2), dtype=np.int64)
    hit_points = np.empty((num_laser, 2), dtype=np.int64)
    actual_hit_points = np.empty((num_laser, 2), dtype=np.int64)

    free_idx, hit_idx, actual_hit_idx = 0, 0, 0

    for i in range(num_laser):
        range_value = ranges[i] 
        max_range_for_laser = laser_range
        angle = queries[i, 2]

        # Calculate endpoint of the ray
        end_point = (round(robot_pos[0] + range_value * np.cos(angle)),
                     round(robot_pos[1] + range_value * np.sin(angle)))
         
        # Check if the endpoint is out of bounds
        while end_point[0] < 1 or end_point[0] >= occupancy_grid.shape[0]-1 or end_point[1] < 1 or end_point[1] >= occupancy_grid.shape[1] - 1:
            # decrement range by 1
            range_value -= 1
            max_range_for_laser -= 1
            end_point = (int(robot_pos[0] + range_value * np.cos(angle)), int(robot_pos[1] + range_value * np.sin(angle)))
        # Append the endpoint to the appropriate list
        if range_value == max_range_for_laser: # no hit, and in bounds
            hit_points[hit_idx] = end_point
            hit_idx += 1
        else:
            hit_points[hit_idx] = end_point # got a lidar measurement, and in boudns
            hit_idx += 1
            if occupancy_grid[end_point[0], end_point[1]] == 1: # check if it stoppped because of hitting obs
                actual_hit_points[actual_hit_idx] = end_point
                actual_hit_idx += 1
        # print("max_range_for_laser", max_range_for_laser)

    # Trim arrays to actual size
    free_points = free_points[:free_idx]
    hit_points = hit_points[:hit_idx]
    actual_hit_points = actual_hit_points[:actual_hit_idx]


    return free_points, hit_points, actual_hit_points

@jit(nopython=True)
def get_bresenham_for_all_endpoints(robot_pos, end_points):
    bresenham_points_list = []
    for end_point in end_points:
        bresenham_points = bresenham(robot_pos, end_point)
        bresenham_points_list.append(bresenham_points)
    return bresenham_points_list

@jit(nopython=True)
def get_free_points(occupancy_grid, robot_pos, laser_range=10, num_laser=100, use_prob=False, hit_prob_thresh=1.0):
    assert hit_prob_thresh <= 1.0   
    if use_prob == False and hit_prob_thresh < 1.0:
        raise ValueError("If use_prob is False, hit_prob_thresh must be 1.0")
        
    # Preallocate arrays for maximum possible size
    max_points = num_laser * laser_range
    free_points = np.empty((max_points, 2), dtype=np.int64)
    hit_points = np.empty((num_laser, 2), dtype=np.int64)
    actual_hit_points = np.empty((num_laser, 2), dtype=np.int64)

    free_idx, hit_idx, actual_hit_idx = 0, 0, 0

    orientation = np.linspace(0, 2*np.pi, num_laser)
    end_points = np.zeros((num_laser, 2), dtype=np.int64)
    end_points[:, 0] = robot_pos[0] + laser_range * np.cos(orientation)
    end_points[:, 1] = robot_pos[1] + laser_range * np.sin(orientation)
    
    bresenham_points_list = get_bresenham_for_all_endpoints(robot_pos, end_points)
    for endpoint_i, end_point in enumerate(end_points):
        
        bresenham_points = bresenham_points_list[endpoint_i]

        for i in range(len(bresenham_points)):
            accum_hit_prob = 0
            x, y = bresenham_points[i]
            if x < 1 or x >= occupancy_grid.shape[0]-1 or y < 1 or y >= occupancy_grid.shape[1]-1:
                if i != 0:
                    hit_points[hit_idx] = bresenham_points[i-1]
                    hit_idx += 1
                break 
            
            # Define break condition 
            should_break = False
            if use_prob:
                accum_hit_prob += occupancy_grid[x, y]
                if accum_hit_prob >= hit_prob_thresh:
                    should_break = True
            else:
                should_break = (occupancy_grid[x, y] == 1)
                
            if should_break:
                actual_hit_points[actual_hit_idx] = bresenham_points[i]
                actual_hit_idx += 1
                hit_points[hit_idx] = bresenham_points[i]
                hit_idx += 1
                break
            else:
                free_points[free_idx] = bresenham_points[i]
                free_idx += 1

                if i == len(bresenham_points) - 1:
                    hit_points[hit_idx] = end_point
                    hit_idx += 1
                    break

    # Trim arrays to actual size
    free_points = free_points[:free_idx]
    hit_points = hit_points[:hit_idx]
    actual_hit_points = actual_hit_points[:actual_hit_idx]

    return free_points, hit_points, actual_hit_points


@njit
def flood_fill_simple(center_point, occupancy_map):
    """
    center_point: starting point (x,y) of fill
    occupancy_map: occupancy map generated from Bresenham ray-tracing
    """
    # Fill empty areas with list method (as a stack)
    occupancy_map = np.copy(occupancy_map)
    sx, sy = occupancy_map.shape
    fringe = []
    fringe.append(center_point)
    while fringe:
        n = fringe.pop()
        nx, ny = n
        unknown_val = 0.5
        # West
        if nx > 0:
            if occupancy_map[nx - 1, ny] == unknown_val:
                occupancy_map[nx - 1, ny] = 0
                fringe.append((nx - 1, ny))
        # East
        if nx < sx - 1:
            if occupancy_map[nx + 1, ny] == unknown_val:
                occupancy_map[nx + 1, ny] = 0
                fringe.append((nx + 1, ny))
        # North
        if ny > 0:
            if occupancy_map[nx, ny - 1] == unknown_val:
                occupancy_map[nx, ny - 1] = 0
                fringe.append((nx, ny - 1))
        # South
        if ny < sy - 1:
            if occupancy_map[nx, ny + 1] == unknown_val:
                occupancy_map[nx, ny + 1] = 0
                fringe.append((nx, ny + 1))
    return occupancy_map


def flood_fill_simple_nonumba(center_point, occupancy_map):
    """
    center_point: starting point (x,y) of fill
    occupancy_map: occupancy map generated from Bresenham ray-tracing
    """
    # Fill empty areas with queue method
    occupancy_map = np.copy(occupancy_map)
    sx, sy = occupancy_map.shape
    fringe = deque()
    fringe.appendleft(center_point)
    while fringe:
        
        n = fringe.pop()
        nx, ny = n
        unknown_val = 0.5
        # West
        if nx > 0:
            if occupancy_map[nx - 1, ny] == unknown_val:
                occupancy_map[nx - 1, ny] = 0
                fringe.appendleft((nx - 1, ny))
        # East
        if nx < sx - 1:
            if occupancy_map[nx + 1, ny] == unknown_val:
                occupancy_map[nx + 1, ny] = 0
                fringe.appendleft((nx + 1, ny))
        # North
        if ny > 0:
            if occupancy_map[nx, ny - 1] == unknown_val:
                occupancy_map[nx, ny - 1] = 0
                fringe.appendleft((nx, ny - 1))
        # South
        if ny < sy - 1:
            if occupancy_map[nx, ny + 1] == unknown_val:
                occupancy_map[nx, ny + 1] = 0
                fringe.appendleft((nx, ny + 1))
    return occupancy_map


def init_flood_fill(robot_pos, obstacle_points, occ_grid_shape):
    """
    center_point: center point
    obstacle_points: detected obstacles points (x,y)
    xy_points: (x,y) point pairs
    """
    # center_x, center_y = robot_pos
    # prev_ix, prev_iy = center_x - 1, center_y
    occupancy_map = (np.ones(occ_grid_shape)) * 0.5
    # append first obstacle point to last
    obstacle_points = np.vstack((obstacle_points, obstacle_points[0]))
    prev_ix = obstacle_points[0,0]
    prev_iy = obstacle_points[0,1]
    for (x, y) in zip(obstacle_points[:,0], obstacle_points[:,1]):
        # x coordinate of the the occupied area
        ix = int(x)
        # y coordinate of the the occupied area
        iy = int(y)
        free_area = bresenham((prev_ix, prev_iy), (ix, iy))
        for fa in free_area:
            # print(fa, (prev_ix, prev_iy), (ix, iy))
            occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
        prev_ix = ix
        prev_iy = iy
    # grid_xmin, grid_xmax = (0, occ_grid_shape[1])
    # grid_ymin, grid_ymax = (0, occ_grid_shape[0])
    # plt.imshow(occupancy_map, origin='lower', extent=[grid_xmin, grid_xmax, grid_ymin, grid_ymax])
    # plt.scatter(obstacle_points[:,1], obstacle_points[:,0], c='r')
    # plt.scatter(obstacle_points[0,1], obstacle_points[0,0], c='g')
    # plt.title('Red: Obstacle points, Green: first obstacle poitn')
    # plt.show()
    return occupancy_map

show_animation = False

class AStarPlanner:

    def __init__(self, ox, oy, resolution, occ_grid):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.min_x = 0
        self.min_y = 0
        self.max_x = occ_grid.shape[0] - 1
        self.max_y = occ_grid.shape[1] - 1

        self.x_width = int(round((self.max_x - self.min_x) / self.resolution)) + 1
        self.y_width = int(round((self.max_y - self.min_y) / self.resolution)) + 1
        self.obstacle_map = None
        self.motion = self.get_motion_model()
        start_time = time.time()
        self.calc_obstacle_map(ox, oy)
        # print("Time to calc obstacle map: ", np.round(time.time() - start_time, 3), "s")
        # import pdb; pdb.set_trace()
        self.occ_grid = occ_grid
        floor_val = 10 
        # TODO: don't put unknown as occupied
        self.distance_transform = (np.clip(scipy.ndimage.distance_transform_cdt(self.occ_grid == 0) * -1 , -floor_val, 0) + floor_val) * 0
        # import pdb; pdb.set_trace()
        # plt.subplot(1,2,1)
        # plt.imshow(self.occ_grid)
        # plt.subplot(1,2,2)
        # plt.imshow(self.distance_transform)
        # plt.show()

        # print('initialized')

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        # signal.alarm(2)  # Set an alarm for 10 seconds

        # try: 
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                            self.calc_xy_index(sy, self.min_y), self.distance_transform[sx, sy], -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                            self.calc_xy_index(gy, self.min_y), self.distance_transform[gx, gy], -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                    open_set[
                                                                        o]))
            current = open_set[c_id]

            # show graph
            show_animation = False
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                        self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                            lambda event: [exit(
                                                0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            # if current.x == goal_node.x and current.y == goal_node.y:
            goal_tolerance = 1
            if abs(current.x - goal_node.x) <= goal_tolerance and abs(current.y - goal_node.y) <= goal_tolerance:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                current.y + self.motion[i][1],
                                current.cost + self.motion[i][2] + 
                                self.distance_transform[current.x + self.motion[i][0], current.y + self.motion[i][1]],
                                c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry
        # except TimeoutException:
        #     print("A* timed out!")
        #     return [], []

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        # print("checking: ", node.x, node.y)
        if self.obstacle_map[int(node.x)][int(node.y)]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        # print("Time to calc min/max x/y: ", np.round(time.time() - start_time, 3), "s")
        # Use a NumPy array for the obstacle map
        start_time = time.time()
        self.obstacle_map = np.full((self.x_width, self.y_width), False)
        # print("Time to make full obstacle map: ", np.round(time.time() - start_time, 3), "s")

        # Use NumPy's advanced indexing to set the obstacle positions
        start_time = time.time()
        self.obstacle_map[ox, oy] = True
        # print("Time to set obstacle map: ", np.round(time.time() - start_time, 3), "s")
        # self.min_x = round(min(ox))
        # self.min_y = round(min(oy))
        # self.max_x = round(max(ox))
        # self.max_y = round(max(oy))
        # # print("min_x:", self.min_x)
        # # print("min_y:", self.min_y)
        # # print("max_x:", self.max_x)
        # # print("max_y:", self.max_y)

        # self.x_width = int(round((self.max_x - self.min_x) / self.resolution)) + 1
        # self.y_width = int(round((self.max_y - self.min_y) / self.resolution)) + 1
        # # print("x_width:", self.x_width)
        # # print("y_width:", self.y_width)

        # # obstacle map generation
        # # print("Generating obstacle map ...")
        # self.obstacle_map = [[False for _ in range(self.y_width)]
        #                      for _ in range(self.x_width)]
        
        
        # for i in range(len(ox)):
        #     self.obstacle_map[ox[i]][oy[i]] = True


    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        multiplier = 1
        motion = [[1*multiplier, 0*multiplier, 1],
                  [0*multiplier, 1*multiplier, 1],
                  [-1*multiplier, 0*multiplier, 1],
                  [0*multiplier, -1*multiplier, 1],
                  [-1*multiplier, -1*multiplier, math.sqrt(2)],
                  [-1*multiplier, 1*multiplier, math.sqrt(2)],
                  [1*multiplier, -1*multiplier, math.sqrt(2)],
                  [1*multiplier, 1*multiplier, math.sqrt(2)]]
        

        return motion
    
def get_hit_ponits(occupancy_grid, robot_pos, laser_range=50, num_laser=100, raycast_mode='range_libc', occ_map_type='numpy', occ_map_obj=None, hit_prob_threshold=1.0):
    """
        mode: 'simple', 'range_libc', 'probabilistic'
        simple: uses our own implementation of raycasting with numba 
        range_libc: uses range_libc for raycasting
        probabilistic: uses probabilistic raycasting, where we raycast they ray has accumulated up to a hit probobility
    Since it is a custom raycast implementation, we will use our numba implementation for easy development. (Mainly changed is get_free_points)
        Flood fill is the same for all.
    """
    #! should return vis_mask, or rename to get_vis_ind
    if hit_prob_threshold != 1.0 and raycast_mode != 'probabilistic':
        raise ValueError("hit_prob_threshold only works probabilistic raycast_mode, currently {}".format(raycast_mode))
    
     #! If robot_pos is occupied (in obstacle), then all points are not visible 
    if occupancy_grid[robot_pos[0], robot_pos[1]] == 1:
        #print("Robot pos is occupied")
        return np.ones_like(occupancy_grid) * 0.5, np.empty((0, 2), dtype=np.int64)

    if raycast_mode == 'simple':
        # print("using numba, pause here to confirm. If confirmed, delete this line")
        # import pdb; pdb.set_trace()
        
        free_points, hit_points, actual_hit_points = get_free_points(occupancy_grid, robot_pos, laser_range=laser_range, num_laser=num_laser)
        # print("Time for free points (numba):", time.time()-start_time)
        # print("Hit points:", hit_points)
        # print("Actual hit points:", actual_hit_points)

    elif raycast_mode == 'range_libc': 
        # print('Getting free points (range_libc)')
        free_points, hit_points, actual_hit_points = get_free_points_range_libc(occupancy_grid, robot_pos, laser_range=laser_range, num_laser=num_laser, occ_map_type=occ_map_type, occ_map_obj=occ_map_obj)
        # print("Time for free points (range_libc):", time.time()-start_time)
        # print("Hit points:", hit_points)
        # print("Actual hit points:", actual_hit_points)
    elif raycast_mode == 'probabilistic':
        free_points, hit_points, actual_hit_points = get_free_points(occupancy_grid, robot_pos, laser_range, num_laser, use_prob=True, hit_prob_thresh=hit_prob_threshold)

    #! if all hit points are the same, then no points are visible
    if np.all(hit_points == hit_points[0]):
        #print("All hit points are the same")
        return np.ones_like(occupancy_grid) * 0.5, np.empty((0, 2), dtype=np.int64)
    
    new_occ_grid = np.ones_like(occupancy_grid) * 0.5
    if len(free_points.shape) == 2:
        new_occ_grid[free_points[:,0], free_points[:,1]] = 0 # free points
    new_occ_grid[hit_points[:,0], hit_points[:,1]] = 1 # hit points
    

    return new_occ_grid, hit_points
    

def get_vis_mask(occupancy_grid, robot_pos, laser_range=50, num_laser=100, raycast_mode='range_libc', occ_map_type='numpy', occ_map_obj=None, hit_prob_threshold=1.0, skip_raycast=False):
    """
        mode: 'simple', 'range_libc', 'probabilistic'
        simple: uses our own implementation of raycasting with numba 
        range_libc: uses range_libc for raycasting
        probabilistic: uses probabilistic raycasting, where we raycast they ray has accumulated up to a hit probobility
    Since it is a custom raycast implementation, we will use our numba implementation for easy development. (Mainly changed is get_free_points)
    
    
        Flood fill is the same for all.
    """
    #! should return vis_mask, or rename to get_vis_ind
    if hit_prob_threshold != 1.0 and raycast_mode != 'probabilistic':
        raise ValueError("hit_prob_threshold only works probabilistic raycast_mode, currently {}".format(raycast_mode))
    
    start_time = time.time()
    if raycast_mode == 'simple':
        free_points, hit_points, actual_hit_points = get_free_points(occupancy_grid, robot_pos, laser_range=laser_range, num_laser=num_laser)

    elif raycast_mode == 'range_libc': 
        free_points, hit_points, actual_hit_points = get_free_points_range_libc(occupancy_grid, robot_pos, laser_range=laser_range, num_laser=num_laser, occ_map_type=occ_map_type, occ_map_obj=occ_map_obj)
    elif raycast_mode == 'probabilistic':
        free_points, hit_points, actual_hit_points = get_free_points(occupancy_grid, robot_pos, laser_range, num_laser, use_prob=True, hit_prob_thresh=hit_prob_threshold)
    else:
        raise ValueError("raycast_mode must be 'simple' or 'range_libc'")

    #! If robot_pos is occupied (in obstacle), then all points are not visible 
    if occupancy_grid[robot_pos[0], robot_pos[1]] == 1:
        print("Robot pos is occupied")
        return np.empty((0, 2), dtype=np.int64), np.ones_like(occupancy_grid) * 0.5, np.ones_like(occupancy_grid) * 0.5, np.empty((0, 2), dtype=np.int64), np.ones_like(occupancy_grid) * 0.5
    
    #! if all hit points are the same, then no points are visible
    if len(np.unique(hit_points, axis=0)) == 1:
        print("All hit points are the same")
        return np.empty((0, 2), dtype=np.int64), np.ones_like(occupancy_grid) * 0.5, np.ones_like(occupancy_grid) * 0.5, np.empty((0, 2), dtype=np.int64), np.ones_like(occupancy_grid) * 0.5

    new_occ_grid = np.ones_like(occupancy_grid) * 0.5
    if len(free_points.shape) == 2:
        new_occ_grid[free_points[:,0], free_points[:,1]] = 0 # free points
    new_occ_grid[hit_points[:,0], hit_points[:,1]] = 1 # hit points
    


    # # Expand the init flood mask by 1 to avoid internal flood fill boundaries that cause incorrect vis masks
    # # Create a Polygon from the boundary points
    polygon_shape = Polygon(hit_points)

    # Expand the boundary by 1 unit
    expanded_shape = polygon_shape.buffer(1)

    # If you need the boundary points of the expanded shape
    if type(expanded_shape) == MultiPolygon:
        import pdb; pdb.set_trace()
    expanded_boundary_points = np.array(expanded_shape.exterior.coords).astype(int)
    
    if skip_raycast:
        inited_flood_grid = np.ones_like(occupancy_grid) * 0.5 # all unknown
        inited_flood_grid[occupancy_grid == 1] = 0 # occupuied is 0 for flood grid
    else:
        inited_flood_grid = init_flood_fill(robot_pos, expanded_boundary_points, new_occ_grid.shape)
    inited_boundary_points = np.argwhere(inited_flood_grid == 0)


    # Check if robot_pos is already marked as free area, then repick a new seed point 
    seed_point = robot_pos
    while inited_flood_grid[seed_point[0], seed_point[1]] == 0:
        print("Robot pos is in boundary of init_flood_mask, repicking seed point")
        polygon = Polygon(hit_points)
        interior_point = polygon.representative_point()
        seed_point = (int(interior_point.x), int(interior_point.y))
    flooded_grid = flood_fill_simple(seed_point, inited_flood_grid) #previous
    flooded_grid[inited_boundary_points[:,0], inited_boundary_points[:,1]] = 0.5 # set boundary to 0.5
    vis_ind = np.argwhere(flooded_grid == 0)
    return vis_ind, new_occ_grid, inited_flood_grid, actual_hit_points, flooded_grid

def make_data_folders(data_output_folder_name):
    module_path = os.path.dirname(os.path.realpath(__file__))
    output_path_dict = {}

    for folder_name in ['train','test']:
        data_map_output_folder_path = os.path.join(data_output_folder_name,folder_name, 'global_map')
        data_mask_output_folder_path = os.path.join(data_output_folder_name,folder_name,'global_mask')
        data_ego_map_output_folder_path = os.path.join(data_output_folder_name,folder_name,'ego_map')
        data_ego_mask_output_folder_path = os.path.join( data_output_folder_name,folder_name,'ego_mask')
        data_viz_output_folder_path = os.path.join(data_output_folder_name,folder_name,'viz')
        os.makedirs(data_map_output_folder_path, exist_ok=True)
        os.makedirs(data_mask_output_folder_path, exist_ok=True)
        os.makedirs(data_viz_output_folder_path, exist_ok=True)
        os.makedirs(data_ego_map_output_folder_path, exist_ok=True)
        os.makedirs(data_ego_mask_output_folder_path, exist_ok=True)

        

        output_path_dict[folder_name+'_global_map_folder_path'] = data_map_output_folder_path
        output_path_dict[folder_name+'_global_mask_folder_path'] = data_mask_output_folder_path
        output_path_dict[folder_name+'_ego_map_folder_path'] = data_ego_map_output_folder_path
        output_path_dict[folder_name+'_ego_mask_folder_path'] = data_ego_mask_output_folder_path
        output_path_dict[folder_name+'_viz_folder_path'] = data_viz_output_folder_path
    
    # print("Created data folders at:", d[ata_map_output_folder_path, data_mask_output_folder_path, data_viz_output_folder_path)
    print("output_path_dict:", output_path_dict)
    return output_path_dict


def convert_01_single_channel_to_0_255_3_channel(img):
    img = np.stack([img, img, img], axis=2)
    img *= 255
    return img

def convert_012_labels_to_maskutils_labels(occ_map):
    """Converts occupancy map that is of 012 label (0: unknown, 1: occupied, 2: free) 
    to maskutils labels (0: free, 0.5: unknown, 1: occupied)"""
    # TODO: ideally this should not be needed by moving mask_utils to using 012 labels
    occ_map_copy = np.zeros_like(occ_map).astype(np.float32)
    occ_map_copy[occ_map == 2] = 0
    occ_map_copy[occ_map == 1] = 1
    occ_map_copy[occ_map == 0] = 0.5
    return occ_map_copy

def make_kth_dataset(map_configs, output_path_dict, num_maps, max_workers):
    """
    Make folders of KTH buildings and masks for training and testing.

    Num_maps: number of maps to generate with. We choose randomly between the maps, so some maps may repeat.
    """

    pass 
    # for map_i in tqdm(range(num_maps), desc='Generating toy building masks. # maps: '):
    #     make_kth_dataset_onemap([map_configs, output_path_dict, map_i]) 
    # # r = process_map(make_toy_building_dataset_onemap, [(map_configs, output_path_dict, map_i) for map_i in range(num_maps)], max_workers=max_workers)

def make_kth_mask_onemap(pgm_path, map_configs, output_path_dict, map_i):
    """
    Make visibility training data masks from a single pgm image generated from the KTH dataset. 
    """

    # Open pgm image and convert to occupancy map 
    pgm_img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)

    # Convert to occupancy map (0: unknown, 1: occupied, 2: free)
    kth_occ_map = np.zeros_like(pgm_img)
    kth_occ_map[pgm_img == 0] = 1 # occupied
    kth_occ_map[pgm_img > 0] = 2 # free
    kth_orig_occ_map = kth_occ_map.copy()

    # Make the image 10x smaller for faster planning, when interpolating choose the cell with lower number (occupied)
    kth_occ_map = block_reduce(kth_occ_map, block_size=(5, 5), func=np.min, cval=1)
    # TODO: check if cval=1 is reasonable

    assert np.min(kth_occ_map) == 1, "kth_occ_map should be 1 (occupied) or 2 (free). There is unknown..."
    # import pdb; pdb.set_trace()
    # plt.subplot(1,2,1)
    # plt.imshow(kth_orig_occ_map)
    # plt.subplot(1,2,2)
    # plt.imshow(kth_occ_map)
    # plt.title('Resized ')
    # plt.show()
    # Convert occ_map to what is needed for mask_utils
    # before: (0: unknown, 1: occupied, 2: free)
    # after: (0: free, 0.5: unknown, 1: occupied)
    kth_occ_map = convert_012_labels_to_maskutils_labels(kth_occ_map)

    train_mask_count = 0
    test_mask_count = 0
    # Decide if train or test given probability
    if np.random.rand() < map_configs['percent_test']:
        train_or_test = 'test'
    else:
        # choose a random map from train map paths
        train_or_test = 'train'

    # make mask list
    map, mask_list, local_mask_list, local_gt_list, pose_list = make_masklist_with_rand_traj_in_map(kth_occ_map, map_configs)
    map = 1 - map # invert map so that free space is white
    map = np.stack([map, map, map], axis=2)
    map *= 255 
    for mask_i, mask in enumerate(mask_list):

        # make outputs 3 channels
        
        mask = np.stack([mask, mask, mask], axis=2)
        local_mask = np.stack([local_mask_list[mask_i], local_mask_list[mask_i], local_mask_list[mask_i]], axis=2)  
        local_gt = np.stack([local_gt_list[mask_i], local_gt_list[mask_i], local_gt_list[mask_i]], axis=2)
        mask *= 255
        local_mask *= 255
        local_gt *= 255

        # saving
        if train_or_test == 'train':
            map_path = os.path.join(output_path_dict[train_or_test+'_global_map_folder_path'], '{:06d}_{:06d}.png'.format(map_i, train_mask_count))
            mask_path = os.path.join(output_path_dict[train_or_test+'_global_mask_folder_path'], '{:06d}_{:06d}.png'.format(map_i, train_mask_count))
            ego_map_path = os.path.join(output_path_dict[train_or_test+'_ego_map_folder_path'], '{:06d}_{:06d}.png'.format(map_i, train_mask_count))
            ego_mask_path = os.path.join(output_path_dict[train_or_test+'_ego_mask_folder_path'],'{:06d}_{:06d}.png'.format(map_i, train_mask_count))
            # plt.savefig(os.path.join(output_path_dict[train_or_test+'_viz_folder_path'], '{:06d}.png'.format(train_mask_count)))
            train_mask_count += 1
        else:
            map_path = os.path.join(output_path_dict[train_or_test+'_global_map_folder_path'], '{:06d}_{:06d}.png'.format(map_i, test_mask_count))
            mask_path = os.path.join(output_path_dict[train_or_test+'_global_mask_folder_path'], '{:06d}_{:06d}.png'.format(map_i, test_mask_count))
            ego_map_path = os.path.join(output_path_dict[train_or_test+'_ego_map_folder_path'], '{:06d}_{:06d}.png'.format(map_i, test_mask_count))
            ego_mask_path = os.path.join(output_path_dict[train_or_test+'_ego_mask_folder_path'], '{:06d}_{:06d}.png'.format(map_i, test_mask_count))
            # plt.savefig(os.path.join(output_path_dict[train_or_test+'_viz_folder_path'], '{:06d}.png'.format(test_mask_count)))
            test_mask_count += 1
        
        cv2.imwrite(map_path, map)
        cv2.imwrite(mask_path, mask)   
        cv2.imwrite(ego_map_path, local_gt)
        cv2.imwrite(ego_mask_path, local_mask)
        plt.close()

def crop_around_point(image, center, crop_size):
    """
    Returns a cropped image centered around a given point.

    Args:
        image (numpy.ndarray): The input image.
        center (tuple): The center point (x, y).
        crop_size (tuple): The size of the cropped image (width, height).

    Returns:
        numpy.ndarray: The cropped image.
    """
    x, y = center
    w, h = crop_size

    pad_value = 0.5
    # Calculate the top-left and bottom-right corners of the crop.
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    # Pad the crop region with zeros if it extends beyond the bounds of the input image.
    if x1 < 0:
        pad_left = abs(x1)
        x1 = 0
    else:
        pad_left = 0

    if x2 > image.shape[1]:
        pad_right = x2 - image.shape[1]
        x2 = image.shape[1]
    else:
        pad_right = 0

    if y1 < 0:
        pad_top = abs(y1)
        y1 = 0
    else:
        pad_top = 0

    if y2 > image.shape[0]:
        pad_bottom = y2 - image.shape[0]
        y2 = image.shape[0]
    else:
        pad_bottom = 0

    crop = image[y1:y2, x1:x2]
    crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0.5)

    return crop



def make_masklist_with_rand_traj_in_map(occupancy_grid, map_configs, show_viz=False):
    ox, oy = np.where(occupancy_grid == 1)
    a_star = AStarPlanner(ox, oy,resolution=1)

    mask_list = []
    local_gt_list = []
    local_mask_list = []
    pose_list = []

    grid_xmin, grid_xmax = (0, occupancy_grid.shape[1])
    grid_ymin, grid_ymax = (0, occupancy_grid.shape[0])
    extent = [grid_xmin, grid_xmax, grid_ymin, grid_ymax]

    for traj_i in range(map_configs['num_rand_traj_per_map']):
        # make random start and end points
        # print(np.unique(occupancy_grid))
        sx, sy = sample_free_position(occupancy_grid) # (row, col)
        gx, gy = sample_free_position(occupancy_grid)

        # ensure that start and end are not the same, and not too close
        while np.linalg.norm(np.array([sx, sy]) - np.array([gx, gy])) < map_configs['min_start_end_dist']:
            print('start and end too close, resampling')
            gx, gy = sample_free_position(occupancy_grid)
        # print('start: {}, end: {}'.format((sx, sy), (gx, gy)))

        # gy = 1050
        # gx = 1100
        # import pdb; pdb.set_trace()
        # Get A* trajectory
        print('Getting A* trajectory')
        plan_x, plan_y = a_star.planning(sx, sy, gx, gy)
        print('Found A* Trajectory')
        plan_points = np.array([plan_x, plan_y]).T

        # For loop: Get vis mask for every traveled distance of 5m
        vis_mask_tot = np.ones_like(occupancy_grid) * 0.5
        cumulative_distances = np.cumsum([np.linalg.norm(np.array(plan_points[i]) - np.array(plan_points[i-1])) for i in range(1, len(plan_points))])
        # Select the points that are collect_interval_m m apart
        interval = map_configs['collect_interval_m']
        selected_points = [plan_points[0]]
        for i in range(1, len(plan_points)):
            if cumulative_distances[i-1] >= interval:
                selected_points.append(plan_points[i])
                interval += map_configs['collect_interval_m']
        selected_points = np.array(selected_points)

        # At selected points, get vis mask and combine
        
        # initialize np.array of 2 col as accum_hit_points
        accum_hit_points = np.zeros((0, 2)).astype(int)
        for query_point in selected_points:
            vis_ind, lidar_mask, inited_flood_grid, actual_hit_points, flooded_grid = get_vis_mask(occupancy_grid,
                                                (int(query_point[0]), int(query_point[1])), 
                                                laser_range=map_configs['laser_range'], num_laser=map_configs['num_laser'])

            # Compile into total visibility mask 
            # should start with unknown

            # Add hit points to accum_hit_points
            # print('accum_hit_points: {}'.format(accum_hit_points.shape))
            # print('actual_hit_points: {}'.format(actual_hit_points.shape))
            if actual_hit_points.shape[0] == 0:
                accum_hit_points = actual_hit_points
            else:
                accum_hit_points = np.vstack((accum_hit_points, actual_hit_points))
            # Add to total visibility mask
            # print('vis_ind: {}'.format(vis_ind.shape))
            
            vis_mask_tot[vis_ind[:,0], vis_ind[:,1]] = 1 # update free
            vis_mask_tot[accum_hit_points[:,0], accum_hit_points[:,1]] = 0 # update occupied cells

            # plt.imshow(vis_mask_tot)
            # plt.scatter(query_point[1], query_point[0], c='r')
            # plt.savefig('hi.png')
            # plt.close()
            # print('local map size: {}'.format(map_configs['local_map_size']))
            local_mask = crop_around_point(vis_mask_tot, (query_point[1], query_point[0]), (map_configs['local_map_size'],map_configs['local_map_size']))  
            crop_pic = crop_around_point(1-occupancy_grid, (query_point[1], query_point[0]), (map_configs['local_map_size'],map_configs['local_map_size']))  
            # plt.imshow(crop_pic)
            # plt.scatter(crop_pic.shape[0]/2, crop_pic.shape[1]/2, c='r')
            # plt.savefig('hi2.png')
            # plt.close()
            # import pdb; pdb.set_trace()
            mask_list.append(np.copy(vis_mask_tot))
            local_gt_list.append(np.copy(crop_pic))
            local_mask_list.append(np.copy(local_mask))
            pose_list.append(np.copy(query_point))
            # import pdb; pdb.set_trace()

            # # # Visualization
            if show_viz:

                plt_row = 2
                plt_col = 3
                plt.figure(figsize=(20, 5))
                plt.subplot(plt_row,plt_col,1)
                plt.imshow(occupancy_grid, cmap='gray', extent=extent, origin='lower')
                plt.plot(sy, sx, marker="o")

                plt.plot(plan_y, plan_x, "-r")
                plt.plot(selected_points[:,1], selected_points[:,0], "xb")
                plt.plot(gy, gx, marker="x", color='r')
                plt.plot(query_point[1], query_point[0], "*", color='yellow', markersize=20)
                plt.title('GT Map + Plan')

                plt.subplot(plt_row,plt_col,2)
                plt.imshow(vis_mask_tot, extent=extent, vmin=0, vmax=1, origin='lower')
                plt.title('Visibility Mask')

                plt.subplot(plt_row,plt_col,3)
                plt.imshow(lidar_mask, extent=extent, vmin=0, vmax=1, origin='lower')
                plt.title('Lidar Mask')

                plt.subplot(plt_row,plt_col,4)
                plt.imshow(inited_flood_grid, extent=extent, vmin=0, vmax=1, origin='lower')
                plt.title('Init Flooded Mask')

                plt.subplot(plt_row,plt_col,5)
                plt.imshow(flooded_grid, extent=extent, vmin=0, vmax=1, origin='lower')
                plt.title('Flooded Mask')
                plt.show()
    pose_list = np.array(pose_list)
    return occupancy_grid, mask_list, local_mask_list, local_gt_list, pose_list 

def make_synthetic_map_with_rand_trajs(map_configs):
    """ For a given map config, generate an occupancy map. 
    Then within that, generate masks with several random trajectories
    
    Output:
        map: occupancy grid
        mask_list: list of masks"""

    # Make occ grid 
    occupancy_grid = make_random_occgrid(map_configs)

    occupancy_grid, mask_list, local_mask_list, local_gt_list, pose_list = make_masklist_with_rand_traj_in_map(occupancy_grid, map_configs)    

    return occupancy_grid, mask_list, local_mask_list, local_gt_list, pose_list




def make_random_occgrid(map_config):
    # Specify the dimensions of the occupancy grid
    num_rows = map_config['num_rows']
    num_cols = map_config['num_cols']

    # Create an empty occupancy grid with a border of 1
    occupancy_grid = np.ones((num_rows+2, num_cols+2), dtype=int)

    # Set the inside of the occupancy grid to 0
    occupancy_grid[1:-1, 1:-1] = 0

    # Specify the dimensions of the central square
    square_size = map_config['square_size']

    # Calculate the starting row and column for the square
    square_start_row = (num_rows - square_size) // 2 + 1
    square_start_col = (num_cols - square_size) // 2 + 1

    # Fill in the central square
    for row in range(square_start_row, square_start_row + square_size):
        for col in range(square_start_col, square_start_col + square_size):
            occupancy_grid[row, col] = 1

    # Fill in smaller squares
    for _ in range(map_config['num_small_squares']):
        smaller_square_size = map_config['smaller_square_size']
        # Sample a position in free space
        position = sample_free_position(occupancy_grid)

        # Calculate the starting row and column for the square
        square_start_row = position[0] - 1
        square_start_col = position[1] - 1

        # Fill in the central square
        for row in range(square_start_row, square_start_row + smaller_square_size):
            for col in range(square_start_col, square_start_col + smaller_square_size):
                occupancy_grid[row, col] = 1


    # Make this grid 10 times bigger
    occupancy_grid = np.repeat(np.repeat(occupancy_grid, map_config['num_times_bigger'], axis=0), 
                               map_config['num_times_bigger'], axis=1)

    # Print the occupancy grid
    # print(occupancy_grid)

    # plt.imshow(occupancy_grid)
    # plt.show()
    return occupancy_grid
