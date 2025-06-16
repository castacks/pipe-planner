import numpy as np 

def make_room_array(height, width, border_width, door_position, door_width, door_side='left'):
    """Create a 2D array representing a room with a door.

    Args:
        height (int): Height of the room.
        width (int): Width of the room.
        border_width (int): Width of the border.
        door_position (int): Position of the door along height. For now, door starts here.
        door_width (int): Width of the door. #TODO: center on door position

    Returns:
        np.array: 2D array representing the room with 0,1,2 labels. 
    """
    room = np.ones((height, width), dtype=np.uint8) * 2 # intiialize as free space

    # then add walls at the border
    room[0:border_width, :] = 1 # wall
    room[-border_width:, :] = 1 # wall
    room[:, 0:border_width] = 1 # wall
    room[:, -border_width:] = 1 # wall

    # add door
    if door_side == 'left':
        room[door_position:door_position+door_width, 0:border_width] = 2
    elif door_side == 'right':
        room[door_position:door_position+door_width, -border_width:] = 2
    # print(door_position, border_width)
    return room

def arrange_rooms_in_building():
    """Given a larger map, assign indices that correspond to different rooms.
    Output is a 2D array with each index corresponding to different rooms as different labels.
    
    Another output is a list of 2D arrays of room left-top corner and if door is on the left or right side."""

    num_vertical_hallways = np.random.randint(1, 4)
    room_width = 10
    room_height = 10 
    hallway_width = 10
    building_width = (num_vertical_hallways + 1) * (room_width * 2 + hallway_width)
    num_room_along_height = np.random.randint(5, 10)
    building_height = num_room_along_height * room_height
    building_map = np.zeros((building_height, building_width), dtype=np.uint8)

    


    # TODO: add assert statements to make sure that the room height and width are divisible by the building map height and width

    room_counter = 1 # start at 1 because 0 is reserved for free space
    room_list = []

    # left-most side, row of rooms
    left_room_col_start = 0
    for i in range(num_room_along_height):
        building_map[i*room_height:(i+1)*room_height, left_room_col_start:left_room_col_start+room_width] = room_counter
        room_list.append([i*room_height, left_room_col_start, 'right'])
        room_counter += 1

    # right-most side, row of rooms
    right_room_col_start = building_map.shape[1] - room_width
    for i in range(num_room_along_height):
        building_map[i*room_height:(i+1)*room_height, right_room_col_start:right_room_col_start+room_width] = room_counter
        room_list.append([i*room_height, right_room_col_start, 'left'])
        room_counter += 1

    # randomly choose which row is the hallway 
    num_horiz_hallways = np.random.randint(2, 3)
    possible_hallway_rows = range(0,num_room_along_height)
    hallway_rows = np.random.choice(possible_hallway_rows, size=num_horiz_hallways, replace=True)
    for k in range(num_vertical_hallways):
        center_room_col_start = (room_width + hallway_width) +  k * (room_width * 2 + hallway_width)
        for i in range(num_room_along_height):
            if i  in hallway_rows:
                continue
            # center-left room
            building_map[i*room_height:(i+1)*room_height, center_room_col_start:center_room_col_start+room_width] = room_counter
            room_list.append([i*room_height, center_room_col_start, 'left'])
            room_counter += 1
            # center-right room
            building_map[i*room_height:(i+1)*room_height, center_room_col_start+room_width:center_room_col_start+(room_width*2)] = room_counter
            room_list.append([i*room_height, center_room_col_start+room_width, 'right'])
            room_counter += 1


    return building_map, room_list

def make_building_map_given_room_list(rooms, building_map):
    """Given a list of rooms, place them in a larger building map.

    Args:
        rooms (list): List of rooms. Each room is a list of [row, col, door_side]
        building_map (np.array): 2D array representing the building map. Mainly use for shape.

    Returns:
        np.array: 2D array representing the building occupancy map.
    """
    building_occ_map = np.ones((building_map.shape[0], building_map.shape[1]), dtype=np.uint8) * 2 # start as free space
    # add walls at the border
    building_occ_map[0, :] = 1 # wall
    building_occ_map[-1, :] = 1 # wall
    building_occ_map[:, 0] = 1 # wall
    building_occ_map[:, -1] = 1 # wall
    for room in rooms:
        room_height = 10 
        room_width = 10
        room_door_pos = 5
        room_door_width = 2
        room_row = room[0]
        room_col = room[1]
        room_door_side = room[2]
        room_array = make_room_array(room_height, room_width, 1, room_door_pos, room_door_width,door_side=room_door_side)
        building_occ_map[room_row:room_row+room_height, room_col:room_col+room_width] = room_array

    return building_occ_map

def make_building_occ_map():
    #TODO: add params later 
    building_map, room_list = arrange_rooms_in_building()
    building_occ_map = make_building_map_given_room_list(room_list, building_map)
    return building_occ_map