import numpy as np
import tkinter

import os 
import cv2
from tqdm.contrib.concurrent import process_map  # or thread_map
import time
from omegaconf import OmegaConf
import hydra 
import torch 
from torchvision.transforms import ToTensor, Resize, Compose
import pyastar2d    
import json
import multiprocessing
from tqdm import tqdm
import os 
import sys
import argparse

import simple_mask_utils as smu
import sim_utils
import matplotlib.pyplot as plt
import random

def sample_goalpose_from_validspace(validspace_map):
    valid_index = np.where(validspace_map == 1)
    valid_index_list = list(zip(valid_index[0],valid_index[1]))
    random_indices = random.sample(valid_index_list, 100)
    return np.array(random_indices)


map_start_pairs = [['50010535_PLAN1', [515,515]],
                   ['50010535_PLAN1', [773,1512]],
                   ['50010535_PLAN1', [768,551]],
                   ['50010535_PLAN1', [515,1472]],
                   ['50010535_PLAN2', [515,515]],
                   ['50010535_PLAN2', [768,551]],
                   ['50010535_PLAN2', [515,1517]],
                   ['50010535_PLAN2', [773,1512]],
                   ['50010536_PLAN3', [515,515]],
                   ['50010536_PLAN3', [768,515]],
                   ['50010536_PLAN3', [515,1515]],
                   ['50010536_PLAN3', [768,1515]],
                   ['50052749',[515,515]],
                   ['50052749',[513,880]],
                   ['50052749',[800,583]],
                   ['50052749',[802,913]],
                   ['50052750',[515,515]],
                   ['50052750',[513,880]],
                   ['50052750',[800,583]],
                   ['50052750',[802,913]],
                   ['50052751',[515,515]],
                   ['50052751',[515,880]],
                   ['50052751',[615,515]],
                   ['50052751',[615,880]],
                   ['50052752',[515,515]],
                   ['50052752',[515,880]],
                   ['50052752',[615,515]],
                   ['50052752',[615,880]],
                   ['50052753',[515,515]],
                   ['50052753',[515,880]],
                   ['50052753',[615,515]],
                   ['50052753',[615,880]],
                   ['50052754',[515,515]],
                   ['50052754',[515,880]],
                   ['50052754',[615,540]],
                   ['50052754',[615,880]],
                   ['50037765_PLAN3',[540,535]],
                   ['50037765_PLAN3',[700,525]],
                   ['50037765_PLAN3',[515,1815]],
                   ['50037765_PLAN3',[615,1780]]
                   ]

methods = ['nearest', 'upen', 'hectoraug', 'visvarprob']
#methods = ['onlyvar', 'obsunk', 'visunk', 'visvar', 'visvarprob']

results_path = '/home/seungchan/Documents/map_prediction_toolbox/experiments/results/'
processed_kth_map_folder = '/home/seungchan/MapEx/kth_test_maps/'
topological_exp_path = '/home/seungchan/MapEx/experiments/topological'

#generate random poses per each map_id and start_pairs, and save them in a ./random_poses/ folder as a .npy file 
generate_new_randomposes = False
if generate_new_randomposes:
    for map_id_start_pose_pairs in map_start_pairs:
        map_id = map_id_start_pose_pairs[0]
        start_pose = map_id_start_pose_pairs[1]
        print(map_id, start_pose[0], start_pose[1])
        map_occ_npy_path = os.path.join(processed_kth_map_folder, map_id, 'occ_map.npy')
        map_valid_space_npy_path = os.path.join(processed_kth_map_folder, map_id, 'valid_space.npy')
        occ_map, validspace_map = sim_utils.get_kth_occ_validspace_map(map_occ_npy_path, map_valid_space_npy_path)
        random_indices = sample_goalpose_from_validspace(validspace_map)
        random_indice_save_fpath = os.path.join(topological_exp_path, 'random_poses', map_id+'_'+str(start_pose[0])+'_'+str(start_pose[1])+'.npy')
        np.save(random_indice_save_fpath, random_indices)
    exit()

timesteps = [100,200,300,400,500,600,700,800,900]

pd_size = 500
for map_id_start_pose_pairs in map_start_pairs:
    map_id = map_id_start_pose_pairs[0]
    start_pose = map_id_start_pose_pairs[1]
    f = open(os.path.join(topological_exp_path, map_id), 'a')

    map_occ_npy_path = os.path.join(processed_kth_map_folder, map_id, 'occ_map.npy')
    map_valid_space_npy_path = os.path.join(processed_kth_map_folder, map_id, 'valid_space.npy')
    occ_map, validspace_map = sim_utils.get_kth_occ_validspace_map(map_occ_npy_path, map_valid_space_npy_path)

    gt_h, gt_w = occ_map.shape[0],occ_map.shape[1]
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
    
    random_endposes_npy = os.path.join(topological_exp_path, 'random_poses', map_id+'_'+str(start_pose[0])+'_'+str(start_pose[1])+'.npy')
    random_indices = np.load(random_endposes_npy)
    for method in methods:
        for timestep in timesteps:
            print(map_id, start_pose, method, timestep)
            mapid_path = results_path + map_id
            folderslist = os.listdir(mapid_path)
            folderlist = [folder for folder in folderslist if map_id+'_'+str(start_pose[0])+'_'+str(start_pose[1]) in folder and folder.endswith(method)]
            assert len(folderlist) == 1
            foldername = folderlist[0]
            predfile = os.path.join(mapid_path, foldername, 'global_pred', '00000'+str(timestep)+'_pred.npy')
            predmap_np = np.load(predfile)
            predmap_np_reshaped = predmap_np
            predmap_np_reshaped = predmap_np[pad_h1:-pad_h2, pad_w1:-pad_w2]
            predmap_utils = np.zeros((predmap_np_reshaped.shape[0], predmap_np_reshaped.shape[1]))
            predmap_utils[predmap_np_reshaped[:,:,0]> 128] = 1 #occ

            occ_grid_pyastar = np.zeros_like(predmap_utils, dtype=np.float32)
            occ_grid_pyastar[predmap_utils == 0] = 1 #free
            occ_grid_pyastar[predmap_utils > 0] = np.inf #occ
            plt.figure()
            plt.imshow(occ_grid_pyastar[pd_size:-pd_size,pd_size:-pd_size])
            fail = 0
            succeed = 0
            for random_index in random_indices:
                end_pose = [random_index[0], random_index[1]]
                plt.scatter(end_pose[1]-pd_size,end_pose[0]-pd_size,color='y',s=10)
                try:
                    a_star_path = pyastar2d.astar_path(occ_grid_pyastar, start_pose, end_pose, allow_diagonal=False)
                    if a_star_path is None:
                        fail += 1
                    else:
                        if np.any(occ_map[a_star_path[:,0],a_star_path[:,1]] == 1):
                            fail += 1 
                            plt.scatter(a_star_path[:,1]-pd_size,a_star_path[:,0]-pd_size,color='r',s=0.1)
                        else:
                            succeed += 1
                            plt.scatter(a_star_path[:,1]-pd_size,a_star_path[:,0]-pd_size,color='g',s=0.1)
                except:
                    pass
            # print("succeed: ", succeed, " fail: ", fail, " total: ", succeed+fail)
            total = succeed+fail
            f.write(map_id + ',start,'+str(start_pose[0])+','+str(start_pose[1])+','+method+',timestep,'+str(timestep)+',succeed,'+str(succeed)+',fail,'+str(fail)+',total,'+str(total)+',succeed_rate,'+str(float(100*succeed/total))+',fail_rate,'+str(float(100*fail/total))+'\n')
            f.flush()
            plt.savefig(os.path.join(topological_exp_path, 'figures', map_id+'_'+str(start_pose[0])+'_'+str(start_pose[1])+'_'+method+'_'+str(timestep)))
            plt.close()