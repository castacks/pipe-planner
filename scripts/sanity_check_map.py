import numpy as np
import matplotlib.pyplot as plt
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
sys.path.append('../')
from scripts.gen_building_utils import *
from scripts import simple_mask_utils as smu 
from models.predictors.map_predictor_model import OccupancyPredictor
from models.networks.unet_model import UNet
import eval_deploy.viz_utils as vutils
import eval_deploy.deploy_utils as dutils
from eval_deploy import glocal_utils as glocal
import scripts.sim_utils as sim_utils

processed_map_dir = '/home/bsj/MapEx/kth_test_maps/'

save = True # if you want to save the new map_file into .npy file (overwrite) --> True

# Available map IDs based on folder names in kth_test_maps:
# - 50052749 (2749)
# - 50052750 (2750)
# - 50010535_PLAN1 (PLAN1)
# - 50010535_PLAN2 (PLAN2)
# - 50010536_PLAN3 (PLAN3)
# - 50052751 (2751)
# - 50052752 (2752)
# - 50052753 (2753)
# - 50052754 (2754)
# - 50015847 (5847)
# - 50015848 (5848)
# - 50037764_PLAN1 (7764_PLAN1)
# - 50037765_PLAN3 (7765_PLAN3)
# - 50052748 (2748)

# Specify which map you want to visualize by setting its ID here
# Available options: '2749', '2750', 'PLAN1', 'PLAN2', 'PLAN3', '2751', '2752', '2753', '2754', '5847', 
# '5848', '7764_PLAN1', '7765_PLAN3', '2748'
map_id_to_visualize = '5848'  # Set to None to visualize nothing

# The following code will automatically handle the visualization based on map_id_to_visualize
Visualize_2749 = map_id_to_visualize == '2749'
Visualize_2750 = map_id_to_visualize == '2750'
Visualize_PLAN1 = map_id_to_visualize == 'PLAN1'
Visualize_PLAN2 = map_id_to_visualize == 'PLAN2'
Visualize_PLAN3 = map_id_to_visualize == 'PLAN3'
Visualize_2751 = map_id_to_visualize == '2751'
Visualize_2752 = map_id_to_visualize == '2752'
Visualize_2753 = map_id_to_visualize == '2753'
Visualize_2754 = map_id_to_visualize == '2754'
Visualize_5847 = map_id_to_visualize == '5847'
Visualize_5848 = map_id_to_visualize == '5848'
Visualize_7764_PLAN1 = map_id_to_visualize == '7764_PLAN1'
Visualize_7765_PLAN3 = map_id_to_visualize == '7765_PLAN3'
Visualize_2748 = map_id_to_visualize == '2748'

if Visualize_2749:
    map_id2749 = '50052749'
    map_id2749_dir = processed_map_dir + map_id2749
    occmap2749_dir = os.path.join(map_id2749_dir, 'occ_map.npy')
    validmap2749_dir = os.path.join(map_id2749_dir, 'valid_space.npy')
    
    map2749 = np.load(occmap2749_dir)
    valid2749 = np.load(validmap2749_dir)
    
    # Fix leak
    map2749[97:108, 13:15] = 0
    map2749[456:471, 145:148] = 0
    map2749[0:648, 0:2] = 0
    map2749[0:648, 850:852] = 0
    map2749[0:2, 0:850] = 0
    map2749[648:650, 0:850] = 0

    map2749[147:154, 184:187] = 0
    map2749[147:154, 205:207] = 0
    map2749[252:261, 255:257] = 0
    map2749[252:261, 278:280] = 0
    map2749[107:110, 380:401] = 0
    map2749[107:110, 500:521] = 0
    map2749[98:102, 58:68] = 0


    valid2749[97:108, 13:15] = 0
    valid2749[456:471, 145:148] = 0
    valid2749[0:648, 0:2] = 0
    valid2749[0:648, 850:852] = 0
    valid2749[0:2, 0:850] = 0
    valid2749[648:650, 0:850] = 0

    valid2749[147:154, 184:187] = 0
    valid2749[147:154, 205:207] = 0
    valid2749[252:261, 255:257] = 0
    valid2749[252:261, 278:280] = 0
    valid2749[107:110, 380:401] = 0
    valid2749[107:110, 500:521] = 0
    valid2749[98:102, 58:68] = 0

    # Plot the labeled map with the colormap
    plt.imshow(map2749)
    plt.show()
    
    if save:
        np.save(occmap2749_dir, map2749)
        np.save(validmap2749_dir, valid2749)


if Visualize_2750:
    map_id2750 ='50052750'
    map_id2750_dir = processed_map_dir + map_id2750
    occmap2750_dir = os.path.join(map_id2750_dir,'occ_map.npy')
    validmap2750_dir = os.path.join(map_id2750_dir,'valid_space.npy')
    map2750 = np.load(occmap2750_dir)
    valid2750 = np.load(validmap2750_dir)

    map2750[146:157, 385:388] = 0
    valid2750[146:157, 385:388] = 0

    map2750[146:157, 405:408] = 0
    valid2750[146:157, 405:408] = 0

    map2750[146:157, 265:268] = 0
    valid2750[146:157, 265:268] = 0

    map2750[146:157, 285:288] = 0
    valid2750[146:157, 285:288] = 0

    map2750[146:155, 183:186] = 0
    valid2750[146:155, 183:186] = 0

    map2750[146:155, 205:208] = 0
    valid2750[146:155, 205:208] = 0

    map2750[253:263, 174:177] = 0
    valid2750[253:263, 174:177] = 0

    map2750[260:263, 252:258] = 0
    valid2750[260:263, 252:258] = 0



    plt.imshow(map2750)
    plt.show()
    
    if save:
        np.save(occmap2750_dir, map2750)
        np.save(validmap2750_dir, valid2750)


if Visualize_PLAN1:
    map_id_plan1 = '50010535_PLAN1'
    map_id_plan1_dir = processed_map_dir + map_id_plan1
    occmap_plan1_dir = os.path.join(map_id_plan1_dir,'occ_map.npy')
    validmap_plan1_dir = os.path.join(map_id_plan1_dir,'valid_space.npy')
    map_plan1 = np.load(occmap_plan1_dir)
    valid_plan1 = np.load(validmap_plan1_dir)
    a,b = map_plan1.shape
    map_plan1[0:2,0:b] = 0
    map_plan1[a-2:a,0:b] = 0
    map_plan1[0:a,0:2] = 0
    map_plan1[0:a,b-2:b] = 0
    map_plan1[125:128, 540:545] = 0
    map_plan1[126:131, 644:649] = 0
    map_plan1[453:456, 537:545] = 0
    map_plan1[450:452, 691:695] = 0
    map_plan1[445:452, 695:698] = 0
    map_plan1[422:425, 689:695] = 0

    valid_plan1[0:2,0:b] = 0
    valid_plan1[a-2:a,0:b] = 0
    valid_plan1[0:a,0:2] = 0
    valid_plan1[0:a,b-2:b] = 0
    valid_plan1[125:128, 540:545] = 0
    valid_plan1[126:131, 644:649] = 0
    valid_plan1[453:456, 537:545] = 0
    valid_plan1[450:452, 691:695] = 0
    valid_plan1[445:452, 695:698] = 0
    valid_plan1[422:425, 689:695] = 0

    map_plan1[58:63, 613:616] = 0
    valid_plan1[58:63, 613:616] = 0

    map_plan1[58:63, 598:601] = 0
    valid_plan1[58:63, 598:601] = 0

    map_plan1[89:92, 586:590] = 0
    valid_plan1[89:92, 586:590] = 0

    map_plan1[93:98, 722:725] = 0
    valid_plan1[93:98, 722:725] = 0

    map_plan1[128:133, 703:706] = 0
    valid_plan1[128:133, 703:706] = 0

    map_plan1[128:133, 723:726] = 0
    valid_plan1[128:133, 723:726] = 0

    map_plan1[235:245, 758:761] = 0
    valid_plan1[235:245, 758:761] = 0

    map_plan1[235:245, 777:780] = 0
    valid_plan1[235:245, 777:780] = 0

    map_plan1[210:214, 812:823] = 0
    valid_plan1[210:214, 812:823] = 0

    map_plan1[232:235, 812:823] = 0
    valid_plan1[232:235, 812:823] = 0

    map_plan1[446:453, 1655:1658] = 0
    valid_plan1[446:453, 1655:1658] = 0

    map_plan1[445:454, 1241:1244] = 0
    valid_plan1[445:454, 1241:1244] = 0

    map_plan1[125:128, 1660:1668] = 0
    valid_plan1[125:128, 1660:1668] = 0

    map_plan1[142:145, 1665:1668] = 0
    valid_plan1[142:145, 1665:1668] = 0

    map_plan1[168:175, 1665:1668] = 0
    valid_plan1[168:175, 1665:1668] = 0

    map_plan1[252:256, 1685:1688] = 0
    valid_plan1[252:256, 1685:1688] = 0

    map_plan1[298:301, 1660:1666] = 0
    valid_plan1[298:301, 1660:1666] = 0

    map_plan1[322:327, 1660:1666] = 0
    valid_plan1[322:327, 1660:1666] = 0

    map_plan1[478:481, 1656:1663] = 0
    valid_plan1[478:481, 1656:1663] = 0

    map_plan1[92:98, 1653:1656] = 0
    valid_plan1[92:98, 1653:1656] = 0

    map_plan1[97:100, 1661:1667] = 0
    valid_plan1[97:100, 1661:1667] = 0

    map_plan1[92:98, 1794:1797] = 0
    valid_plan1[92:98, 1794:1797] = 0

    map_plan1[92:98, 1807:1810] = 0
    valid_plan1[92:98, 1807:1810] = 0

    map_plan1[57:63, 1758:1763] = 0
    valid_plan1[57:63, 1758:1763] = 0

    map_plan1[57:63, 1738:1742] = 0
    valid_plan1[57:63, 1738:1742] = 0

    map_plan1[168:175, 1690:1693] = 0
    valid_plan1[168:175, 1690:1693] = 0

    map_plan1[295:298, 1811:1818] = 0
    valid_plan1[295:298, 1811:1818] = 0

    map_plan1[330:333, 1811:1818] = 0
    valid_plan1[330:333, 1811:1818] = 0

    map_plan1[128:133, 1843:1846] = 0
    valid_plan1[128:133, 1843:1846] = 0

    map_plan1[128:133, 1865:1868] = 0
    valid_plan1[128:133, 1865:1868] = 0



    plt.imshow(map_plan1)
    plt.show()
    
    if save:
        np.save(occmap_plan1_dir, map_plan1)
        np.save(validmap_plan1_dir, valid_plan1)

if Visualize_PLAN2:
    map_id_plan2 = '50010535_PLAN2'
    map_id_plan2_dir = processed_map_dir + map_id_plan2
    occmap_plan2_dir = os.path.join(map_id_plan2_dir,'occ_map.npy')
    validmap_plan2_dir = os.path.join(map_id_plan2_dir,'valid_space.npy')
    map_plan2 = np.load(occmap_plan2_dir)
    valid_plan2 = np.load(validmap_plan2_dir)
    a,b = map_plan2.shape
    map_plan2[0:2,0:b] = 0
    map_plan2[a-2:a,0:b] = 0
    map_plan2[0:a,0:2] = 0
    map_plan2[0:a,b-2:b] = 0
    map_plan2[127:134, 703:705] = 0
    map_plan2[127:134, 724:727] = 0
    map_plan2[127:134, 1064:1067] = 0
    map_plan2[127:134, 1086:1089] = 0
    map_plan2[127:134, 1504:1507] = 0
    map_plan2[127:134, 1523:1526] = 0
    map_plan2[127:133, 1562:1565] = 0
    map_plan2[127:133, 1585:1588] = 0
    map_plan2[465:468, 1813:1816] = 0
    map_plan2[465:468, 1873:1877] = 0

    valid_plan2[0:2,0:b] = 0
    valid_plan2[a-2:a,0:b] = 0
    valid_plan2[0:a,0:2] = 0
    valid_plan2[0:a,b-2:b] = 0
    valid_plan2[127:134, 703:705] = 0
    valid_plan2[127:134, 724:727] = 0
    valid_plan2[127:134, 1064:1067] = 0
    valid_plan2[127:134, 1086:1089] = 0
    valid_plan2[127:134, 1504:1507] = 0
    valid_plan2[127:134, 1523:1526] = 0
    valid_plan2[127:133, 1562:1565] = 0
    valid_plan2[127:133, 1585:1588] = 0
    valid_plan2[465:468, 1813:1816] = 0
    valid_plan2[465:468, 1873:1877] = 0

    plt.imshow(map_plan2)
    plt.show()
    if save:
        np.save(occmap_plan2_dir, map_plan2)
        np.save(validmap_plan2_dir, valid_plan2)

if Visualize_PLAN3:
    map_id_plan3 = '50010536_PLAN3'
    map_id_plan3_dir = processed_map_dir + map_id_plan3
    occmap_plan3_dir = os.path.join(map_id_plan3_dir,'occ_map.npy')
    validmap_plan3_dir = os.path.join(map_id_plan3_dir,'valid_space.npy')
    map_plan3 = np.load(occmap_plan3_dir)
    valid_plan3 = np.load(validmap_plan3_dir)
    a,b = map_plan3.shape
    map_plan3[0:2,0:b] = 0
    map_plan3[a-2:a,0:b] = 0
    map_plan3[0:a,0:2] = 0
    map_plan3[0:a,b-2:b] = 0
    map_plan3[470:473, 538:543] = 0
    map_plan3[447:452, 644:646] = 0


    valid_plan3[0:2,0:b] = 0
    valid_plan3[a-2:a,0:b] = 0
    valid_plan3[0:a,0:2] = 0
    valid_plan3[0:a,b-2:b] = 0
    valid_plan3[470:473, 538:543] = 0
    valid_plan3[447:452, 644:646] = 0

    plt.imshow(map_plan3)
    plt.show()
    if save:
        np.save(occmap_plan3_dir, map_plan3)
        np.save(validmap_plan3_dir, valid_plan3)

if Visualize_5847:
    map_id5847 ='50015847'
    map_id5847_dir = processed_map_dir + map_id5847
    occmap5847_dir = os.path.join(map_id5847_dir,'occ_map.npy')
    validmap5847_dir = os.path.join(map_id5847_dir,'valid_space.npy')
    map5847 = np.load(occmap5847_dir)
    valid5847 = np.load(validmap5847_dir)
    a,b = map5847.shape
    map5847[736:753,804:808] = 0
    map5847[0:2,0:b] = 0
    map5847[a-2:a,0:b] = 0
    map5847[0:a,0:2] = 0
    map5847[0:a,b-2:b] = 0
    valid5847[736:753,804:808] = 0
    valid5847[0:2,0:b] = 0
    valid5847[a-2:a,0:b] = 0
    valid5847[0:a,0:2] = 0
    valid5847[0:a,b-2:b] = 0
    plt.imshow(map5847)
    plt.show()
    if save:
        np.save(occmap5847_dir, map5847)
        np.save(validmap5847_dir, valid5847)

if Visualize_5848:
    map_id5848 ='50015848'
    map_id5848_dir = processed_map_dir + map_id5848
    occmap5848_dir = os.path.join(map_id5848_dir,'occ_map.npy')
    validmap5848_dir = os.path.join(map_id5848_dir,'valid_space.npy')
    map5848 = np.load(occmap5848_dir)
    valid5848 = np.load(validmap5848_dir)
    a,b = map5848.shape
    map5848[0:2,0:b] = 0
    map5848[a-2:a,0:b] = 0
    map5848[0:a,0:2] = 0
    map5848[0:a,b-2:b] = 0

    valid5848[0:2,0:b] = 0
    valid5848[a-2:a,0:b] = 0
    valid5848[0:a,0:2] = 0
    valid5848[0:a,b-2:b] = 0


    plt.imshow(map5848)
    plt.show()
    if save:
        np.save(occmap5848_dir, map5848)
        np.save(validmap5848_dir, valid5848)

if Visualize_2751:
    map_id2751 ='50052751'
    map_id2751_dir = processed_map_dir + map_id2751
    occmap2751_dir = os.path.join(map_id2751_dir,'occ_map.npy')
    validmap2751_dir = os.path.join(map_id2751_dir,'valid_space.npy')
    map2751 = np.load(occmap2751_dir)
    valid2751 = np.load(validmap2751_dir)
    a,b = map2751.shape
    map2751[0:2,0:b] = 0
    map2751[a-2:a,0:b] = 0
    map2751[0:a,0:2] = 0
    map2751[0:a,b-2:b] = 0
    map2751[250:254,725:731] = 0
    map2751[252:254,234:246] = 0
    map2751[146:155,182:186] = 0
    map2751[146:155,205:208] = 0
    map2751[100:110,0:4] = 0


    valid2751[0:2,0:b] = 0
    valid2751[a-2:a,0:b] = 0
    valid2751[0:a,0:2] = 0
    valid2751[0:a,b-2:b] = 0
    valid2751[250:254,725:731] = 0
    valid2751[252:254,234:246] = 0
    valid2751[146:155,182:186] = 0
    valid2751[146:155,205:208] = 0
    valid2751[100:110,0:4] = 0

    plt.imshow(map2751)
    plt.show()
    if save:
        np.save(occmap2751_dir, map2751)
        np.save(validmap2751_dir, valid2751)

if Visualize_2752:
    map_id2752 ='50052752'
    map_id2752_dir = processed_map_dir + map_id2752
    occmap2752_dir = os.path.join(map_id2752_dir,'occ_map.npy')
    validmap2752_dir = os.path.join(map_id2752_dir,'valid_space.npy')
    map2752 = np.load(occmap2752_dir)
    valid2752 = np.load(validmap2752_dir)
    a,b = map2752.shape
    map2752[0:2,0:b] = 0
    map2752[a-2:a,0:b] = 0
    map2752[0:a,0:2] = 0
    map2752[0:a,b-2:b] = 0
    valid2752[0:2,0:b] = 0
    valid2752[a-2:a,0:b] = 0
    valid2752[0:a,0:2] = 0
    valid2752[0:a,b-2:b] = 0

    plt.imshow(map2752)
    plt.show()
    if save:
        np.save(occmap2752_dir, map2752)
        np.save(validmap2752_dir, valid2752)

if Visualize_2753:
    map_id2753 ='50052753'
    map_id2753_dir = processed_map_dir + map_id2753
    occmap2753_dir = os.path.join(map_id2753_dir,'occ_map.npy')
    validmap2753_dir = os.path.join(map_id2753_dir,'valid_space.npy')
    map2753 = np.load(occmap2753_dir)
    valid2753 = np.load(validmap2753_dir)
    a,b = map2753.shape
    map2753[0:2,0:b] = 0
    map2753[a-2:a,0:b] = 0
    map2753[0:a,0:2] = 0
    map2753[0:a,b-2:b] = 0
    map2753[146:155, 182:186] = 0
    map2753[146:155, 205:208] = 0
    map2753[198:201, 175:178] = 0
    map2753[100:110, 0:4] = 0
    map2753[252:254, 233:246] = 0
    map2753[252:254, 725:784] = 0

    valid2753[0:2,0:b] = 0
    valid2753[a-2:a,0:b] = 0
    valid2753[0:a,0:2] = 0
    valid2753[0:a,b-2:b] = 0
    valid2753[146:155, 182:186] = 0
    valid2753[146:155, 205:208] = 0
    valid2753[198:201, 175:178] = 0
    valid2753[100:110, 0:4] = 0
    valid2753[252:254, 233:246] = 0
    valid2753[252:254, 725:784] = 0

    plt.imshow(map2753)
    plt.show()
    if save:
        np.save(occmap2753_dir, map2753)
        np.save(validmap2753_dir, valid2753)

if Visualize_2754:
    map_id2754 ='50052754'
    map_id2754_dir = processed_map_dir + map_id2754
    occmap2754_dir = os.path.join(map_id2754_dir,'occ_map.npy')
    validmap2754_dir = os.path.join(map_id2754_dir,'valid_space.npy')
    map2754 = np.load(occmap2754_dir)
    valid2754 = np.load(validmap2754_dir)
    a,b = map2754.shape
    map2754[0:2,0:b] = 0
    map2754[a-2:a,0:b] = 0
    map2754[0:a,0:2] = 0
    map2754[0:a,b-2:b] = 0
    valid2754[0:2,0:b] = 0
    valid2754[a-2:a,0:b] = 0
    valid2754[0:a,0:2] = 0
    valid2754[0:a,b-2:b] = 0

    plt.imshow(map2754)
    plt.show()
    if save:
        np.save(occmap2754_dir, map2754)
        np.save(validmap2754_dir, valid2754)


if Visualize_7764_PLAN1:
    map_id7764_PLAN1 ='50037764_PLAN1'
    map_id7764_PLAN1_dir = processed_map_dir + map_id7764_PLAN1
    occmap7764_PLAN1_dir = os.path.join(map_id7764_PLAN1_dir,'occ_map.npy')
    validmap7764_PLAN1_dir = os.path.join(map_id7764_PLAN1_dir,'valid_space.npy')
    map7764_PLAN1 = np.load(occmap7764_PLAN1_dir)
    valid7764_PLAN1 = np.load(validmap7764_PLAN1_dir)
    plt.imshow(map7764_PLAN1)
    plt.show()
    if save:
        np.save(occmap7764_PLAN1_dir, map7764_PLAN1)
        np.save(validmap7764_PLAN1_dir, valid7764_PLAN1)

if Visualize_7765_PLAN3:
    map_id7765_PLAN3 ='50037765_PLAN3'
    map_id7765_PLAN3_dir = processed_map_dir + map_id7765_PLAN3
    occmap7765_PLAN3_dir = os.path.join(map_id7765_PLAN3_dir,'occ_map.npy')
    validmap7765_PLAN3_dir = os.path.join(map_id7765_PLAN3_dir,'valid_space.npy')
    map7765_PLAN3 = np.load(occmap7765_PLAN3_dir)
    valid7765_PLAN3 = np.load(validmap7765_PLAN3_dir)
    map7765_PLAN3[234:238:,859:871] = 0
    map7765_PLAN3[260:264,859:871] = 0
    map7765_PLAN3[171:176,1029:1033] = 0
    map7765_PLAN3[260:263,1780:1790] = 0
    map7765_PLAN3[135:140,1948:1950] = 0
    map7765_PLAN3[135:140,2260:2262] = 0
    map7765_PLAN3[135:140,2382:2384] = 0
    map7765_PLAN3[135:140,2588:2590] = 0
    map7765_PLAN3[90:99,1789:1792] = 0
    map7765_PLAN3[296:300,862:872] = 0

    valid7765_PLAN3[234:238:,859:871] = 0
    valid7765_PLAN3[260:264,859:871] = 0
    valid7765_PLAN3[171:176,1029:1033] = 0
    valid7765_PLAN3[260:263,1780:1790] = 0
    valid7765_PLAN3[135:140,1948:1950] = 0
    valid7765_PLAN3[135:140,2260:2262] = 0
    valid7765_PLAN3[135:140,2382:2384] = 0
    valid7765_PLAN3[135:140,2588:2590] = 0
    valid7765_PLAN3[90:99,1789:1792] = 0
    valid7765_PLAN3[296:300,862:872] = 0
    plt.imshow(map7765_PLAN3)
    plt.show()
    if save:
        np.save(occmap7765_PLAN3_dir, map7765_PLAN3)
        np.save(validmap7765_PLAN3_dir, valid7765_PLAN3)


if Visualize_2748:
    map_id2748 ='50052748'
    map_id2748_dir = processed_map_dir + map_id2748
    occmap2748_dir = os.path.join(map_id2748_dir,'occ_map.npy')
    validmap2748_dir = os.path.join(map_id2748_dir,'valid_space.npy')
    map2748 = np.load(occmap2748_dir)
    valid2748 = np.load(validmap2748_dir)
    map2748[133:139,7:10] = 0
    map2748[183:189,106:110] = 0
    map2748[183:189,131:134] = 0
    map2748[81:86,779:788] = 0
    map2748[295:299,484:494] = 0
    map2748[289:293,138:141] = 0
    map2748[345:347,257:486] = 0
    map2748[38:41,260:266] = 0

    valid2748[133:139,7:10] = 0
    valid2748[183:189,106:110] = 0
    valid2748[183:189,131:134] = 0
    valid2748[81:86,779:788] = 0
    valid2748[295:299,484:494] = 0
    valid2748[289:293,138:141] = 0
    valid2748[345:347,257:486] = 0
    valid2748[38:41,260:266] = 0
    plt.imshow(map2748)
    plt.show()
    if save:
        np.save(occmap2748_dir, map2748)
        np.save(validmap2748_dir, valid2748)