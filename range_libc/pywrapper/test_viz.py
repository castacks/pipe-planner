import range_libc
import numpy as np
import itertools, time

# testMap = range_libc.PyOMap("../maps/basement_hallways_5cm.png",1)
import cv2 
occ_map = 1 - cv2.imread('../maps/basement_hallways_5cm.png', cv2.IMREAD_GRAYSCALE) / 255.0 
start_time = time.time()
testMap = range_libc.PyOMap(occ_map)
print("Time to load map: ", time.time() - start_time)
# testMap = range_libc.PyOMap(b"../maps/synthetic.map.png",1)
# testMap = range_libc.PyOMap("/home/racecar/racecar-ws/src/TA_examples/lab5/maps/basement.png",1)

if testMap.error():
	exit()

def make_scan(x,y,theta,n_ranges, range_pix):
	start_time = time.time()
	bl = range_libc.PyBresenhamsLine(testMap, range_pix)
	# bl = range_libc.PyRayMarching(testMap, 500)
	queries = np.zeros((n_ranges,3),dtype=np.float32)
	ranges = np.zeros(n_ranges,dtype=np.float32)
	queries[:,0] = x
	queries[:,1] = y
	queries[:,2] = np.linspace(0, 2 * np.pi, n_ranges)
	
	bl.calc_range_many(queries,ranges)
	bresenham_time_needed = time.time() - start_time
 
	# show occupancy map 
	import matplotlib.pyplot as plt
	plt.subplot(1,2,1)
	plt.imshow(occ_map, cmap='gray')
	plt.subplot(1,2,2)
	plt.imshow(occ_map, cmap='gray')
	# show rays (queries[0,1] is origin, queries[2] is angle)
	for i in range(n_ranges):
		ray_start = (queries[i,0], queries[i,1])
		ray_end = (queries[i,0] + ranges[i]*np.cos(queries[i,2]), queries[i,1] + ranges[i]*np.sin(queries[i,2]))
		plt.plot([ray_start[1], ray_end[1]], [ray_start[0], ray_end[0]], 'r-', alpha=0.2)
	plt.scatter([y], [x], c='b', s=10, zorder=10)
	plt.title("Raycast took " + str(np.round(bresenham_time_needed, 5)) + " sec " + "for " + str(n_ranges) + " rays")
	plt.show()

x_origin = 960
y_origin = 805
theta_origin = 0 
num_ranges = 10000
range_pix = 10000
make_scan(x_origin, y_origin, theta_origin, num_ranges, range_pix)