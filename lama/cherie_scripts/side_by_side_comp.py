import os
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm 

def display_frames_side_by_side(folder_paths):
    images = []
    
    frame_numbers = [int(f.split('.')[0][:8]) for f in sorted(os.listdir(folder_paths[1])) if f.endswith('.png')]
    for frame_number in tqdm(frame_numbers):
        frame_images = []
        for folder_path in folder_paths:
            if folder_path == '../cherie_data':
                image_path = os.path.join(folder_path, f'{frame_number:08d}.png')
            else:
                image_path = os.path.join(folder_path, f'{frame_number:08d}_mask001.png')
            image = cv2.imread(image_path)
            frame_images.append(image)
        # import pdb; pdb.set_trace()
        # Match the width of all frame images 
        max_width = max(image.shape[1] for image in frame_images)
        frame_images = [cv2.resize(image, (max_width, int(image.shape[0] * max_width / image.shape[1]))) for image in frame_images]
        
        concat_images = cv2.vconcat(frame_images)
        # plt.imshow(concat_images)
        # plt.show()
        cv2.imwrite(f'../output_map_concat/{frame_number:08d}_mask001.png', concat_images)
    #     cv2.imshow('Frames Side by Side', concat_images)
    
    # combined_image = cv2.hconcat(images)
    # cv2.imshow('Frames Side by Side', combined_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# # Example usage
# folder_paths = ['../cherie_data', '../output_map', '../output_map_refine_2', '../output_pred']
# display_frames_side_by_side(folder_paths)

# Output video file
frame_folder_path = '../output_map_concat'
video_file = 'output_video.avi'

# Assuming all images are the same size, get dimensions from the first image
image_files = [img for img in os.listdir(frame_folder_path) if img.endswih(".png")]  # Adjust the extension as per your image files
first_image = cv2.imread(os.path.join(frame_folder_path, image_files[0]))
height, width, layers = first_image.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video = cv2.VideoWriter(video_file, fourcc, 20.0, (width, height))

for image_file in tqdm(sorted(image_files), desc='Writing to video'):
    image_path = os.path.join(frame_folder_path, image_file)
    frame = cv2.imread(image_path)

    # Write the frame to the video
    video.write(frame)

# Release everything when the job is finished
video.release()

print("Video creation completed.")