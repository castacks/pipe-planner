import imageio
import glob
import os

# Set the specific directory path
input_dir = '/home/bsj/MapEx/experiments/20250402_test/20250402_171247_50010535_PLAN1_765_564_pipe/run_viz'

# Get sorted list of PNG files from the specific directory
all_png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

# Filter files up to the one containing '600'
png_files = []
for file in all_png_files:
    if '400' in file:
        png_files.append(file)
        break
    png_files.append(file)

# Read images into a list
images = [imageio.imread(filename) for filename in png_files]

# Create output directory if it doesn't exist
output_dir = os.path.dirname(input_dir)
output_path = os.path.join(output_dir, "output.gif")

# Save the images as an animated GIF with a frame duration of 0.1 seconds
imageio.mimsave(output_path, images, duration=0.02)

print(f"GIF created successfully at: {output_path}")
print(f"Total frames: {len(images)}")