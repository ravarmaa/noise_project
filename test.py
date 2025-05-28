import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

# Step 1: Define correspondences (pixel to lat/lon)
correspondences = [
    (4028, 2758, 58.37916619120761, 26.675538575668313),
    (4336, 3346, 58.36143007189073, 26.691588914890577),
    (5085, 3100, 58.36773980100471, 26.734485484211124),
]

# Extract pixel and lat/lon coordinates
pixels = np.array([(x, y) for x, y, _, _ in correspondences])
latlons = np.array([(lat, lon) for _, _, lat, lon in correspondences])

# Step 2: Fit an affine transformation
def affine_transform(params, latlons, pixels):
    a, b, c, d, e, f = params
    transformed = np.array([
        [a * lat + b * lon + c, d * lat + e * lon + f]
        for lat, lon in latlons
    ])
    return (transformed - pixels).flatten()

# Initial guess for affine parameters
initial_params = [1, 0, 0, 0, 1, 0]

# Optimize the transformation
optimized_params, _ = leastsq(affine_transform, initial_params, args=(latlons, pixels))

# Function to convert lat/lon to pixel coordinates
def latlon_to_pixel(lat, lon, params):
    a, b, c, d, e, f = params
    x = a * lat + b * lon + c
    y = d * lat + e * lon + f
    return int(x), int(y)

# Step 3: Load the image
image_path = "noise_map.jpg"  # Replace with your image file path
image = plt.imread(image_path)

# Display the image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)

# Step 4: Process each folder in curated_data
curated_data_path = "curated_data"  # Replace with your curated_data folder path
for folder in os.listdir(curated_data_path):
    folder_path = os.path.join(curated_data_path, folder)
    location_file = os.path.join(folder_path, "location.txt")
    
    if os.path.isfile(location_file):
        # Read lat/lon from location.txt
        with open(location_file, "r") as f:
            lat, lon = map(float, f.read().strip().split(","))
        
        # Convert lat/lon to pixel coordinates
        pixel_x, pixel_y = latlon_to_pixel(lat, lon, optimized_params)
        
        # Adjust arrow direction and text placement based on folder name
        if folder == "2626":
            arrow_start = (pixel_x + 150, pixel_y)  # Arrow points straight right
            text_x, text_y = arrow_start[0] + 15, arrow_start[1]  # Move text to the right
        elif folder == "Jan-Eric":
            arrow_start = (pixel_x + 150, pixel_y - 150)  # Arrow points to the right diagonally
            text_x, text_y = arrow_start[0], arrow_start[1] - 15
        elif folder == "friend":
            arrow_start = (pixel_x - 150, pixel_y + 150)  # Arrow points to the lower left
            text_x, text_y = arrow_start[0], arrow_start[1] + 15
        elif folder == "Rasmus":
            arrow_start = (pixel_x - 150, pixel_y + 150)  # Arrow points to the lower left
            text_x, text_y = arrow_start[0], arrow_start[1] + 15
        elif folder == "Margot":
            arrow_start = (pixel_x - 150, pixel_y - 150)  # Arrow points to the upper left
            text_x, text_y = arrow_start[0], arrow_start[1] - 15
        else:
            arrow_start = (pixel_x - 150, pixel_y - 150)  # Default: Arrow points to the left diagonally
            text_x, text_y = arrow_start[0], arrow_start[1] - 15
        
        arrow_end = (pixel_x, pixel_y)
        ax.annotate(
            "", xy=arrow_end, xytext=arrow_start,
            arrowprops=dict(facecolor='white', arrowstyle="->", lw=2.5)
        )
        
        # Add folder name as a label with a purple-tinted background box and black border
        ax.text(
            text_x, text_y, folder,
            color="black", fontsize=12, ha="center",  # Increased font size
            bbox=dict(facecolor=(0.9, 0.8, 1, 0.9), edgecolor="black", boxstyle="round,pad=0.3")
        )

# Step 5: Save the updated map
output_path = "noise_map_with_arrows.jpg"
plt.axis("off")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()