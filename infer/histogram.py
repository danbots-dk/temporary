import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
# Load the .npy files
depth_data = np.load('/home/samir/Desktop/blender/pycode/bldev2/scans/30-91822/simulations/Planes/20231019/14h31m55s/render0/nndepth.npy')
dmap_data = np.load('/home/samir/Desktop/blender/pycode/bldev2/scans/30-91822/simulations/Planes/20231019/14h31m55s/render0/dmap.npy')

print("Loaded depth_data:", depth_data)
print("Loaded dmap_data:", dmap_data)

if depth_data.shape != dmap_data.shape:
    print("The two arrays must have the same shape.")
else:
    depth = depth_data*3.93
    dmap = dmap_data*3.93
    error = np.abs(depth-dmap)
     
    # error[error > 1] = np.nan
    error = error * 1000
    # Set errors over 2000 micrometers to NaN
    error[error > 500] = np.nan
    errors_in_micrometers = error.ravel()
    fig, ax = plt.subplots()

    # Plot histogram with density plot
    sns.histplot(errors_in_micrometers, bins=50, kde=True, color='green', ax=ax)

    # Add grid
    ax.grid(False)
    # Remove y-axis
    # ax.yaxis.set_visible(False)

    # Annotating statistics
    ax.set_title('Error Distribution in Micron')
    ax.set_xlabel('Error (Micron)')
    ax.set_ylabel('Count')
    # Save the figure
    plt.savefig('enhanced_depth_comparison.png')