import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files
depth_data = np.load('/home/samir/Desktop/blender/pycode/bldev2/scans/30-91822/simulations/Planes/20231019/14h31m55s/render0/nndepth.npy')
dmap_data = np.load('/home/samir/Desktop/blender/pycode/bldev2/scans/30-91822/simulations/Planes/20231019/14h31m55s/render0/dmap.npy')

if depth_data.shape != dmap_data.shape:
    print("The two arrays must have the same shape.")
else:
    depth = depth_data * 3.93
    dmap = dmap_data * 3.93
    error = np.abs(depth - dmap)
    error = error * 1000  # Convert error to micrometers
    error[error > 500] = np.nan  # Set errors over 500 micrometers to NaN

    plt.figure(figsize=(8, 6))
    im = plt.imshow(error, cmap='hot', interpolation='nearest')

    # Add a color bar to interpret the values
    max_error = np.nanmax(error)
    cbar = plt.colorbar(im)
    cbar.set_label('Error (micron)', rotation=270, labelpad=15)
    # ticks = np.arange(0, 501, 100)
    ticks = np.arange(0, min(501, max_error + 1), 100) 
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks.astype(int))

     # Setting x and y axis ticks at intervals of 100 units
   

    plt.title("Absolute Error between Depth Maps (micron)")
    plt.xlabel("Pixel X Coordinate")
    plt.ylabel("Pixel Y Coordinate")

    # Adjust the plot axes to match the data range, if necessary
    # This would typically be more relevant if you're zooming in on a specific part of the image
    # plt.xlim([0, error.shape[1]])
    # plt.ylim([error.shape[0], 0])

    plt.ioff()
    plt.savefig('depth_comparison_3.png')