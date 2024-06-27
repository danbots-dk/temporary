import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the point cloud
point_cloud = o3d.io.read_point_cloud(
    "/home/kazi/Downloads/wand_point_cloud_file_20240220144425.23daff6f00b14cc19a5cc3fe1a754ab0.ply")

# Convert to numpy array
points = np.asarray(point_cloud.points)

# Convert points from millimeters to microns
points_microns = points * 1000  # Multiply by 1000 to convert from mm to microns

# Compute the centroid
centroid_microns = np.mean(points_microns, axis=0)

# Center the points
centered_points_microns = points_microns - centroid_microns

# Compute the covariance matrix
covariance_matrix = np.cov(centered_points_microns, rowvar=False)

# Compute the eigenvalues and eigenvectors of the covariance matrix
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# Find the index of the smallest eigenvalue
min_eigenvalue_index = np.argmin(eigen_values)

# Get the corresponding eigenvector (normal vector of the plane)
normal_vector = eigen_vectors[:, min_eigenvalue_index]

# Compute the distance from the origin to the plane
d_microns = -np.dot(normal_vector, centroid_microns)

# Calculate the distance of each point from the plane in microns
distances_microns = np.abs((points_microns.dot(normal_vector) + d_microns) / np.linalg.norm(normal_vector))

# Mean distance of points from the plane in microns
mean_distance_microns = np.mean(distances_microns)
print(f"Mean distance of points from the plane: {mean_distance_microns} microns")

# Depth comparasion
# Plotting the error (distance) of each point from the plane using a scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(points_microns[:, 0], points_microns[:, 1], c=distances_microns, cmap='hot')

cbar = plt.colorbar(scatter, label='Distance from plane (microns)')

# Set custom ticks on the X and Y axes at intervals of 100 microns
# Determine the min and max for both axes to set the ticks appropriately
x_min, x_max = np.min(points_microns[:, 0]), np.max(points_microns[:, 0])
y_min, y_max = np.min(points_microns[:, 1]), np.max(points_microns[:, 1])

# Create tick marks from 0 to max_tick at intervals of 100 microns
cbar_ticks = np.arange(0, 1100, 100)
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([str(int(tick)) for tick in cbar_ticks])

# Set labels and title
ax.set_xlabel('X coordinate (microns)')
ax.set_ylabel('Y coordinate (microns)')
ax.set_title('Point Cloud Distance from Fitted Plane in Microns')

# Create tick marks at intervals of 100 microns within the range of your data
x_ticks = np.arange(start=(x_min - x_min % 100), stop=(x_max + 100), step=100)
y_ticks = np.arange(start=(y_min - y_min % 100), stop=(y_max + 100), step=100)

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

# Optional: Rotate X axis tick labels if they overlap
plt.xticks(rotation=45)
plt.show()

# Histogram
fig1, ax_hist = plt.subplots()
sns.histplot(distances_microns, bins=50, kde=True, color='green', ax=ax_hist)
# Add grid
ax_hist.grid(False)
# Remove y-axis
# ax.yaxis.set_visible(False)

# Annotating statistics
ax_hist.set_title('Error Distribution in Micron')
ax_hist.set_xlabel('Error (Micron)')
ax_hist.set_ylabel('Count')
plt.show()
