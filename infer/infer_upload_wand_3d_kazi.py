printstat = True # True for full prints/debugging
import os
import numpy as np
from upload_point_cloud import upload_wand_ply
from plyfile import PlyData
from io import StringIO
import open3d as o3d

# suppress tf tons of annoying messages: 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import *
PI = np.pi



# start = 0
# stop =12
# step =1
# modelid = 5

wrapin = '/im_wrap1.png'
# wrapin = '/nnwrap1.png'


def load_model(model_filename):
    return tf.keras.models.load_model(model_filename, compile=False)

def generate_pointcloud(mask, rgb_file, mydepth, ply_file):
    """
    Generates a 3D point cloud from an RGB image and a corresponding depth map, saving the result in a .ply file.

    The function flips the RGB image and the depth map vertically to align them correctly. It then iterates through each
    pixel, and for those pixels where the mask is True, it calculates the 3D coordinates based on the depth information.
    These coordinates are adjusted with a 90-degree counter-clockwise rotation around the Z-axis. The points are
    either colored based on the RGB image or given a static color, depending on the 'color_option' variable.

    Parameters:
    - mask (np.ndarray): A 2D boolean array where True indicates a pixel is part of the object to include in the point cloud.
    - rgb_file (str): The file path to the RGB image.
    - mydepth (np.ndarray): A 2D array representing the depth map, where each value indicates the distance from the camera.
    - ply_file (str): The file path where the .ply file should be saved.

    The depth values in 'mydepth' are assumed to be in millimeters (mm) and are scaled by a factor of 3.93 during processing.
    The field of view (FOV) used for projecting 2D pixels to 3D space is implicitly assumed to be a specific value,
    calibrated for a particular camera setup.

    Returns:
    None. TThe result is directly written to a .ply file specified by 'ply_file', with the unit of measurement being millimeters.
    """

    # rgb = Image.open(rgb_file)
    # depth = mydepth  # np.load(depth_file )
    # # depth = np.fliplr(depth)
    rgb = Image.open(rgb_file).transpose(Image.FLIP_TOP_BOTTOM)  # Flips the RGB image vertically
    depth = np.flipud(mydepth)*3.93  # Flips the depth array vertically assuming 'mydepth' is a NumPy array
    mask =  np.flipud(mask)

    points = []
    print(rgb.size[1], rgb.size[0])
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):

            color = rgb.getpixel((v, u))

            if mask[u, v]:  # (mask.getpixel((v,u))>15):
                # Z = depth.getpixel((u, v))
                Z = depth[u, v] * 1
                Y = .1655 * (v - 80) * Z / 80  # .306 = tan(FOV/2) = tan(48/2)
                X = .1655 * (u - 80) * Z / 80

                # Applying 90-degree counter-clockwise rotation around the Z-axis
                X_new = -Y
                Y_new = X

                color_option = 1  # 1 for image colors, 0 for static color
                if (color_option):
                    # points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
                    points.append("%f %f %f %d %d %d 0\n" % (X_new, Y_new, Z, color[0], color[1], color[2]))
                else:
                    # points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, 127, 127, 127))
                    points.append("%f %f %f %d %d %d 0\n" % (X_new, Y_new, Z, 127, 127, 127))

    # file = open(ply_file,"w")
    # file.write(
    ply_text = '''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
''' % (len(points), "".join(points))
    # file.close()

    with StringIO(ply_text) as f:
        plydata = PlyData.read(f)
    plydata.text = False
    plydata.byte_order = '<'
    plydata.write(f'{ply_file}')


def gammacorr(image, gamma):
    imout = 255 * (image / 255) ** (1 / gamma)
    return (imout)

def infer_upload_wand_3d(*, upload_model_id, wrap_model, depth_model, lum_model, opa_model, batch_name,
                            input_folder, start, stop, step, depth_comparison,
                            x_displacement, y_displacement, z_displacement):
    try:
        date,ts = print_user_info()
        this_dir = create_infer_nnwrap_output_dir(date,ts)

        # create_dictionary
        if not os.path.isdir(this_dir + '/nn_depth'):
            os.mkdir(this_dir+'/nn_depth' )
        nndepthfolder = this_dir+'/nn_depth'

        if not os.path.isdir(this_dir + '/nn_wrap'):
            os.mkdir(this_dir+'/nn_wrap' )
        nnwrapfolder = this_dir+'/nn_wrap'

        if not os.path.isdir(this_dir + '/sqimage'):
            os.mkdir(this_dir+'/sqimage' )
        sqimagefolder = this_dir + '/sqimage'

        if not os.path.isdir(this_dir + '/ply'):
            os.mkdir(this_dir+'/ply' )
        plyfolder = this_dir+'/ply'

        lum_model_predict = load_model(lum_model + '.h5')
        wrap_model_predict = load_model(wrap_model + '.h5')
        depth_model_predict = load_model(depth_model + '.h5')

        flist = []
        for i in range(start, stop, step):
            print(i)

            inpfile = input_folder + '/render' + str(i) + '/sqimage.png'
            # inpfile2 = input_folder + '/render' + str(i) + '/image8.png'
            inpfile2 = input_folder + '/render' + str(i) + '/flash.png'
            
            col2 = cv2.imread(inpfile2).astype(np.float32)
            col2 = cv2.resize(col2, (160, 160), interpolation=cv2.INTER_AREA)
            # cv2.imwrite(input_folder + '/render' + str(i) + '/image8.png', (col2))
            cv2.imwrite(input_folder + '/render' + str(i) + '/flash.png', (col2))
            
            # inpfile = inFolder  + str(i)+ '.png'                  # Choose one or the other !!!!
            sqimage = cv2.imread(inpfile).astype(np.float32)
            img = make_grayscale(sqimage)
            img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
            sqimagepath = sqimagefolder + '/' + str(i) + '.png'
            inp_img = img / 255

            lum_input = lum_model_predict.predict(np.array([np.expand_dims(inp_img, -1)]))
            lum_input = lum_input.squeeze()
            lum = lum_input * 255
            mask = (lum > 25)

            wrapInput = wrap_model_predict.predict(np.array([np.expand_dims(inp_img, -1)]))
            wrapInput = wrapInput.squeeze()
            wrapInput = mask * wrapInput

            wr_save = input_folder + '/render' + str(i) + '/nnwrap.png'
            wrappath = nnwrapfolder + '/' + str(i) + '.png'
            cv2.imwrite(nnwrapfolder + '/' + str(i) + '.png', (256 * wrapInput))

            depthInput = depth_model_predict.predict(np.array([np.expand_dims(wrapInput, -1)]))
            nndepth = depthInput.squeeze()
            imdepth = mask * nndepth
            nndepth = mask * nndepth
            nndepth_save = input_folder + '/render' + str(i) + '/nndepth.npy'
            print(f"Saving nndepth to {nndepth_save}")

            np.save(nndepth_save, 255 * nndepth, allow_pickle=False)
            
            if os.path.exists(nndepth_save):
                print("File successfully saved.")
                depth_data = np.load(nndepth_save)
                print("Loaded depth_data:", depth_data)

                 # Normalize the data to the range 0-255
                depth_data_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)

                # Convert to uint8
                depth_data_uint8 = depth_data_normalized.astype(np.uint8)

                # Save as PNG
                depthpath = nndepthfolder + '/' + str(i) + '.png'
                cv2.imwrite(depthpath, depth_data_uint8)
            else:
                print("File not found after saving:", nndepth_save)
                print("Current working directory:", os.getcwd())
                print("Checking disk space...")
                depthpath = nndepthfolder + '/' + str(i) + '.png'
                cv2.imwrite(depthpath, (256 * nndepth))
                                   
            cv2.imwrite(sqimagefolder + '/' + str(i) + '.png', (img))
            generate_pointcloud(mask, inpfile2, 255*nndepth,
                                plyfolder + '/' + str(i) + '.ply')  # divide nndepth by 2 (128 instead of 256 multiply)

            plypath = plyfolder + '/' + str(i) + '.ply'

            # Depth comparison
            depth_comparison_path = ''
            hist_comparison_path = ''
            if depth_comparison == 'plane':
                point_cloud = o3d.io.read_point_cloud(plypath)
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

                # Filter out distances greater than 500 microns
                valid_distances_microns = distances_microns[distances_microns <= 500]
                # Mean distance of points from the plane, excluding those over 500 microns
                mean_distance_microns = np.mean(valid_distances_microns)
                print(f"Mean distance of valid points from the plane: {mean_distance_microns} microns")
                                # Plotting the error (distance) of each point from the plane using a scatter plot
                
                fig, ax = plt.subplots()
                # Only plot points with distances <= 500 microns
                valid_points_mask = distances_microns <= 500
                
                scatter = ax.scatter(points_microns[valid_points_mask, 0], points_microns[valid_points_mask, 1],
                                    c=valid_distances_microns, cmap='hot')

                cbar = plt.colorbar(scatter, label='Distance from plane (microns)')
                cbar_ticks = np.arange(0, 501, 100)  # Adjusted to go up to 500
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels([str(tick) for tick in cbar_ticks])        
                
                depth_comparison_path = nndepthfolder + '/' + str(i) + 'depth_comparison.png'

                # Optional: Rotate X axis tick labels if they overlap
                plt.xticks(rotation=45)
                plt.ioff()
                # Saving the image
                plt.savefig(depth_comparison_path)
                plt.clf()  # Clear the current figure
                print("Depth comparison saved at:", depth_comparison_path)

                # Histogram
                fig1, ax_hist = plt.subplots()
                sns.histplot(valid_distances_microns, bins=50, kde=True, color='green', ax=ax_hist)
                ax_hist.set_title('Valid Error Distribution in Micron')
                ax_hist.set_xlabel('Error (Micron)')
                ax_hist.set_ylabel('Count')    
                hist_comparison_path = nndepthfolder + '/' + str(i) + 'hist_comparison.png'       
                plt.ioff()
                # Saving the image
                plt.savefig(hist_comparison_path)
                plt.clf()  # Clear the current figure
                print("Histogram comparison saved at:", hist_comparison_path)

            elif depth_comparison == 'sphere':
                pass
            elif depth_comparison == 'stl':
                pass
            else:
                depth_comparison_path = None
                hist_comparison_path = None
            flist.append({
                'point_cloud_file': plypath,
                'fringe_image': sqimagepath,
                'wrap_image': wrappath,
                'depth_image': depthpath,
                'depth_comparison_img': depth_comparison_path,
                'hist_comparison_img': hist_comparison_path,
                'name': batch_name,
                'model': upload_model_id,
                'position': i,
                'x_displacement': x_displacement,
                'y_displacement': y_displacement,
                'z_displacement': z_displacement
            })
            
        responses = upload_wand_ply(files=flist, batch_name=batch_name, model=upload_model_id, depth_model=depth_model, wrap_model=wrap_model, lum_model=lum_model, opa_model=opa_model)
        print(responses)

        # creating/opening a file
        f = open("/home/samir/sal_github/docker/inference-dev-server/infer/save_error.log", "a")

        # writing in the file
        f.write(f"saving successful at {this_dir}\n")  

        # closing the file
        f.close() 

    except Exception as Argument:

        # creating/opening a file
        f = open("/home/samir/sal_github/docker/inference-dev-server/infer/save_error.log", "a")

        # writing in the file
        f.write(str(Argument) + "\n")
        
        # closing the file
        f.close() 