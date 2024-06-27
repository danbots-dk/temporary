import os
import numpy as np
from upload_point_cloud import upload_simulation_ply
from plyfile import PlyData
from io import StringIO

import matplotlib

matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

# suppress tf tons of annoying messages:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import *

PI = np.pi




# start = 0
# stop =10000
# step =43

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
    # depth = np.fliplr(depth)

    rgb = Image.open(rgb_file).transpose(Image.FLIP_TOP_BOTTOM)  # Flips the RGB image vertically
    depth = np.flipud(mydepth)*3.93 # Flips the depth array vertically assuming 'mydepth' is a NumPy array

    points = []
    print(rgb.size[1], rgb.size[0])
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):

            color = rgb.getpixel((v, u))

            if mask[u, v]:  # (mask.getpixel((v,u))>15):
                # Z = depth.getpixel((u, v))
                Z = depth[u, v] * 1
                
                # Y = .23 * (v - 80) * Z / 80  # .306 = tan(FOV/2) = tan(48/2)
                # X = .23 * (u - 80) * Z / 80
                
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
                    points.append("%f %f %f %d %d %d 0\n" % (X_new, Y_new, Z, 127, 127, 127))
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


def generate_col_pointcloud(mask, rgb_file, mydepth, ply_file):

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
    # depth = np.fliplr(depth)

    rgb = Image.open(rgb_file).transpose(Image.FLIP_TOP_BOTTOM)  # Flips the RGB image vertically
    depth = np.flipud(mydepth)*3.93 # Flips the depth array vertically assuming 'mydepth' is a NumPy array

    points = []
    print(rgb.size[1], rgb.size[0])
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):

            color = rgb.getpixel((v, u))

            if mask[u, v]:  # (mask.getpixel((v,u))>15):
                # Z = depth.getpixel((u, v))
                Z = depth[u, v] * 1
                
                # Y = .23 * (v - 80) * Z / 80  # .306 = tan(FOV/2) = tan(48/2)
                # X = .23 * (u - 80) * Z / 80
                
                Y = .1655 * (v - 80) * Z / 80  # .306 = tan(FOV/2) = tan(48/2)
                X = .1655 * (u - 80) * Z / 80
                
                # Applying 90-degree counter-clockwise rotation around the Z-axis
                X_new = -Y
                Y_new = X
                
                color_option = 1  # 1 for image colors, 0 for static color
                if (color_option):
                    # points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
                    points.append("%f %f %f %d %d %d 0\n" % (X_new, Y_new, Z, 140, 120, 0))
                else:
                    points.append("%f %f %f %d %d %d 0\n" % (X_new, Y_new, Z, 127, 127, 127))
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

def infer_upload_simulation(*, upload_model_id, wrap_model, depth_model, lum_model, opa_model, batch_name,
                            input_folder, start, stop, step):
    date, ts = print_user_info()
    this_dir = create_infer_nnwrap_output_dir(date, ts)

    # create_dictionary
    try:
        if not os.path.isdir(this_dir + '/nn_depth'):
            os.mkdir(this_dir + '/nn_depth')
        nndepthfolder = this_dir + '/nn_depth'

        if not os.path.isdir(this_dir + '/nn_wrap'):
            os.mkdir(this_dir + '/nn_wrap')
        nnwrapfolder = this_dir + '/nn_wrap'

        if not os.path.isdir(this_dir + '/sqimage'):
            os.mkdir(this_dir + '/sqimage')
        sqimagefolder = this_dir + '/sqimage'

        if not os.path.isdir(this_dir + '/opaimg'):
            os.mkdir(this_dir + '/opaimg')
        opaimgfolder = this_dir + '/opaimg'

        if not os.path.isdir(this_dir + '/ply'):
            os.mkdir(this_dir + '/ply')
        plyfolder = this_dir + '/ply'

        lum_model_predict = load_model(lum_model + '.h5')
        wrap_model_predict = load_model(wrap_model + '.h5')
        depth_model_predict = load_model(depth_model + '.h5')
        opa_model_predict = load_model(opa_model + '.h5')
        flist = []
        for i in range(start, stop, step):
            
            print(i)

            if os.path.isfile(input_folder + 'render' + str(i) + '/wimage0.png'):
                inpfile = input_folder + 'render' + str(i) + '/wimage0.png'
            elif os.path.isfile(input_folder + 'render' + str(i) + '/image0.png'):
                inpfile = input_folder + 'render' + str(i) + '/image0.png'
            elif os.path.isfile(input_folder + 'render' + str(i) + '/wimage.png'):
                inpfile = input_folder + 'render' + str(i) + '/wimage.png'
            elif os.path.isfile(input_folder + 'render' + str(i) + '/sqimage.png'):
                inpfile = input_folder + 'render' + str(i) + '/sqimage.png'
            
            if os.path.isfile(input_folder + 'render' + str(i) + '/image8.png'):
                inpfile2 = input_folder + 'render' + str(i) + '/image8.png' 
            else:
                inpfile2 = inpfile
                
            # inpfile3 = input_folder + 'render' + str(i) + '/im_wrap1.png'
            
            
            # inpfile = inFolder  + str(i)+ '.png'                  # Choose one or the other !!!!
            sqimage = cv2.imread(inpfile).astype(np.float32)
            img = make_grayscale(sqimage)
            img = gammacorr(img, 1 ** (-1))
            sqimagepath = sqimagefolder + '/' + str(i) + '.png'
            inp_img = img / 255

            opa_input = opa_model_predict.predict(np.array([np.expand_dims(inp_img, -1)]))
            opa_input = opa_input.squeeze()
            opa_input_img = opa_input * 255
            cv2.imwrite(opaimgfolder + '/' + str(i) + '.png', (opa_input_img))
            
            lum_input = lum_model_predict.predict(np.array([np.expand_dims(opa_input, -1)]))
            lum_input = lum_input.squeeze()
            lum = lum_input * 255
            mask = (lum > 25)
            fmask = np.flipud(mask)
            
            # wrapin = cv2.imread(inpfile3).astype(np.float32)
            # wrapin = make_grayscale(wrapin)/255
            # wrapin = mask * wrapin

            wrapInput = wrap_model_predict.predict(np.array([np.expand_dims(inp_img, -1)]))
            wrapInput = wrapInput.squeeze()
            wrapInput = mask * wrapInput
            
            wr_save = input_folder + '/render' + str(i) + '/nnwrap.png'
            wrappath = nnwrapfolder + '/' + str(i) + '.png'
            cv2.imwrite(nnwrapfolder + '/' + str(i) + '.png', (255 * wrapInput))

            depthInput = depth_model_predict.predict(np.array([np.expand_dims(wrapInput, -1)]))
            nndepth = depthInput.squeeze()
            imdepth = mask * nndepth
            nndepth = mask * nndepth
            nndepth_save = input_folder + '/render' + str(i) + '/nndepth.npy'
            print(f"Saving nndepth to {nndepth_save}")

            np.save(nndepth_save, 255 * nndepth, allow_pickle=False)

            if os.path.exists(nndepth_save):
                print("File successfully saved.")
                # depth_data = np.load(nndepth_save)
                # print("Loaded depth_data:", depth_data)

                # Normalize the data to the range 0-255
                depth_data_normalized = cv2.normalize(imdepth, None, 0, 255, cv2.NORM_MINMAX)

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
                cv2.imwrite(depthpath, (255 * nndepth))

            # Depth comparison
            dmap_saved = input_folder + '/render' + str(i) + '/dmap.npy'
            # wrap_data = wrapin *255
            nnwrap_data = wrapInput*255
            depth_comparison_path = ''
            hist_comparison_path = ''
            if os.path.exists(dmap_saved) and os.path.exists(nndepth_save):
                depth_data = np.load(nndepth_save)
                dmap_data = np.load(dmap_saved)*mask

                generate_pointcloud(fmask, inpfile2, 1*dmap_data,
                                plyfolder + '/truth_' + str(i) + '.ply')  # divide nndepth by 2 (128 instead of 256 multiply)

                truth_ply =  plyfolder + '/truth_' + str(i) + '.ply'
                
                print("Loaded depth_data:", depth_data)
                print("Loaded dmap_data:", dmap_data)

                if depth_data.shape != dmap_data.shape:
                    print("The two arrays must have the same shape.")
                else:
                    depth = depth_data*3.93
                    dmap = dmap_data*3.93
                    depth_error = np.abs(depth - dmap)
                    depth_error = depth_error * 1000
                    depth_error[depth_error > 500] = np.nan
                    
                    # Plotting
                    # plt.imshow(depth_error, cmap='hot', interpolation='nearest', vmin=0,vmax=.3)
                    plt.imshow(depth_error, cmap='hot', interpolation='nearest')
                    # Add a color bar to interpret the values
                    cbar = plt.colorbar()
                    cbar.set_label('Error (micron)', rotation=270, labelpad=15)

                    # # Set the ticks and labels so that '0' is replaced by '100'
                    # ticks = np.linspace(0, 1, 11)  # Normalized tick positions from 0 to 1
                    # labels = (ticks * 1000).astype(int)  # Scale and convert to integer for display
                    # cbar.set_ticks(ticks)
                    # cbar.set_ticklabels(labels)


                    # Calculate the color bar ticks and labels
                    max_error = np.nanmax(depth_error)  # Find the maximum error, ignoring NaN
                    # ticks = np.arange(0, min(501, max_error + 1), 100)  # Create ticks every 100 microns up to 500 or max_error
                    ticks = np.arange(0, 501, 100)
                    cbar.set_ticks(ticks)  # Set the ticks on the color bar
                    cbar.set_ticklabels(ticks.astype(int))  # Set tick labels, converting to integer for readability


                    plt.title("Absolute Error between Depth Maps (micron)")
                    depth_comparison_path = nndepthfolder + '/' + str(i) + 'depth_comparison.png'
                    plt.ioff()
                    # Saving the image
                    plt.savefig(depth_comparison_path)
                    plt.clf()  # Clear the current figure
                    print("Depth comparison saved at:", depth_comparison_path)

                    # Histogram
                    hist_error = np.abs(depth - dmap)
                    
                    hist_error = hist_error * 1000
                    hist_error[hist_error > 500] = np.nan
                    errors_in_micrometers = hist_error.ravel()
                    fig, ax = plt.subplots()

                    # Plot histogram with density plot
                    sns.histplot(errors_in_micrometers, bins=50, kde=True, color='green', ax=ax)
                    ax.grid(False)
                    # Remove y-axis
                    ax.yaxis.set_visible(False)
                    # Annotating statistics
                    ax.set_title('Error Distribution in Micron')
                    ax.set_xlabel('Error (Micron)')
                    ax.set_ylabel('Count')
                    hist_comparison_path = nndepthfolder + '/' + str(i) + 'hist_comparison.png'
                    plt.ioff()
                    # Saving the image
                    plt.savefig(hist_comparison_path)
                    plt.clf()  # Clear the current figure
                    print("Histogram comparison saved at:", hist_comparison_path)
            else:
                depth_comparison_path = None
                hist_comparison_path = None
                truth_ply = None
                print("dmap or nndepth file missing")

            # depthpath = nndepthfolder + '/' + str(i) + '.png'
            # cv2.imwrite(depthpath, (256 * nndepth))
            # # cv2.imwrite(nndepthfolder + '/' + str(i) + '.png', (20 * nndepth))
            # np.save(wr_save, 256 * nndepth, allow_pickle=False)
        
            cv2.imwrite(sqimagefolder + '/' + str(i) + '.png', (img))
            generate_pointcloud(fmask, inpfile2, 255*nndepth,
                                plyfolder + '/' + str(i) + '.ply')  # divide nndepth by 2 (128 instead of 256 multiply)

          
            plypath = plyfolder + '/' + str(i) + '.ply'
          
            
            flist.append({
                'point_cloud_file': plypath,
                'grand_truth_point_cloud_file': truth_ply,
                'fringe_image': sqimagepath,
                'wrap_image': wrappath,
                'depth_image': depthpath,
                'depth_comparison_img': depth_comparison_path,
                'hist_comparison_img': hist_comparison_path,
                'name': batch_name,
                'model': upload_model_id,
                'position': i
            })

        responses = upload_simulation_ply(files=flist, batch_name=batch_name, model=upload_model_id,
                                          depth_model=depth_model, wrap_model=wrap_model, lum_model=lum_model, opa_model=opa_model)
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