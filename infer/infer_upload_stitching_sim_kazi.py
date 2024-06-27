import os
import time
from upload_point_cloud import upload_simulation_ply
from upload_stitching_ply import upload_stitching_simulation_ply
from plyfile import PlyData
from io import StringIO
import open3d as o3d
from help_functions import *

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
    Generates a 3D point cloud from an RGB image and a depth map, saving the result in a .ply file.
    The depth values are scaled by a factor of 3.93 to match the measurement units expected by the algorithm.

    Parameters:
    - mask (np.ndarray): A 2D boolean array specifying which pixels in the depth map should be included in the point cloud.
                         True indicates inclusion of the pixel at that position.
    - rgb_file (str): Path to the RGB image file. The image is opened and used to extract color information for the point cloud.
    - mydepth (np.ndarray): A 2D array representing the depth information for each pixel, scaled by 3.93.
                            Assumes depth values are aligned with the RGB image.
    - ply_file (str): Path where the generated .ply file will be saved.

    This function iterates over each pixel in the RGB image. For pixels where the mask is True, it calculates
    the 3D coordinates (X, Y, Z) using the depth information. The X and Y coordinates are calculated to simulate
    a projection from 2D to 3D space, considering a fixed field of view. The function supports two color modes
    for the point cloud: using the original colors from the RGB image or using a static color.

    The depth values in 'mydepth' are assumed to be in millimeters (mm) and are scaled by a factor of 3.93 during processing.
    The field of view (FOV) used for projecting 2D pixels to 3D space is implicitly assumed to be a specific value,
    calibrated for a particular camera setup.

    Returns:
    None. TThe result is directly written to a .ply file specified by 'ply_file', with the unit of measurement being millimeters.
    """
    rgb = Image.open(rgb_file)
    depth = mydepth*3.93  # np.load(depth_file )
    # depth = np.fliplr(depth)
    points = []
    print(rgb.size[1], rgb.size[0])
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):

            color = rgb.getpixel((v, u))
            # print(color)

            if mask[u, v]:  # (mask.getpixel((v,u))>15):
                # Z = depth.getpixel((u, v))
                Z = depth[u, v] * 1
                Y = .1655 * (v - 80) * Z / 80  # .306 = tan(FOV/2) = tan(48/2)
                X = .1655 * (u - 80) * Z / 80
                color_option = 1  # 1 for image colors, 0 for static color
                if (color_option):
                    # print("color")
                    points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
                else:
                    # print("no color")
                    points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, 127, 127, 127))

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

def global_and_icp_registration(folder_path, source, target, title="temp"):
    source_down, target_down, source_fpfh, target_fpfh, processed_source, processed_target, trans_init = prepare_dataset(
        voxel_size, source, target)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size, trans_init)
    print("result_ransac")
    print("inlier_rmse = ", result_ransac.inlier_rmse)
    print("fitness = ", result_ransac.fitness)

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac)

    result = save_transformation(folder_path, source, target, result_icp.transformation, title, True)

    return result, result_icp.transformation

def infer_upload_stitching_simulation(*, upload_model_id, wrap_model, depth_model, lum_model, opa_model, batch_name,
                            input_folder, start, stop, step, nb_points, radius, voxel_size ):
    date,ts = print_user_info()
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

        if not os.path.isdir(this_dir + '/ply'):
            os.mkdir(this_dir + '/ply')
        plyfolder = this_dir + '/ply'

        lum_model_predict = load_model(lum_model + '.h5')
        wrap_model_predict = load_model(wrap_model + '.h5')
        depth_model_predict = load_model(depth_model + '.h5')
        flist = []
        stitching_flist = []
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

            lum_input = lum_model_predict.predict(np.array([np.expand_dims(inp_img, -1)]))
            lum_input = lum_input.squeeze()
            lum = lum_input * 255
            mask = (lum > 25)
            
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
            if os.path.exists(dmap_saved) and os.path.exists(nndepth_save):
                depth_data = np.load(nndepth_save)
                dmap_data = np.load(dmap_saved)

                print("Loaded depth_data:", depth_data)
                print("Loaded dmap_data:", dmap_data)

                if depth_data.shape != dmap_data.shape:
                    print("The two arrays must have the same shape.")
                else:
                    depth = depth_data*3.93
                    dmap = dmap_data*3.93
                    error = np.abs(depth - dmap)
                    
                    error = error * 1000
                    error[error > 2000] = np.nan
                    
                    # Plotting
                    plt.imshow(error, cmap='hot', interpolation='nearest', vmin=0,vmax=.3)
                    # Add a color bar to interpret the values
                    cbar = plt.colorbar()
                    cbar.set_label('Error (micron)', rotation=270, labelpad=15)

                    # # Set the ticks and labels so that '0' is replaced by '100'
                    # ticks = np.linspace(0, 1, 11)  # Normalized tick positions from 0 to 1
                    # labels = (ticks * 1000).astype(int)  # Scale and convert to integer for display
                    # cbar.set_ticks(ticks)
                    # cbar.set_ticklabels(labels)

                    # Calculate the color bar ticks and labels
                    max_error = np.nanmax(error)  # Find the maximum error, ignoring NaN
                    ticks = np.arange(0, min(2001, max_error + 1), 100)  # Create ticks every 100 microns up to 2000 or max_error
                    cbar.set_ticks(ticks)  # Set the ticks on the color bar
                    cbar.set_ticklabels(ticks.astype(int))  # Set tick labels, converting to integer for readability


                    plt.title("Absolute Error between Depth Maps (micron)")
                    depth_comparison_path = nndepthfolder + '/' + str(i) + 'depth_comparison.png'
                    plt.ioff()
                    # Saving the image
                    plt.savefig(depth_comparison_path)
                    plt.clf()  # Clear the current figure
                    print("Image saved at:", depth_comparison_path)
            else:
                depth_comparison_path = None
                print("dmap or nndepth file missing")

            # depthpath = nndepthfolder + '/' + str(i) + '.png'
            # cv2.imwrite(depthpath, (256 * nndepth))
            # # cv2.imwrite(nndepthfolder + '/' + str(i) + '.png', (20 * nndepth))
            # np.save(wr_save, 256 * nndepth, allow_pickle=False)

            cv2.imwrite(sqimagefolder + '/' + str(i) + '.png', (img))
            generate_pointcloud(mask, inpfile2, 255*nndepth,
                                plyfolder + '/' + str(i) + '.ply')  # divide nndepth by 2 (128 instead of 256 multiply)

            plypath = plyfolder + '/' + str(i) + '.ply'
            flist.append({
                'point_cloud_file': plypath,
                'fringe_image': sqimagepath,
                'wrap_image': wrappath,
                'depth_image': depthpath,
                'depth_comparison_img': depth_comparison_path,
                'name': batch_name,
                'model': upload_model_id,
                'position': i

            })
        point_cloud_files = [item['point_cloud_file'] for item in flist]
        sm_responses = upload_simulation_ply(files=flist, batch_name=batch_name, model=upload_model_id,
                                          depth_model=depth_model, wrap_model=wrap_model, lum_model=lum_model, opa_model=opa_model)
        print("simulation_ply\n", sm_responses)

        # stitching
        print('stitching')
        if not os.path.isdir(this_dir + '/ply/stitching'):
            os.mkdir(this_dir + '/ply/stitching')

        plyfolder = this_dir + '/ply/stitching'
        start_time = time.time()
        source = o3d.io.read_point_cloud(point_cloud_files[0])
        output_file = f'{plyfolder}/external.ply'
        for i in range(0,len(point_cloud_files)-1):
            target = o3d.io.read_point_cloud(point_cloud_files[i+1])
            print(point_cloud_files[i+1])
            source, transformation = global_and_icp_registration(plyfolder, source, target)
        
        processed_source, outlier_index = source.remove_radius_outlier(nb_points=nb_points, radius=radius)    
        processed_source = processed_source.voxel_down_sample(voxel_size=voxel_size)
        o3d.io.write_point_cloud(output_file, processed_source)

        end_time = time.time()
        total_time = end_time - start_time
        total_time_minutes = total_time // 60  # Integer division to get the minutes
        total_time_seconds = total_time % 60  # The remaining seconds after calculating minutes
        print(f"Total time taken: {total_time_minutes} minutes {total_time_seconds:.2f} seconds for stitching")
        
        stitching_flist.append({
            'point_cloud_file': output_file,
            'fringe_image': sqimagepath,
            'wrap_image': wrappath,
            'depth_image':depthpath,
            'name': batch_name,
            'model': upload_model_id,
            'nb_points':str(nb_points),
            'radius':str(radius),
            'voxel_size':str(voxel_size),
            'total_minutes': str(total_time_minutes),
            'total_seconds': str(total_time_seconds),
            'input_folder': input_folder,
            'output_folder': plyfolder,
            'start':start,
            'stop':stop,
            'step':step
            
        })
        print("plyfolder:", plyfolder)
        print("stitching_flist", stitching_flist)
        st_response = upload_stitching_simulation_ply(
            files=stitching_flist, 
            batch_name=batch_name, 
            model=upload_model_id, 
            depth_model=depth_model, 
            wrap_model=wrap_model, 
            lum_model=lum_model,
            opa_model=opa_model
            )
        print("stitching\n", st_response)
        print('completed stitching')
        
        # creating/opening a file
        f = open("/home/samir/sal_github/docker/inference-dev-server/infer/save_error.log", "a")

        # writing in the file
        f.write(f"saving succesful at {this_dir}")

        # closing the file
        f.close()

    except Exception as Argument:

        # creating/opening a file
        f = open("/home/samir/sal_github/docker/inference-dev-server/infer/save_error.log", "a")

        # writing in the file
        f.write(str(Argument))

        # closing the file
        f.close()