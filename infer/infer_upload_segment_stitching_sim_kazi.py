import os
import time
from upload_segment_stitching import upload_seg_stitching_point_cloud
from upload_segment_stitching_element import upload_seg_stitching_element_simulation_ply
from plyfile import PlyData
from io import StringIO
import open3d as o3d
import shutil
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

def infer_upload_segment_stitching_simulation(*, model, batch_name, point_cloud_files, nb_points, radius, voxel_size):
    date,ts = print_user_info()
    this_dir = create_infer_nnwrap_output_dir(date, ts)

    # create_dictionary
    try:
        if not os.path.isdir(this_dir + '/segment'):
            os.mkdir(this_dir + '/segment')
        plyfolder = this_dir + '/segment'

        # stitching
        print('segment stitching')
       
        if not os.path.isdir(plyfolder + '/stitching'):
            os.mkdir(plyfolder + '/stitching')
        
        seg_ment_st_fl = plyfolder + '/stitching'
        
        if not os.path.isdir(plyfolder + '/element'):
            os.mkdir(plyfolder + '/element')

        elemetn_path = plyfolder + '/element'


        start_time = time.time()

        source = o3d.io.read_point_cloud(point_cloud_files[0]['output_folder'])
        output_file = f'{seg_ment_st_fl}/segment_external.ply'
        for i in range(0,len(point_cloud_files)-1):
            target = o3d.io.read_point_cloud(point_cloud_files[i+1]['output_folder'])
            source, transformation = global_and_icp_registration(seg_ment_st_fl, source, target)
        processed_source, outlier_index = source.remove_radius_outlier(nb_points=nb_points, radius=radius)    
        processed_source = processed_source.voxel_down_sample(voxel_size=voxel_size)
        o3d.io.write_point_cloud(output_file, processed_source)

        end_time = time.time()
        total_time = end_time - start_time
        total_time_minutes = total_time // 60  # Integer division to get the minutes
        total_time_seconds = total_time % 60  # The remaining seconds after calculating minutes
        print(f"Total time taken: {total_time_minutes} minutes {total_time_seconds:.2f} seconds for stitching")
        

        filename = output_file.split('/')[-1]

        data = {
            'model':model,
            'name': batch_name,
            'nb_points': nb_points,
            'radius': radius,
            'voxel_size': voxel_size,
            'total_minutes': total_time_minutes,
            'total_seconds': total_time_seconds
        }

        parent_id, response = upload_seg_stitching_point_cloud(data=data, filename=filename, file_path=output_file)
        if parent_id:
            # Generate segment_elements array based on the paths.
            segment_elements = []
            for index, path_info in enumerate(point_cloud_files):
                # path = os.path.join(path_info['output_folder'], 'external.ply')
                file_index = index + 1
                dst = f"{elemetn_path}/{file_index}.ply"
                shutil.copy2(path_info['output_folder'], dst)
                segment_elements.append({
                    'sim_segment_stitch_point_cloud': parent_id,
                    'name': path_info['name'],
                    'point_cloud_file': dst,
                    'input_folder': os.path.dirname(path_info['output_folder']),
                    'position': file_index
                })
            response = upload_seg_stitching_element_simulation_ply(files=segment_elements)
         
            if response:
                filename = '/home/samir/sal_github/docker/inference-dev-server/infer/segment_elements.json'
                with open(filename, 'w', encoding='utf-8') as file:
                    json.dump(segment_elements, file, ensure_ascii=False, indent=4)
        
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