from background_task import background
from infer_upload_sim_kazi import infer_upload_simulation
from infer_upload_wand_kazi import infer_upload_wand
from infer_upload_stitching_sim_kazi import infer_upload_stitching_simulation
from infer_upload_updated_stitching_sim_kazi import infer_upload_stitching_simulation_updated
from infer_upload_segment_stitching_sim_kazi import infer_upload_segment_stitching_simulation
from infer_upload_wand_3d_kazi import infer_upload_wand_3d
from requests.exceptions import HTTPError
from conf.settings import INFERENCE_SERVER
import logging
import requests

logger = logging.getLogger(__name__)

def post_model_file(infer_url, model_file_path, file_key):
    try:
        with open(model_file_path, 'rb') as file:
            files = [(file_key, (f'{file_key}.h5', file))]
            response = requests.post(infer_url, files=files)
            response.raise_for_status()  # Raises HTTPError for bad responses
            return True  # Indicates success
    except HTTPError as http_err:
        logging.error(f'HTTP error occurred during model file posting: {http_err}')
    except Exception as err:
        logging.error(f'Unexpected error occurred during model file posting: {err}')
    return False  # Indicates failure

def update_network_model(nndb_url, net_value, model_value, is_currently_used):
    try:
        data = {
            "net": net_value,
            "model": model_value,
            "is_currently_used": is_currently_used,
            "infer_server_ip": INFERENCE_SERVER
        }
        response = requests.post(nndb_url, data=data)
        response.raise_for_status()  # Raises HTTPError for bad responses
        return True
    except HTTPError as http_err:
        logging.error(f'HTTP error occurred during network model update: {http_err}')
    except Exception as err:
        logging.error(f'Unexpected error occurred during network model update: {err}')
    return False

@background
def simulation_model_task(uploaded_model, wrap_model, depth_model, lum_model, opa_model, batch_name, input_folder, start, stop, step):
    logging.info("Task 'simulation_model_task' is starting")

    infer_upload_simulation(
        upload_model_id=uploaded_model,
        wrap_model=wrap_model,
        depth_model=depth_model,
        lum_model=lum_model,
        opa_model=opa_model,
        batch_name=batch_name,
        input_folder=input_folder,
        start=int(start),
        stop=int(stop),
        step=int(step)
    )

    logging.info("Task 'simulation_model_task' has completed")

@background
def wand_model_task(uploaded_model, wrap_model, depth_model, lum_model, opa_model, batch_name, input_folder, start, stop, step, depth_comparison):
    logging.info("Task 'infer_upload_wand' is starting")
    
    infer_upload_wand(
        upload_model_id=uploaded_model,
        wrap_model=wrap_model,
        depth_model=depth_model,
        lum_model=lum_model,
        opa_model=opa_model,
        batch_name=batch_name,
        input_folder=input_folder,
        start=int(start),
        stop=int(stop),
        step=int(step),
        depth_comparison=depth_comparison
    )
    
    logging.info("Task 'infer_upload_wand' has completed")


@background
def stitching_simulation_model_task(uploaded_model, wrap_model, depth_model, lum_model, opa_model, batch_name, input_folder, start, stop, step, nb_points, radius, voxel_size):
    logging.info("Task 'infer_upload_stitching_simulation' is starting")
    
    infer_upload_stitching_simulation(
        upload_model_id=uploaded_model,
        wrap_model=wrap_model,
        depth_model=depth_model,
        lum_model=lum_model,
        opa_model=opa_model,
        batch_name=batch_name,
        input_folder=input_folder,
        start=int(start),
        stop=int(stop),
        step=int(step),
        nb_points=int(nb_points),
        radius=float(radius),
        voxel_size=float(voxel_size)
    )
    
    logging.info("Task 'infer_upload_stitching_simulation' has completed")



@background
def inference_upload_model_task(infer_url, nndb_url, model_path, file_key, net, model, is_currently_used):
    logging.info("Task 'inference_upload_model_task' is starting")
    if post_model_file(infer_url=infer_url, model_file_path=model_path, file_key=file_key):
        update_network_model(nndb_url=nndb_url, net_value=net, model_value=model, is_currently_used=is_currently_used)
        logging.info("Network model updated successfully after posting model file.")
    else:
        logging.error("Failed to post model file. Network model update skipped.")
    logging.info("Task 'inference_upload_model_task' has completed")


@background
def segment_stitching_simulation_model_task(model, batch_name, point_cloud_files, nb_points, radius, voxel_size):
    logging.info("Task 'segment_stitching_simulation_model_task' is starting")
    infer_upload_segment_stitching_simulation(
        model=model,
        batch_name=batch_name,                
        point_cloud_files=point_cloud_files,
        nb_points=int(nb_points),
        radius=float(radius),                
        voxel_size=float(voxel_size)
    )
    logging.info("Task 'segment_stitching_simulation_model_task' has completed")


@background
def wand_model_task_3d(upload_model_id, wrap_model, depth_model, lum_model, opa_model, batch_name, input_folder, start, stop, step, depth_comparison, x_displacement, y_displacement, z_displacement):
    logging.info("Task 'infer_upload_wand_3d' is starting")
    
    infer_upload_wand_3d(
        upload_model_id=upload_model_id,
        wrap_model=wrap_model,
        depth_model=depth_model,
        lum_model=lum_model,
        opa_model=opa_model,
        batch_name=batch_name,
        input_folder=input_folder,
        start=int(start),
        stop=int(stop),
        step=int(step),
        depth_comparison=depth_comparison,
        x_displacement=str(x_displacement),
        y_displacement=str(y_displacement),
        z_displacement=str(z_displacement)
    )
    
    logging.info("Task 'infer_upload_wand_3d' has completed")


@background
def updated_stitching_simulation_model_task(uploaded_model, wrap_model, depth_model, lum_model, opa_model, batch_name, input_folder, start, stop, step, nb_points, radius, voxel_size):
    logging.info("Task 'updated_stitching_simulation_model_task' is starting")
    
    infer_upload_stitching_simulation_updated(
        upload_model_id=uploaded_model,
        wrap_model=wrap_model,
        depth_model=depth_model,
        lum_model=lum_model,
        opa_model=opa_model,
        batch_name=batch_name,
        input_folder=input_folder,
        start=int(start),
        stop=int(stop),
        step=int(step),
        nb_points=int(nb_points),
        radius=float(radius),
        voxel_size=float(voxel_size)
    )
    
    logging.info("Task 'updated_stitching_simulation_model_task' has completed")