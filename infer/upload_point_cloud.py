import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

SERVER_URL = "http://nndb.danbots.com:81/apis/v1/"
UPLOAD_SIMULATION_PLY_API = SERVER_URL + "simulation-point-cloud"
UPLOAD_WAND_PLY_API = SERVER_URL + "wand-point-cloud"


def upload_simulation_ply(*, files, batch_name, model, depth_model, wrap_model, lum_model, opa_model):
    """
    Uploads simulation PLY files and associated images to the server and performs the simulation.

    Parameter
        files (list of dict): A list of dictionaries containing file information for each simulation.
            Each dictionary should have the following keys:
                - 'model': The model ID for the simulation.
                - 'name': The name of the simulation batch.
                - 'point_cloud_file': The path to the simulation PLY file.
                - 'fringe_image' (optional): The path to the fringe image file.
                - 'wrap_image' (optional): The path to the wrap image file.
                - 'depth_image' (optional): The path to the depth image file.
                - 'input_image' (optional): The path to the input image file.
        batch_name (str): The name of the simulation batch.
        model (int): The ID of the model to be used for the simulation.
        depth_model (int): The ID of the depth model to be used for the simulation.
        wrap_model (int): The ID of the warp model to be used for the simulation.
        lum_model (int): The ID of the lum model to be used for the simulation.
        opa_model (int): The ID of the opa model to be used for the simulation.


    Returns:
        bool: True if the simulation was successful, False otherwise.
    """

    def get_file_info(file_path=None):
        try:
            if file_path is not None and os.path.isfile(file_path):
                file_extension = os.path.splitext(os.path.basename(file_path))[1]
                file_object = open(file_path, "rb")
            else:
                return None, None
            return file_extension, file_object
        except Exception as e:
            raise ValueError(f"Error processing file '{file_path}': {str(e)}")

    file_list = []
    for item in files:
        file_dict = {
            "model": (None, item.get("model", "")),
            "name": (None, item.get("name", "")),
            "point_cloud_file": get_file_info(item.get("point_cloud_file", "")),
            "grand_truth_point_cloud_file": get_file_info(item.get("grand_truth_point_cloud_file", "")),
            "fringe_image": get_file_info(item.get("fringe_image", "")),
            "wrap_image": get_file_info(item.get("wrap_image", "")),
            "position": (None, item.get("position", "")),
        }

        if item.get("depth_image"):
            file_dict["depth_image"] = get_file_info(item.get("depth_image"))
        else:
            file_dict["depth_image"] = (None, None) 

        if item.get("input_image"):
            file_dict["input_image"] = get_file_info(item.get("input_image"))
        else:
            file_dict["input_image"] = (None, None)  

        if item.get("depth_comparison_img"):
            file_dict["depth_comparison_img"] = get_file_info(item.get("depth_comparison_img"))
        else:
            file_dict["depth_comparison_img"] = (None, None)  

        if item.get("hist_comparison_img"):
            file_dict["hist_comparison_img"] = get_file_info(item.get("hist_comparison_img"))
        else:
            file_dict["hist_comparison_img"] = (None, None)  


        file_list.append(file_dict)

    def simulation_data(data):
        try:
            with requests.Session() as session:
                response = session.post(UPLOAD_SIMULATION_PLY_API, files=data)
                response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
                return response.json()
        except requests.exceptions.RequestException as e:
            error_message = str(e.response.json())
            return {"error": error_message}

    def delete_previous_simulation_ply(name, model_id, d_model, w_model, l_model, o_model):
        UPLOAD_SIMULATION_PLY_API_Q = (f"{UPLOAD_SIMULATION_PLY_API}?batch_name={name}&model_id={model_id}"
                                       f"&depth_model={d_model}&wrap_model={w_model}&lum_model={l_model}&opa_model={o_model}")
        try:
            response = requests.delete(UPLOAD_SIMULATION_PLY_API_Q)
            if response.status_code == 200:
                print("Deletion successful.")
                return True
            else:
                print("Deletion failed.")
                return False
        except requests.exceptions.RequestException as e:
            print("Error:", e)
            return False

    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        delete_future = executor.submit(delete_previous_simulation_ply, batch_name, model, depth_model, wrap_model, lum_model, opa_model)
        delete_responses = delete_future.result()
        if delete_responses:
            responses = list(executor.map(simulation_data, file_list))
            print(responses)
            end_time = time.time()
            total_time = end_time - start_time
            total_time_minutes = total_time // 60  # Integer division to get the minutes
            total_time_seconds = total_time % 60  # The remaining seconds after calculating minutes
            print(f"Total time taken: {total_time_minutes} minutes {total_time_seconds:.2f} seconds")
            return True
        else:
            print("Delete operation failed. Aborting simulation_data.")
            return False


def upload_wand_ply(*, files, batch_name, model, depth_model, wrap_model, lum_model, opa_model):
    """
    Uploads Wand PLY files and associated images to the server and performs the simulation.

    Parameter
       files (list of dict): A list of dictionaries containing file information for each simulation.
           Each dictionary should have the following keys:
               - 'model': The model ID for the simulation.
               - 'name': The name of the simulation batch.
               - 'point_cloud_file': The path to the simulation PLY file.
               - 'fringe_image' (optional): The path to the fringe image file.
               - 'wrap_image' (optional): The path to the wrap image file.
               - 'depth_image' (optional): The path to the depth image file.
               - 'input_image' (optional): The path to the input image file.
       batch_name (str): The name of the simulation batch.
       model (int): The ID of the model to be used for the simulation.
       depth_model (int): The ID of the depth model to be used for the simulation.
       wrap_model (int): The ID of the warp model to be used for the simulation.
       lum_model (int): The ID of the lum model to be used for the simulation.
       opa_model (int): The ID of the opa model to be used for the simulation.

    Returns:
       bool: True if the simulation was successful, False otherwise.
    """

    def get_file_info(file_path=None):
        try:
            if file_path is not None and os.path.isfile(file_path):
                file_extension = os.path.splitext(os.path.basename(file_path))[1]
                file_object = open(file_path, "rb")
            else:
                return None, None
            return file_extension, file_object
        except Exception as e:
            raise ValueError(f"Error processing file '{file_path}': {str(e)}")

    file_list = []
    for item in files:
        file_dict = {
            "model": (None, item.get("model", "")),
            "name": (None, item.get("name", "")),
            "point_cloud_file": get_file_info(item.get("point_cloud_file", "")),
            "fringe_image": get_file_info(item.get("fringe_image", "")),
            "wrap_image": get_file_info(item.get("wrap_image", "")),
            "position": (None, item.get("position", "")),
            "x_displacement": (None, item.get("x_displacement", "")),
            "y_displacement": (None, item.get("y_displacement", "")),
            "z_displacement": (None, item.get("z_displacement", "")),
        }

        if item.get("depth_image"):
            file_dict["depth_image"] = get_file_info(item.get("depth_image"))
        else:
            file_dict["depth_image"] = (None, None) 

        if item.get("input_image"):
            file_dict["input_image"] = get_file_info(item.get("input_image"))
        else:
            file_dict["input_image"] = (None, None)  

        if item.get("depth_comparison_img"):
            file_dict["depth_comparison_img"] = get_file_info(item.get("depth_comparison_img"))
        else:
            file_dict["depth_comparison_img"] = (None, None)  
        
        if item.get("hist_comparison_img"):
            file_dict["hist_comparison_img"] = get_file_info(item.get("hist_comparison_img"))
        else:
            file_dict["hist_comparison_img"] = (None, None)  

        
        
        file_list.append(file_dict)

    def wand_data(data):
        try:
            with requests.Session() as session:
                response = session.post(UPLOAD_WAND_PLY_API, files=data)
                response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
                return response.json()
        except requests.exceptions.RequestException as e:
            error_message = str(e.response.json())
            return {"error": error_message}

    def delete_previous_wand_ply(name, model_id, d_model, w_model, l_model, o_model):
        UPLOAD_WAND_PLY_API_Q = (f"{UPLOAD_WAND_PLY_API}?batch_name={name}&model_id={model_id}"
                                 f"&depth_model={d_model}&wrap_model={w_model}&lum_model={l_model}&opa_model={o_model}")
        try:
            response = requests.delete(UPLOAD_WAND_PLY_API_Q)
            if response.status_code == 200:
                print("Deletion successful.")
                return True
            else:
                print("Deletion failed.")
                return False
        except requests.exceptions.RequestException as e:
            print("Error:", e)
            return False

    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        delete_future = executor.submit(delete_previous_wand_ply, batch_name, model, depth_model, wrap_model, lum_model, opa_model)
        delete_responses = delete_future.result()
        if delete_responses:
            responses = list(executor.map(wand_data, file_list))
            print(responses)
            end_time = time.time()
            total_time = end_time - start_time
            total_time_minutes = total_time // 60  # Integer division to get the minutes
            total_time_seconds = total_time % 60  # The remaining seconds after calculating minutes
            print(f"Total time taken: {total_time_minutes} minutes {total_time_seconds:.2f} seconds")
            return True
        else:
            print("Delete operation failed. Aborting simulation_data.")
            return False

