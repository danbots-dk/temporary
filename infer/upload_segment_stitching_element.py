import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests


SERVER_URL = "http://nndb.danbots.com:81/apis/v1/"
# SERVER_URL = 'http://127.0.0.1:8000/apis/v1/'

UPLOAD_SIM_SEGMENT_ELEMENTS_STITCHING_PLY = SERVER_URL + "sim-segment-stitching-element-point-clouds/"


def upload_seg_stitching_element_simulation_ply(files):
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
            "sim_segment_stitch_point_cloud": (None, item.get("sim_segment_stitch_point_cloud", "")),
            "point_cloud_file": get_file_info(item.get("point_cloud_file", "")),
            "name": (None, item.get("name", "")),
            "input_folder": (None, item.get("input_folder", "")),
            "position": (None, item.get("position", "")),
        }
        file_list.append(file_dict)

    def stitching_simulation_element_data(data):
        try:
            with requests.Session() as session:
                response = session.post(UPLOAD_SIM_SEGMENT_ELEMENTS_STITCHING_PLY, files=data)
                response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
                return response.json()
        except requests.exceptions.RequestException as e:
            error_message = str(e.response.json())
            return {"error": error_message}

    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(stitching_simulation_element_data, file_list))
        # print(responses)
        end_time = time.time()
        total_time = end_time - start_time
        total_time_minutes = total_time // 60  # Integer division to get the minutes
        total_time_seconds = total_time % 60  # The remaining seconds after calculating minutes
        print(f"Total time taken for element: {total_time_minutes} minutes {total_time_seconds:.2f} seconds")
        return True
