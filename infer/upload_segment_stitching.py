import requests
import json
import logging
from requests.exceptions import RequestException

# Configure logging
# logging.basicConfig(filename='/home/samir/sal_github/docker/inference-dev-server/infer/segment_stitching_logs.txt', level=logging.INFO,
#                     format='%(asctime)s %(levelname)s:%(message)s')

logger = logging.getLogger(__name__)

SERVER_URL = "http://nndb.danbots.com:81/apis/v1/"
# SERVER_URL = 'http://127.0.0.1:8000/apis/v1/'

UPLOAD_SIM_SEGMENT_STITCHING_POINT_CLOUDS = SERVER_URL + "sim-segment-stitching-point-clouds/"


def upload_seg_stitching_point_cloud(data, filename, file_path):
    # Use a context manager to open the file
    with open(file_path, 'rb') as file:
        files = {
            'point_cloud_file': (filename, file, 'application/octet-stream'),
        }
        # Send the POST request within the context manager where the file is open
        try:
            response = requests.post(UPLOAD_SIM_SEGMENT_STITCHING_POINT_CLOUDS, data=data, files=files)
            # Attempt to decode JSON response
            try:
                response_data = response.json()
            except ValueError:  # Not JSON
                logging.error('The response is not in JSON format.')
                return None, f"The server response was not in JSON format: {response.text}"

            if response.status_code in (200, 201):
                logging.info(f"Upload successful, Response: {json.dumps(response_data, indent=4)}")
                return response_data.get('id'), response_data
            else:
                logging.error(
                    f"Failed to upload. Status Code: {response.status_code}, Response: {json.dumps(response_data, indent=4)}")
                return None, response_data
        except RequestException as e:
            logging.error(f"An error occurred while making the request: {e}")
            return None, f"An error occurred: {e}"
