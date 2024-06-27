import time
import numpy as np
import requests
import moonrakerpy as moonpy

# Initialize the printer
printer = moonpy.MoonrakerPrinter('http://klippy.local:4408')
url = 'http://klippy.local:4408/printer/objects/query?endstops'
response = requests.get(url)
if response.status_code == 200:
    data = response.json()  # Call the method to get JSON data
    print(data['result']['status']['endstops'])
else:
    raise ValueError(f"Failed to get endstop status: {response.content}")

# res = printer.send_gcode('M119')
# print(res.status_code)

# # Function to send G-code
# def send_gcode(command):
#     success = printer.send_gcode(command)
#     print("ddd", success)
#     if not success:
#         raise ValueError(f"Failed to send G-code command: {command}")

# # Function to get endstop status
# def get_endstop_status():
#     url = 'http://klippy.local:4408/printer/objects/query?endstops'
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()
#         # Assuming the response structure has 'result' and endstop states
#         return data['result']['status']['endstops']
#     else:
#         raise ValueError(f"Failed to get endstop status: {response.content}")

# # Check the status of the endstops to determine if the printer is already homed
# def is_homed():
#     endstop_status = get_endstop_status()
#     # Check if any endstop is triggered
#     for key, value in endstop_status.items():
#         if value == 'TRIGGERED':
#             return True
#     return False

# # Home the printer if it is not already homed
# if not is_homed():
#     send_gcode('G28')

# # Move Z axis incrementally and capture images
# z_displacements = 1
# img_number = 10  # Define img_number
# for i in np.arange(1, img_number + 1):
#     z_value = i * z_displacements
#     send_gcode(f'G1 X50 Y200 Z{z_value}')
#     # Wait for the motors to finish
#     send_gcode('M400')
#     print(f'Moved to Z={z_value}')
#     # Get picture from wand (simulated by sleep here)
#     time.sleep(0.4)

# # Turn motors off
# send_gcode('M18')
