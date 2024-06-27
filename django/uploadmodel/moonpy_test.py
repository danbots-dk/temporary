import moonrakerpy as moonpy
import time
import numpy as np

# Instantiate a `MoonrakerPrinter` object using the web/IP address of the target
# Moonraker installation.
printer = moonpy.MoonrakerPrinter('http://klippy.local:4408')
print(printer)

# Send arbitrary g-code commands

printer.send_gcode('G28')

z_diplacements = 1
img_number = 5
# move in global coordinates
for i in np.arange(1, img_number):
    print("dd")
    z_value = i * z_diplacements
    printer.send_gcode(f'G1 X0 Y0 Z{z_value}')
    # wait for the motors to finish
    printer.send_gcode('M400')

    print(i)
    # get picture from wand
    time.sleep(0.4)

# turn motors off
printer.send_gcode('M18')

