import cv2
import urllib.request
import os
import glob
import time
import requests, zipfile, io
import moonrakerpy as moonpy
import time
import numpy as np



class WandPrinter:
    def __init__(self, wand_ip, dataset_size, base_path, flash, dias, x_displacement, y_displacement, z_displacement):
        self.wand_ip = wand_ip
        self.dataset_size = dataset_size
        self.base_path = base_path
        self.OLD_SCAN_METHOD = 1
        self.flash = flash
        self.dias = dias
        self.x_displacement = x_displacement
        self.y_displacement = y_displacement
        self.z_displacement = z_displacement

    def __clear_jpg(self, folder_path, name):
        try:
            files = glob.glob(f"{folder_path}/render{name}/*.jpg")
            for f in files:
                os.remove(f)
            return True
        except Exception as e:
            print(f"Error while deleting files: {e}")
            return False

    def __taking_picture(self, folder_path, name):
        self.img_size = 160
        self.img_zoom = 0.5
        fringeImageName = "sqimage"
        
        try:
            os.makedirs(f"{folder_path}/")
        except FileExistsError:
            pass
        try:
            os.makedirs(f"{folder_path}/render{name}")
        except FileExistsError:
            pass
        # try:
        #     print("Taking a picture - no light")
        #     urllib.request.urlretrieve(url, f"{folder_path}/render{name}/image8.jpg")
        #     im = cv2.imread(f"{folder_path}/render{name}/image8.jpg")
        #     cv2.imwrite(f"{folder_path}/render{name}/image8.png", im)
        # except Exception as e:
        #     print(f"Error capturing no light image: {e}")
        if self.OLD_SCAN_METHOD == 1:
            url = f"http://{self.wand_ip}:8080/pic/picture?size={self.img_size}&zoom={self.img_zoom}"
            try:
                print("Taking a picture - flash")
                urllib.request.urlretrieve(f"{url}&flash=0.5", f"{folder_path}/render{name}/flash.jpg")
                im = cv2.imread(f"{folder_path}/render{name}/flash.jpg")
                cv2.imwrite(f"{folder_path}/render{name}/flash.png", im)
            except Exception as e:
                print(f"Error capturing no flash image: {e}")

            try:
                print("Taking a picture - dias")
                urllib.request.urlretrieve(f"{url}&dias=1", f"{folder_path}/render{name}/{fringeImageName}.jpg")
                im = cv2.imread(f"{folder_path}/render{name}/{fringeImageName}.jpg")
                cv2.imwrite(f"{folder_path}/render{name}/{fringeImageName}.png", im)
            except Exception as e:
                print(f"Error capturing dias image: {e}")
        else:
            try:
                url = f"http://{self.wand_ip}:8080/pic/picture2?size={self.img_size}&zoom={self.img_zoom}&flash={self.flash}&dias={self.dias}"
                print("Capturing image pair - {url}")
                print(f"using url - {url}")
                r = requests.get(url)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(f"{folder_path}/render{name}/")

                im = cv2.imread(f"{folder_path}/render{name}/pic1.jpg")
                cv2.imwrite(f"{folder_path}/render{name}/{fringeImageName}.png",im)

                im = cv2.imread(f"{folder_path}/render{name}/pic2.jpg")
                cv2.imwrite(f"{folder_path}/render{name}/flash.png",im)

                self.__clear_jpg(folder_path, name)
            except Exception as e:
                print(f"Error capturing image pair: {e}")
        return self.__clear_jpg(folder_path, name)
        # return 1


    def collect_dataset(self):
        current_datetime = time.strftime("%d%m%y")
        top_folder = os.path.join(self.base_path, current_datetime)
        printer = moonpy.MoonrakerPrinter('http://klippy.local:4408')

        # Send arbitrary g-code commands
        printer.send_gcode('G28')

        try:
            os.makedirs(top_folder)
        except FileExistsError:
            pass
        time_stamp = time.strftime("%H%M%S")
        folder_path = os.path.join(top_folder, time_stamp)
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            pass
        for i in range(self.dataset_size):
            z_value = i * self.z_displacement
            x_value = i * self.x_displacement
            y_value = i * self.y_displacement
            printer.send_gcode(f'G1 X{x_value} Y{y_value} Z{z_value}')
            # wait for the motors to finish
            printer.send_gcode('M400')
            if not self.__taking_picture(folder_path, i):
                return False
            time.sleep(0.4)
        # turn motors off
        printer.send_gcode(f'G1 X0 Y0 Z50')
        printer.send_gcode('M18')

        # print("path", folder_path)
        return folder_path


if __name__ == "__main__":
    wand = WandPrinter(
        wand_ip="cm4-3.local", 
        dataset_size=1, 
        base_path="/home/samir/sal_github/docker/inference-dev-server", 
        flash=0.5, 
        dias=1,
        x_displacement=0,
        y_displacement=0,
        z_displacement=0)
    wand.collect_dataset()