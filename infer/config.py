import os
original_wd = os.getcwd() # this should be /003_src
#print (original_wd)

#--------------------------------
# configuration
#--------------------------------
# os.chdir("../../") #ROOT

# https://github.com/open-mpi/ompi/issues/6535
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from _00_configuration._00_settings import *
from _00_configuration._01_variables import *

# functions
#--------------------------------
from _01_functions._00_helper_functions import *
from _01_functions._01_functions_images import *
# from _01_functions._02_functions_data_processing import * 
# from _01_functions._03_functions_network import *
# #from _01_functions._04_functions_plots import *
# from _01_functions._04_functions_plots_v2 import *
# from _01_functions._05_functions_validation import *
# from _01_functions._06_functions_custom_losses import *
# from _01_functions._07_functions_custom_callbacks import *
# from _01_functions._08_functions_database import *
# from _01_functions._09_function_check_dataset import *
# from _01_functions._10_function_custom_cmap import *
import random
os.chdir(original_wd)

## print statements
print (DASH)
print(f"code root directory: {CODE_ROOT}")
print(f"input directory: {inputFolder}")
print(f"output directory: {OUTPUT_DIR}")
print (DASH)
#print ("")
