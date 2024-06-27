from _00_configuration._00_settings import *
#-----------------------------------------------------------------
## directories
#-----------------------------------------------------------------
#%%
ROOT_LINUX = "/scratch/arita/",
ROOT_WIN32 = "c:/Users/arita/Documents/GitHub/"


## src folder
ROOT = "/home/samir"

data_foder = f"/home/samir/Desktop/blender/pycode/bldev2/scans/30-91822"

## input dir
# dataset_name = 'chess/objsshuf'
dataset_name = '/1125/train/'# "160/trainermasked"
inputFolder = f"{data_foder}/{dataset_name}/" #EDIT
valset_name = '/0815/trval2'# "160/trainermasked"
valFolder = f"{data_foder}/{valset_name}/" #EDIT

## code root
CODE_ROOT = f"{ROOT}/MSThesis_2021-11-15" #EDIT with the latest MSThesis code

## 003_src 
SRC_FOLDER = f"{CODE_ROOT}/003_src"

print (SRC_FOLDER)

#%%
## config folder
CONFIG_FOLDER = os.path.join(CODE_ROOT, "003_src/_00_configuration/")
CONFIG_FOLDER = CONFIG_FOLDER.replace("\\", '/') # w32 stuff... 

FUNCTIONS_FOLDER = os.path.join(CODE_ROOT, "003_src/_01_functions/")
FUNCTIONS_FOLDER = FUNCTIONS_FOLDER.replace("\\", '/')

## intermediateoutput dir
## to store models and runs from IM to WRAPPED IM
INTER_OUTPUT_DIR = os.path.join(CODE_ROOT,'004_intermediate_output/')
INTER_OUTPUT_DIR = INTER_OUTPUT_DIR.replace("\\", '/')
if not os.path.exists(INTER_OUTPUT_DIR):
    Path(INTER_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

## output dir
## to store models and runs from WRAPPED IM to K-BANDS
OUTPUT_DIR = os.path.join(CODE_ROOT,'005_model_output/')
OUTPUT_DIR = "/danbots/data2/data/models/"
OUTPUT_DIR = OUTPUT_DIR.replace("\\", '/')
if not os.path.exists(OUTPUT_DIR):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

OTHERFILES_DIR = os.path.join(CODE_ROOT,"other_files/")
OTHERFILES_DIR = OTHERFILES_DIR.replace("\\", '/')
if not os.path.exists(OTHERFILES_DIR):
    Path(OTHERFILES_DIR).mkdir(parents=True, exist_ok=True)
    
PLOTLY_DIR = os.path.join(CODE_ROOT,"other_files/plotly")
PLOTLY_DIR = PLOTLY_DIR.replace("\\", '/')
if not os.path.exists(PLOTLY_DIR):
    Path(PLOTLY_DIR).mkdir(parents=True, exist_ok=True)
    
FIG_DIR = os.path.join(CODE_ROOT,"other_files/figures_compare")
FIG_DIR = FIG_DIR.replace("\\", '/')
if not os.path.exists(FIG_DIR):
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    
PKL_DIR = os.path.join(CODE_ROOT,"other_files/pickle_files")
PKL_DIR = PKL_DIR.replace("\\", '/')
if not os.path.exists(PKL_DIR):
    Path(PKL_DIR).mkdir(parents=True, exist_ok=True)

#-----------------------------------------------------------------
## global variables
#-----------------------------------------------------------------
os_platform = platform

#https://stackoverflow.com/questions/48658204/tensorflow-failed-call-to-cuinit-cuda-error-no-device 
# check available devices with gpustat
#os.environ['CUDA_VISIBLE_DEVICES'] = "3"
#for k, v in os.environ.items():
   # print (k,v)

if os_platform == "linux":
    username = 'samir' #os.environ['USER']
elif os_platform =="win32":
    username = os.environ['USERNAME']
pVersion = sys.version
try:
    envName = os.path.split(os.getenv('CONDA_PREFIX'))[1]
except Exception as e:
    envName = "Samir_env"

list_yes = ["yes","y","yeah","yeps","true"]
list_no = ["no","n","nope","false"]

dash = "-"*100
DASH = "="*100

all_subfolders = glob.glob(inputFolder+'/render*')
IMAGECOUNT = len(all_subfolders)

try: 
    hostname = socket.gethostname() # crius, theia, ocenaus ...
except Exception as e:
    hostname ="NA"
    
    
# custom colorbar
hex_list = [
             '#000000', # black 
             #'#dbdb8d', 
             '#1f77b4', #blue
             '#1f77b4', #blue
    
             '#ff7f0e',
             '#2ca02c',
             '#d62728', 
             '#9467bd', 
             '#8c564b',
             '#bcbd22',            
             '#e377c2',
             '#7f7f7f',
             #'#17becf',
             '#17becf',
             '#17becf',                  
             #'#9edae5',
             '#9edae5',
             '#9edae5',
             #'#ffffff',
             '#ffffff', # white
             '#ffffff', # white
            ] 

## center_focused
hex_list_2 = [
             '#8c564b', #brown
             '#8c564b', #brown
             '#8c564b', #brown
             
             '#bcbd22', # green
             '#bcbd22', # green
             '#bcbd22', # green
    
             '#9467bd', # violet    
             '#9467bd', # violet
             '#9467bd', # violet   
    
             '#1f77b4', #blue
             '#1f77b4', #blue
    
             '#000000', # black    
             '#ffffff', # white
             '#9edae5', # light blue
             '#2ca02c', # green

    
             '#e377c2', # pink 
             '#e377c2', # pink 
    
             '#7f7f7f', # grey
             '#7f7f7f', # grey

             '#17becf', #bright light blue
             '#17becf', #bright light blue

             '#d62728', #red
             '#d62728', #red

             '#ff7f0e', #orange
             '#ff7f0e', #orange

            ] 
print ("Variables import successful")
