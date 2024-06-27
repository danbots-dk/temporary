from _00_configuration._00_settings import *
from _00_configuration._01_variables import *
#--------------------------------
# CUDA check
#--------------------------------
def cudaOverview_tf():
    """
    just and overview of current status of cpu/gpu
    tf version
    """
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    if not gpu_devices:
        print ("gpu_devices not available, running on cpu")
        print (DASH)
        return "cpu"
    else:
        print ("gpu_devices available")
        numDevice = len(gpu_devices)
        deviceName = tf.test.gpu_device_name()
        print (f"Number of Devices: {numDevice}")
        print (f"\tcurrent GPU memory usage by tensors in bytes:{tf.config.experimental.get_memory_usage('GPU:0')}")
        #Assign cuda GPU located at location '0' to a variable
        cuda0 = tf.device('GPU:0')
        print (DASH)
        return cuda0

#--------------------------------
# SYSTEM and USER INFO
#--------------------------------
def print_user_info(get_all_info = False):
    """
    prints info on the system, env, user
    returns current date and time
    """
    date = get_date(sep = "")
    ts = get_timestamp()
    print (f"user: {username}")
    print (f"platform : {platform}")
    print (f"python version: {pVersion}")
    print (f"env. name: {envName}")
    print (f"date: {date}")
    print (f"time: {ts}")
    print (DASH)
    if get_all_info: 
        return date,ts,username,platform,pVersion,envName
    else: 
        return date,ts

# def create_log_dir(date,ts):
#     log_dir = os.path.join(f'{LOG_DIR}/{date}/{ts}/')
#     log_dir = log_dir.replace("\\", '/')
#     if not os.path.exists(log_dir):
#         Path(log_dir).mkdir(parents=True, exist_ok=True)
#     return log_dir

def create_intermediate_output_dir(date,ts):
    this_dir = os.path.join(f'{INTER_OUTPUT_DIR}/{date}/{ts}')
    this_dir = this_dir.replace("\\", '/')
    this_dir = this_dir.replace("//", '/')
    # if the path doesnt exist, it is created
    if not os.path.exists(this_dir):
        Path(this_dir).mkdir(parents=True, exist_ok=True)
    return this_dir

def create_final_output_dir(date,ts):
    this_dir = os.path.join(f'{OUTPUT_DIR}/{date}/{ts}')
    this_dir = this_dir.replace("\\", '/')
    this_dir = this_dir.replace("//", '/')
    # if the path doesnt exist, it is created
    if not os.path.exists(this_dir):
        Path(this_dir).mkdir(parents=True, exist_ok=True)
    return this_dir

def create_final_nn1_output_dir(date,ts):
    this_dir = os.path.join(f'{OUTPUT_DIR}/nn1/{date}/{ts}')
    this_dir = this_dir.replace("\\", '/')
    this_dir = this_dir.replace("//", '/')
    # if the path doesnt exist, it is created
    if not os.path.exists(this_dir):
        Path(this_dir).mkdir(parents=True, exist_ok=True)
    return this_dir

def create_final_nn2_output_dir(date,ts):
    this_dir = os.path.join(f'{OUTPUT_DIR}/nn2/{date}/{ts}')
    this_dir = this_dir.replace("\\", '/')
    this_dir = this_dir.replace("//", '/')
    # if the path doesnt exist, it is created
    if not os.path.exists(this_dir):
        Path(this_dir).mkdir(parents=True, exist_ok=True)
    return this_dir


def create_final_nn2k_output_dir(date,ts):
    this_dir = os.path.join(f'{OUTPUT_DIR}/nn2k/{date}/{ts}')
    this_dir = this_dir.replace("\\", '/')
    this_dir = this_dir.replace("//", '/')
    # if the path doesnt exist, it is created
    if not os.path.exists(this_dir):
        Path(this_dir).mkdir(parents=True, exist_ok=True)
    return this_dir



def create_infer_nnwrap_output_dir(date,ts):
    this_dir = os.path.join(f'{OUTPUT_DIR}/inferwrap/{date}/{ts}')

    # if the path doesnt exist, it is created
    if not os.path.exists(this_dir):
        Path(this_dir).mkdir(parents=True, exist_ok=True)
    return this_dir

#--------------------------------
# DECORATOR TO RUN FUN ONLY ONCE
#--------------------------------
# https://stackoverflow.com/questions/4103773/efficient-way-of-having-a-function-only-execute-once-in-a-loop
def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper   
    
#--------------------------------
# VARIOUS PRINTING 
#--------------------------------

def printif(content, printstat= True, n= 10):
    """
    prints the content if the printstat is true
    by default prints at most 10 lines (eg if a list is very long)
    the printstat can be given by a condition
    """
    if printstat:
        if isinstance(content, list):
            l = len(content)
            if not l:
                print ("empty")
            elif l< n:
                print (*content, sep = "\n")
            else:
                print (*[f"{i}) {c}" for i,c in enumerate(content[:int(n/2)])], sep = "\n")
                print ("...")
                print (*[f"{l-int(n/2)+i}) {c}" for i,c in enumerate(content[-int(n/2):])], sep = "\n")
        else:
            print (content)

@run_once
def printonceif(content, printstat= True, n= 10):
    """
    prints the content if the printstat is true
    by default prints at most 10 lines (eg if a list is very long)
    the printstat can be given by a condition
    """
    if printstat:
        if isinstance(content, list):
            l = len(content)
            if not l:
                print ("empty")
            elif l< n:
                print (*content, sep = "\n")
            else:
                print (*[f"{i}) {c}" for i,c in enumerate(content[:int(n/2)])], sep = "\n")
                print ("...")
                print (*[f"{l-int(n/2)+i}) {c}" for i,c in enumerate(content[-int(n/2):])], sep = "\n")
        else:
            print (content)

def list_all_in(directory, n = 5, printstat = True):
    """
    list all subdirectories and files in given directory
    """
    printif(f"dir:{directory}",printstat)
    dirpath, dirnames, filenames = next(os.walk(directory))
    if len(dirnames)>0 & printstat:
        print (f'DIRECTORIES\n{"-"*15}')
        print (*[">"+d for d in dirnames], sep= "\n")
    else:
        printif("no directories",printstat)
    print ()
    l = len(filenames)
    if printstat:
        if (l>0 and l <20) or n>=l:
            print (f'FILES: {l}\n{"-"*15}')
            print (*[f" {pad(i,minl = len(str(l)))}) "+f for i,f in enumerate(filenames)], sep= "\n")
        elif l == 0: 
            print ("no files")
        else:
            print (f'FILES: {l}\n{"-"*15}')
            print (*[f" {pad(i,minl = len(str(l)))}) "+f for i,f in enumerate(filenames[:n])], sep= "\n")
            print ("...")
            print (*[f" {pad(l-n+i,minl = len(str(l)))}) "+f for i,f in enumerate(filenames[-n:])], sep= "\n")
    return dirpath, dirnames, filenames

#--------------------------------
# TIME AND DATE
#--------------------------------

def get_timestamp(sep= "-"):
    """
    returns current time as a string in format %hours%minutes%seconds
    used for saving files and directories 
    """
    now = datetime.datetime.now()
    time_obj = now.strftime("%Hh%Mm%Ss")
    return time_obj

def get_date(sep= "-"):
    """
    returns current date as a string in format %year%month%day
    used for saving files and directories 
    """
    date_obj= datetime.date.today()
    date_obj = date_obj.strftime("%Y%m%d")+sep
    return date_obj

#--------------------------------
# HANDLE DIRECTORIES
#--------------------------------

def create_these_dir(list_dir: list):
    for this_dir in list_dir:
        if not os.path.exists(this_dir):
            Path(this_dir).mkdir(parents=True, exist_ok=True)
            print (f"Created dir: {this_dir}")

def create_subfolder_with_timestamp(folder):
    """
    creates a subfolder with timestamp
    """
    date = get_date()
    ts = get_timestamp()
    SAVE_TEMP = os.path.join(folder, f"{date}{ts}")
    Path(SAVE_TEMP).mkdir(parents=True, exist_ok=True)
    print (SAVE_TEMP)
    return SAVE_TEMP

def create_numbered_subdirectory(directory,datestamp):
    """
    counts the existing subdirs of given directory
    creates a new numbered one
    """
    numsubdir = len(os.listdir(directory))
    new_sub_label = "%03d"%(numsubdir+1)
    NEW_DIR= directory+f"{new_sub_label}-{datestamp[:-1]}/"
    if not os.path.exists(NEW_DIR):
        Path(NEW_DIR).mkdir(parents=True, exist_ok=True)
    print (f"new directory created at {NEW_DIR}")
    return NEW_DIR

def delete_empty(directory, printstat = True):
    """
    delete empty subdirectories in a given directory 
    not recursive
    returns a set with the deleted paths
    prints eventually while deleting
    """
    deleted = set()
    walk = list(os.walk(directory))
    for current_dir, _, _ in walk[::-1]:
        if len(os.listdir(current_dir)) == 0:
            printif(f"deleted path: {current_dir}", printstat)
            os.rmdir(current_dir)
            deleted.add(current_dir)
    return deleted


def delete_empty_r(directory, printstat = True):
    """
    delete empty subdirectories in a given directory 
    recursive
    returns a set with the deleted paths
    """
    deleted = set()
    for current_dir, subdirs, files in os.walk(directory, topdown=False):
        still_has_subdirs = any(
            subdir for subdir in subdirs
            if os.path.join(current_dir, subdir) not in deleted
        )
        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            deleted.add(current_dir)
            printif(f"deleted path: {current_dir}", printstat)

    return deleted

#--------------------------------
# HANDLE TEXTS
#--------------------------------

def remove_punctuation(s):
    """
    evokes unicorns
    """
    return s.translate(str.maketrans('', '', string.punctuation))

def pad(string,p = "0", minl= 2):
    """
    to pad the strings as wanted, eg
    input: "7",p= "i",minl  = 4
    ouptut: iii7
    """
    l = len(str(string))
    if l < minl: 
        return p*(minl-l)+str(string)
    else:
        return str(string)

# TODO either find where pad gets overwritten or use "pad_this" everywhere!
def pad_this(string,p = "0", minl= 2):
    """
    to pad the strings as wanted, eg
    input: "7",p= "i",minl  = 4
    ouptut: iii7
    """
    l = len(str(string))
    if l < minl: 
        return p*(minl-l)+str(string)
    else:
        return str(string)


#--------------------------------
## FORMATTING NAMES SAVINGS
#--------------------------------
def clean_string(mystring, lowercase = True):
    """
    modify as needed...
    replaces " " with "_" and " - " with "-"
    makes strings lowercase
    """
    mystring = mystring.replace("'", "")
    mystring = mystring.replace(", ", "_")
    mystring = mystring.replace("(", "-")
    mystring = mystring.replace(")", "-")
    mystring = mystring.replace("[", "_")
    mystring = mystring.replace("]", "_") 
    mystring.translate(str.maketrans('', '', string.punctuation))
    if lowercase == True:
        mystring = mystring.lower()
    return mystring 


def clean_string_to_num(mystring, lowercase = True):
    """
    modify as needed...
    replaces " " with "_" and " - " with "-"
    makes strings lowercase
    """
    mystring = mystring.replace("'", " ")
    mystring = mystring.replace(", ", " ")
    mystring = mystring.replace("(", " ")
    mystring = mystring.replace(")", " ")
    mystring = mystring.replace("[", " ")
    mystring = mystring.replace("]", " ") 
    mystring.translate(str.maketrans('', '', string.punctuation))
    if lowercase == True:
        mystring = mystring.lower()
    return mystring 

def list2string(input_columns):
    return clean_string(str(input_columns))


#--------------------------------
# FLATTEN LIST of LISTS
#--------------------------------
# https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
#def flatten(t):
    #return [item for sublist in t for item in sublist]
def list_flatten(t):
    return [item for sublist in t for item in sublist]


#--------------------------------
# CONVERT STRING TO BOOL
#--------------------------------
def string_to_bool(this_string):
    this_string = remove_punctuation(this_string)
    if this_string == "True" or this_string == "true" or this_string == True:
        return True
    elif this_string == "False" or this_string == "false" or this_string == False:
        return False
    else:
        print (f"\t{this_string}, type{type(this_string)} cannot be converted to a boolean")
        print (f"\treturn False")
        return False

#--------------------------------
# LOAD KEY:VALUE from dict if the key exists
#--------------------------------
def load_key_if_exists(dictionary, key,printstat = True):
    if key in dictionary.keys():
        printif('loaded {KEY}'.format(KEY = key),printstat )
        return dictionary[key]
    else:
        printif('missing {KEY}'.format(KEY = key),printstat )
        printif(f"\tset to None",printstat)
        return None

#--------------------------------------------------------------
# CHECK IF A KEY EXISTS IN A DICTIONARY, IF SO INCREASE THE NUMBERING
#-------------------------------------------------------------- 

def update_dict_with_key(this_key,this_value, this_dict):
    all_keys = list(this_dict.keys())
    count_occurrencies = sum(f'{this_key}' in s for s in all_keys)
    #print (count_occurrencies)
    if count_occurrencies == 0:
        index_label = ""
    else:
        index_label = pad(f"{count_occurrencies+1}",minl = 3)
        index_label = f"_{index_label}"
    this_dict[f"{this_key}{index_label}"] = this_value
    return this_dict

def update_dict_many_keys(new_dict, this_dict):
    for k,v in new_dict.items():
        this_dict = update_dict_with_key(k,v,this_dict)   
    return this_dict

#--------------------------------
# BUILD GIF
#--------------------------------
def build_gif(folder,
              title,
              search = "", 
              fps=55,
              recursive = True,
              delete_tempFiles = True,
              max_n_images = 200
             ):
    """
    titleGif = build_gif (folder,title,filenames,fps=55)
    folder = folder where are the current images to put togheter 
    title = name of gif
    filenames = list of names of images
    """
    
    SAVE_GIFS = "../004_data/gifs/"
    Path(SAVE_GIFS).mkdir(parents=True, exist_ok=True)

    filenames = sorted(glob.glob(folder + "/**/*" +f"*{search}*", recursive=recursive))
    max_limit = min(max_n_images,len(filenames))
    print (f"found {len(filenames)} images in folder : {folder}")
    print (f"the gif will be create using the first {max_limit} images")
    titleGif = os.path.join(SAVE_GIFS, f'{title}.gif')
    with imageio.get_writer(titleGif, mode='I',fps = fps) as writer:
        for i,filename in tqdm(enumerate(filenames[:max_limit])):
            try: 
                image = imageio.imread(filename)
                writer.append_data(image)
            except Exception as e:
                print (e)

            if delete_tempFiles: 
                try: os.remove(filename)
                except Exception as e:
                    print (e)
                    
        if delete_tempFiles:            
            deleted_folders = delete_empty_r(directory= "../004_data/figures",
                                                 printstat = True)
    return titleGif
#
# --------------------------------
# HANDLING DIFFERENT OS
#--------------------------------

def convert_read_path(original_path, 
                        os_platform,
                        ROOT_LINUX = "/scratch/arita/",
                        ROOT_WIN32 = "C:/Users/arita/Documents/GitHub/",
                        ROOT_WIN32_bis = "c:/Users/arita/Documents/GitHub/",
                        ):
    """
    to conver paths from linux on the server to tehhe local w32 machine
    and viceversa
    """
    
    if os_platform == "win32":
        if "scratch" in original_path:
            original_path = original_path.replace(ROOT_LINUX, ROOT_WIN32)
            return original_path.replace(ROOT_LINUX, ROOT_WIN32)
        else: return original_path

    # if running on windows
    elif os_platform =="linux":
        if "GitHub" in original_path:
            original_path = original_path.replace(ROOT_WIN32_bis, ROOT_LINUX)
            original_path = original_path.replace(ROOT_WIN32, ROOT_LINUX)
            return original_path
        else: 
            return original_path


print(f"Helper Functions import successful")
