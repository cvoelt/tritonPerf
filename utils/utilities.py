from sklearn.metrics import zero_one_loss
from torch import Tensor
from . import config
from io import FileIO
import logging
import pathlib
import os
import datetime
import typing
import pytz
import shutil

def enable_console_and_root_folder_logging() -> None:
    #logging to ./performance.log, this file will be overwritten!
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename='performance.log',filemode='w', level=logging.DEBUG)
    #logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)
    #turn off logging for HTTP-requests and jtop library
    logging.getLogger('requests').setLevel(logging.CRITICAL)
    try:
        logging.getLogger('jtop').setLevel(logging.CRITICAL)
    except:
        return

def enable_result_folder_logging(main_config: config.MainConfig) -> None:
    #logging to current result-folder
    temp_folder = main_config.get_results_dir().as_posix() + '/performance.log'
    fhandler = logging.FileHandler(temp_folder)
    fhandler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    fhandler.setLevel(logging.DEBUG)
    logging.getLogger("").addHandler(fhandler)

def store_config_file(main_config: config.MainConfig):
    config_location = main_config.get_results_dir().as_posix() + '/config.ini'
    config = main_config.get_config_file()
    with open(config_location,'w') as config_file:
        config.write(config_file)


def create_results_folder(results_dir: pathlib.Path) -> pathlib.Path:
    time_zone = pytz.timezone('Europe/Berlin')
    results_dir = pathlib.Path(results_dir.as_posix() + '/' + datetime.datetime.now(time_zone).strftime("%Y_%m_%d_%H_%M"))
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def get_jpg_files_from_results_folder(main_config: config.MainConfig) -> typing.List[pathlib.Path]:
    temp_folder: str = main_config.get_results_dir().as_posix()
    return_files: typing.list(pathlib.Path) = []

    for f in os.listdir(temp_folder):
        if os.path.splitext(f)[1].lower() in ('.png'):
            return_files.append(os.path.join(temp_folder, f))
    return return_files


def create_model_folder(model_name: str, instances: int, main_config: config.MainConfig, file_type ='onnx') -> pathlib.Path:
    root = get_model_path(model_name, instances, main_config, create = True)
    subdirs = os.listdir(root)
    if not subdirs: #no subdirectories available
        new_folder = root.as_posix() + '/1/'
        os.makedirs(new_folder, exist_ok=True)
        return pathlib.Path(new_folder)

    else:
        subfolders_numbers = list(map(lambda ele : int(ele) if ele.isdigit() else 1 , subdirs)) #convert all strings to int if possible
        current_max_folder = max(subfolders_numbers) #find the folder with the max number in the model folder
        if not os.path.exists(root.as_posix() + '/' + str(current_max_folder) + '/model.' + file_type): #if current max folder has no model, deploy here
            return pathlib.Path(root.as_posix() + '/' + str(current_max_folder))
        else: #create new folder with consecutive number
            new_folder_path = pathlib.Path(root.as_posix() + '/' + str(current_max_folder + 1)+ '/')
            new_folder_path.mkdir()
            return new_folder_path


def get_model_path(model_name: str, instances: int, main_config: config.MainConfig, create: bool = False) -> pathlib.Path:
    model_root_folder = pathlib.Path(main_config.get_model_dir().as_posix() + '/' + model_name + '_i' + str(instances))
    if not create and not model_root_folder.exists():
        return None
    model_root_folder.mkdir(exist_ok= True)
    return model_root_folder


def create_file(path:pathlib.Path, file_name: str = None) -> FileIO:
    if file_name is None:
        return open(path.as_posix(), 'w')
    else:
        return open(path.as_posix() + '/' + file_name, 'w')

#copy folder from zoo without config, model.onnx is only a symlink to save space, model_folder_suffix
# is a suffix not in the zoo-name but needed in the model folder, e.g. _cpu
def copy_from_zoo(model_name: str, instances: int, main_config: config.MainConfig, filetype: str, model_folder_suffix='') -> pathlib.Path:
    logging.info('Found the model ' + model_name + ' in the zoo. Copying it from there')
    zoo_model_folder = pathlib.Path(main_config.get_zoo_dir().as_posix() + '/' + model_name)
    zoo_model_file = pathlib.Path(zoo_model_folder.as_posix() + '/1/model.' + filetype)
    model_folder = create_model_folder(model_name + model_folder_suffix, instances, main_config).as_posix() 
    onnx_file_to_create = pathlib.Path(model_folder + '/model.' + filetype)
    os.link(zoo_model_file.as_posix(), onnx_file_to_create.as_posix())
    return zoo_model_folder


def model_with_lower_instances(model: str, instances: int, main_config: config.MainConfig) -> pathlib.Path:
    for instances in range(1,512):
        model_path = get_model_path(model, instances, main_config, create=False)
        if model_path is not None:
            return model_path
    return None


def copy_folder(from_path: pathlib.Path, to_path: pathlib.Path) -> None:
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path.as_posix(), to_path.as_posix())
        
def get_onnx_file(model_name: str, instances: int, main_config: config.MainConfig) -> pathlib.Path:
    
    root = get_model_path(model_name, instances, main_config, create = False)
    if  root is None:
        return None

    subdirs = os.listdir(root)
    if not subdirs: #no subdirectories available
        return None

    subfolders_numbers = list(map(lambda ele : int(ele) if ele.isdigit() else 1 , subdirs)) #convert all strings to int if possible
    current_max_folder = max(subfolders_numbers)
    model_file = pathlib.Path(root.as_posix() + '/' + str(current_max_folder) + '/model.onnx')

    if model_file.exists():
        return model_file

    return None


def results_folder_db_filename(main_config: config.MainConfig) -> pathlib.Path:
    return pathlib.Path(main_config.get_results_dir().as_posix() +  "/database.db")

def global_db_filename() -> pathlib.Path:
    return pathlib.Path(os.getcwd() + '/database.db')

def get_timestamp() -> str:
    time_zone = pytz.timezone('Europe/Berlin')
    return datetime.datetime.now(time_zone).strftime("%m_%d_%H_%M_%S")

def find_standard_deviation(perf_analyzer_output: str) -> int:
    perf_analyzer_output = str(perf_analyzer_output)
    begin = perf_analyzer_output.find('deviation')

    end = perf_analyzer_output.find('usec',begin)
    try:
        returnval = int(perf_analyzer_output[begin+10:end-1])
    except:
        returnval = 0
    return returnval