import typing
import configparser
import argparse
import pathlib
import os
import ipaddress
import socket
import logging
import math


def parse_cli_args() -> dict:
    __parser = argparse.ArgumentParser(description='')
    __parser.add_argument('-v','-debug','-debugging', help='enable debugging mode, create file: performance.log', dest = 'debug', action='store_true', default=True)
    __parser.add_argument('-c','-config','-config-file','-configuration-file', nargs='*', help='location of config-file', dest = 'config_dir', default='./utils/default_config.ini', type=str)
    __parser.add_argument('-temp-dir', '-temp', help='folder for the temp-csv-files', dest = 'results_dir', default= os.getcwd() + '/results/', type=str)
    __parser.add_argument('-model-dir', '-model-directory', nargs='*', help='location of the model-repository', dest='model_dir', default='/tmp/models', type=str)
    __parser.add_argument('-m', '-model', '-models', help='Specify a model', dest='models', required=False, default='resnet18' ,type=str)
    __parser.add_argument('-b','-batch', '-batch-sizes', nargs='*', help='Specify the number of concurrent batches', dest='batchsizes', required=False, default=1, type=int)
    __parser.add_argument('-trt','-trt-opt','-tensorrt-optimization', help='enable tensorRT optimization', dest = 'trt', default=False)
    __parser.add_argument('-cpu','-cpu-opt', help='run ONNX-files on CPU', dest = 'cpu', default=False)
    __parser.add_argument('-r', '-request-concurrency', nargs='*', help='number of outstanding inference requests', dest='request-con', default = 1, type=int)
    __parser.add_argument('-ip','-ip-address', '-host', help='Specify a remote ip address', dest='host', required=False, default='localhost', type=str)
    __parser.add_argument('-shape','-img-shape', help='Specify the shape of the image, eg. 3,520,520', dest='shape', required=False, default='3,224,224', type=str)
    
    return vars(__parser.parse_args())


class MainConfig():

    def __init__(self):
        kw_args: typing.Dict = parse_cli_args()
        self.store_cli_args(kw_args)
        self.parse_config_file(self.config_dir)


    def store_cli_args(self, kw_args: typing.Dict) -> None:

        self.config_dir: pathlib.Path = pathlib.Path(kw_args['config_dir'])
        self.results_dir: pathlib.Path = pathlib.Path(kw_args['results_dir'])
        self.triton_server_dir: str = pathlib.Path('/triton_server/bin/tritonserver')
        self.measurement_interval: int = 20 #measurement length in seconds
        self.server_only: bool = False
        self.client_only: bool = False
        self.gpu_metrics: bool = False
        self.warmup: bool = False
        localhost = socket.gethostbyname(socket.gethostname())
        self.gpu_metrics_ip: ipaddress.IPv4Address = ipaddress.IPv4Address(localhost)
        self.gpu_metrics_port: int = 8004
        self.model_dir: pathlib.Path = pathlib.Path(kw_args['model_dir'])
        self.zoo_dir: pathlib.Path = pathlib.Path('./zoo')
        self.models: list[str] = [kw_args['models']]
        self.batchsizes: list[int] = [kw_args['batchsizes']]
        self.grpc: bool = False
        self.trt: bool = kw_args['trt']
        self.chosen_precisions: list[str] = ['fp32'] #possible values: ['fp32','fp16', 'int8']
        self.trt_workspace_size: int = 8192      
        self.cpu: bool = kw_args['cpu']
        self.instances: list[int] = [2]
        self.concurrencies: list[int] = [2]
        self.measurement_configs: typing.Dict[int, MeasurementConfig] = {}
        self.triton_ip: ipaddress.IPv4Address = ipaddress.IPv4Address(localhost)
        self.triton_http_port: int = 8000
        self.triton_grps_port: int = 8001
        self.findmaxbatchsize: int = False

        
        str_shape_list = kw_args['shape'].split(",")  #shape as str
        self.shapes = list(map(int, str_shape_list))   #shape as int


    def parse_config_file(self,config_path:pathlib.Path) -> None:
        
        self.config = configparser.ConfigParser()
        config_file = open(config_path.as_posix(), 'r')
        self.config.read_file(config_file)

        #one boolean value
        sections = ['cpu','grpc','trt','server_only','client_only','gpu_metrics','warmup','findmaxbatchsize']
        for section in sections:
            if self.config.has_section(section):
                # get first value in section
                try:
                    value = self.config.getboolean(section,list(self.config[section].keys())[0])
                    setattr(self,section,value)
                except:
                    logging.error('section ' + section + ' has no values speficied, using default value ' + str(getattr(self,section)))

        ##one int value
        sections = ['measurement_interval', 'trt_workspace_size']
        for section in sections:
            if self.config.has_section(section):
                # get first value in section
                try:
                    value = int(list(self.config[section].values())[0])
                    setattr(self,section,value)
                except:
                    logging.error('section ' + section + ' has no values speficied, using default value ' + str(getattr(self,section)))

        ##one str value
        sections = ['triton_server_dir', 'zoo_dir', 'model_dir']
        for section in sections:
            if self.config.has_section(section):
                # get first value in section
                try:
                    value = pathlib.Path(list(self.config[section].values())[0])
                    setattr(self,section,value)
                except:
                    logging.error('section ' + section + ' has no values speficied, using default value ' + str(getattr(self,section)))

        #multiple str values
        sections = ['models']
        for section in sections:
            if self.config.has_section(section):
                try:
                    value = list(self.config[section].values())
                    setattr(self,section,value)
                except:
                    logging.error('section ' + section + ' has no values speficied, using default value' + str(getattr(self,section)))

        #multiple int values
        sections = ['batchsizes', 'instances', 'concurrencies']
        for section in sections:
            if self.config.has_section(section):
                # get first value in section
                try:
                    value = list(map(int, self.config[section].values()))
                    if not value:
                        raise Exception('input is empty!')
                    setattr(self,section,value)
                except:
                    logging.error('section ' + section + ' has no values speficied, using default value' + str(getattr(self,section)))

        #these inputs need special treatment
        if self.config.has_section('precisions'):
            accepted_values = ['fp16', 'int8']
            for input_value in list(self.config['precisions'].values()):
                if input_value in accepted_values:
                    self.chosen_precisions.append(input_value)
                else:
                    logging.error('precision-option: ' + input_value + ' not supported!')
        hostname = socket.gethostname()

        if self.config.has_section('host'):
            address = list(dict(self.config['host']).values())[0]
            if(address == 'localhost'):
                localhost = socket.gethostbyname(socket.gethostname())
                self.gpu_metrics_ip: ipaddress.IPv4Address = ipaddress.IPv4Address(localhost)
                self.triton_ip = socket.gethostbyname(socket.gethostname())
            else:
                self.gpu_metrics_ip = ipaddress.IPv4Address(list(self.config['host'].values())[0])
                self.triton_ip  = ipaddress.IPv4Address(list(self.config['host'].values())[0])

        if self.config.has_section('shapes'):
            list_of_shapes = self.config['shapes']
            self.shapes = []
            for comma_separated_shapes in list_of_shapes:
                self.shapes.append(list(map(int, list_of_shapes[comma_separated_shapes].split(','))))

    ''' if self.config.has_section('trt_shapes'):
        list_of_shapes = self.config['trt_shapes']
        self.shapes = []
        for comma_separated_shapes in list_of_shapes:
            self.shapes.append(list(map(int, list_of_shapes[comma_separated_shapes].split(','))))'''

    def get_config_file(self) -> configparser.ConfigParser:
        return self.config

    def get_measurement_interval(self) -> int:
        return self.measurement_interval

    def is_server_only(self) -> bool:
        return self.server_only

    def get_triton_server_dir(self) -> bool:
        return self.triton_server_dir

    def is_client_only(self) -> bool:
        return self.client_only

    def warmup_is_on(self) -> bool:
        return self.warmup

    def gpu_metrics_on(self) -> bool:
        return self.gpu_metrics

    def get_metrics_server_ip(self) -> ipaddress.IPv4Address:
        return self.gpu_metrics_ip

    def get_metrics_server_port(self) -> int:
        return self.gpu_metrics_port

    def get_num_measurements(self) -> int:
        number_measurements =  len(self.models) * (1 + self.cpu_is_on()*1 + self.trt_is_on()*len(self.chosen_precisions)) *len(self.instances) * len(self.concurrencies) * len(self.batchsizes) * len(self.get_str_shapes())
        return number_measurements

    def grpc_is_on(self) -> bool:
        return self.grpc
    
    def get_grpc_iterator(self) -> typing.List[bool]:
        if self.grpc_is_on():
            return [False,True]
        else:
            return [False]

    def trt_is_on(self) -> bool:
        return self.trt

    def get_chosen_trt_precisions(self) -> typing.List[str]:
        return self.chosen_precisions

    def only_fp32_precision_on(self) -> bool:
        if 'fp32' in self.chosen_precisions and len(self.chosen_precisions) == 1:
            return True
        return False

    def get_trt_workspace_size(self) -> int:
        return self.trt_workspace_size

    def get_trt_cpu_suffixes(self) -> typing.List[str]:
        returnlist = ['']
        if self.cpu_is_on():
            returnlist.append('_cpu')
        if self.trt_is_on():
            for precision in self.get_chosen_trt_precisions():
                returnlist.append('_trt_' + precision)
        return returnlist

    def cpu_is_on(self) -> bool:
        return self.cpu

    def get_instance_count(self) -> int:
        return self.instances

    def get_instance_suffixes(self) -> typing.Dict[int, str]:
        return {i_count : '_i' + str(i_count) for i_count in self.get_instance_count()}

    def get_concurrencies(self) -> int:
        return self.concurrencies

    def set_results_dir(self, results_dir: pathlib.Path) -> None:
        self.results_dir = results_dir

    def get_results_dir(self) -> pathlib.Path:
        return self.results_dir

    def get_csv_dir(self) -> pathlib.Path:
        return pathlib.Path(self.get_results_dir().as_posix() + '/measurement.csv')

    def get_model_dir(self) -> pathlib.Path:
        return self.model_dir

    def get_zoo_dir(self) -> pathlib.Path:
        return self.zoo_dir

    def add_models(self, models: typing.List[str]) -> None:
        self.models.extend(models)

    def get_models(self) -> dict:
        return self.models

    def get_num_of_models(self) -> int:
        return len(self.get_models())

    def get_batchsizes(self) -> typing.List[int]:
        return self.batchsizes

    def get_max_batchsize(self) -> int:
        return max(self.batchsizes)

    def find_max_batchsize_mode_on(self) -> bool:
        return self.findmaxbatchsize

    def get_triton_ip(self) -> ipaddress.IPv4Address:
        return ipaddress.IPv4Address(self.triton_ip)

    def get_triton_http_port(self) -> int:
        return self.triton_http_port

    def get_shapes(self) -> list:
        return self.shapes

    def get_str_shapes(self) -> list:
        returnlist: list[str] = []
        for shape in self.shapes:
            str_list_shapes = [str(int) for int in shape] #int values as string list
            str_shapes = ",".join(str_list_shapes) #string list to a single string
            returnlist.append(str_shapes) 
        return returnlist

    def add_measurement_config(self, execution_count: int, measurement_config) -> None:
        self.measurement_configs.update({execution_count:measurement_config})
    
    def get_measurement_config(self, execution_count: int):
        return self.measurement_configs[execution_count]


class MeasurementConfig():

    def __init__(self, main_config: MainConfig, execution_count: int, measurement_interval: int, model_name: str, grpc_on: bool, trt_cpu_suffix: str, instances: int, model_number: int = None, batchsize: int = None, shape: list = None, concurrency: int = None) -> None:
        self.execution_count: int = execution_count
        self.measurement_interval: int = measurement_interval        
        self.model_name: str = model_name
        self.model_nr: int = model_number
        self.grpc_on: bool = grpc_on
        self.tensorRT: bool = 'trt' in trt_cpu_suffix
        self.onnx_cpu: bool = 'cpu' in trt_cpu_suffix
        self.onnx_gpu: bool = not self.tensorRT and not self.onnx_cpu
        self.fp32: bool = self.onnx_cpu or self.onnx_gpu or 'fp32' in trt_cpu_suffix
        self.fp16: bool = 'fp16' in trt_cpu_suffix
        self.int8: bool = 'int8' in trt_cpu_suffix
        self.instances: int = instances
        self.batchsize: int = batchsize
        self.shape: str = str(shape)
        self.concurrency: int = concurrency



    def get_execution_count(self) -> int:
        return self.execution_count

    def get_measurement_interval(self) -> int:
        return self.measurement_interval

    def get_folder_name(self) -> str:
        folder_name = self.model_name
        if self.onnx_cpu:
            folder_name += '_cpu'
        if self.tensorRT:
            folder_name += '_trt_'
            if self.fp32:
                folder_name += 'fp32'
            elif self.fp16:
                folder_name += 'fp16'
            elif self.int8:
                folder_name += 'int8'
        folder_name += '_i' + str(self.instances)
        return folder_name

    def grpc_is_on(self) -> bool:
        return self.grpc_on
        
    def get_model_nr(self) -> int:
        return self.model_nr

    def get_batchsize(self) -> int:
        return self.batchsize

    def get_shape(self) -> str:
        return self.shape

    def get_concurrency(self) -> int:
        return  self.concurrency

    def add_value(self, name: str, value: typing.Any) -> None:
        setattr(self, name, value)

    def get_values(self) -> typing.Iterable:
        return vars(self).items()